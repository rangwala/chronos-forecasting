# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chronos-2 native vLLM implementation — progressive layer replacement."""

import copy
from typing import ClassVar, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear, QKVParallelLinear, RowParallelLinear,
)
from vllm.model_executor.layers.pooler import IdentityPooler
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig, MultiModalInputs, MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import DictEmbeddingItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder, BaseMultiModalProcessor, BaseProcessingInfo,
)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import IsAttentionFree, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import attn_type

# ── Multimodal plumbing (unchanged from adapter) ──────────────────────

_FIELD_NAMES = {"context", "group_ids", "future_covariates", "num_output_patches"}


def _field_config_factory(hf_inputs):
    return {k: MultiModalFieldConfig.batched("image") for k in hf_inputs if k in _FIELD_NAMES}


class Chronos2DataParser(MultiModalDataParser):
    def _parse_image_data(self, data):
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data, modality="image", required_fields={"context"},
                fields_factory=lambda d: _field_config_factory(d),
            )
        return super()._parse_image_data(data)

    def parse_mm_data(self, mm_data):
        if "image" not in mm_data:
            mm_data = {"image": mm_data}
        return super().parse_mm_data(mm_data)


class Chronos2ProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self):
        return Chronos2DataParser(expected_hidden_size=self._get_expected_hidden_size())

    def get_supported_mm_limits(self):
        return {"image": None}


class Chronos2DummyInputBuilder(BaseDummyInputsBuilder[Chronos2ProcessingInfo]):
    def get_dummy_text(self, mm_counts):
        return ""

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None):
        return {"context": torch.randn(64)}


class Chronos2Processor(BaseMultiModalProcessor[Chronos2ProcessingInfo]):
    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs, *, is_shared=True):
        return _field_config_factory(hf_inputs)

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return []

    def apply(self, prompt, mm_items, hf_processor_mm_kwargs,
              tokenization_kwargs=None, mm_uuids=None):
        mm_hashes = self._hash_mm_items(
            mm_items, hf_processor_mm_kwargs, tokenization_kwargs or {}, mm_uuids=mm_uuids)
        _, passthrough_data = self._get_hf_mm_data(mm_items)
        mm_processed = BatchFeature(
            {k: torch.as_tensor(v).unsqueeze(0) for k, v in passthrough_data.items()},
            tensor_type="pt",
        )
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed,
            self._get_mm_fields_config(mm_processed, hf_processor_mm_kwargs, is_shared=False),
        )
        return MultiModalInputs(
            type="multimodal", prompt_token_ids=[0],
            mm_kwargs=mm_kwargs, mm_hashes=mm_hashes,
            mm_placeholders={"image": [PlaceholderRange(offset=0, length=0)]},
        )


# ── Native layers ─────────────────────────────────────────────────────

class NativeRoPE(nn.Module):
    """RoPE matching Chronos-2's implementation (Llama-style, neox)."""

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ pos.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)

    @staticmethod
    def apply(q, k, cos, sin, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class NativeMHA(nn.Module):
    """Multi-head attention with vLLM parallel linears."""

    def __init__(self, d_model, d_kv, num_heads, dropout, rope_theta, use_rope=True,
                 quant_config=None, prefix=""):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.n_heads = num_heads
        self.dropout = dropout
        self.inner_dim = num_heads * d_kv

        self.qkv = QKVParallelLinear(
            hidden_size=d_model, head_size=d_kv,
            total_num_heads=num_heads, total_num_kv_heads=num_heads,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.qkv")
        self.o = RowParallelLinear(
            input_size=self.inner_dim, output_size=d_model,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.o")

        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = NativeRoPE(dim=d_kv, base=rope_theta)

    def forward(self, hidden_states, mask, position_ids=None):
        B, S, _ = hidden_states.shape
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.split([self.inner_dim, self.inner_dim, self.inner_dim], dim=-1)
        q = q.view(B, S, self.n_heads, self.d_kv).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.d_kv).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.d_kv).transpose(1, 2)

        if self.use_rope and position_ids is not None:
            cos, sin = self.rope_embed(v, position_ids)
            q, k = NativeRoPE.apply(q, k, cos, sin)

        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0,
        )
        out = out.transpose(1, 2).reshape(B, S, self.inner_dim)
        out, _ = self.o(out)
        return out


class NativeTimeSelfAttention(nn.Module):
    def __init__(self, d_model, d_kv, num_heads, dropout, rope_theta, eps,
                 quant_config=None, prefix=""):
        super().__init__()
        self.self_attention = NativeMHA(d_model, d_kv, num_heads, dropout, rope_theta, use_rope=True,
                                        quant_config=quant_config, prefix=f"{prefix}.self_attention")
        self.layer_norm = RMSNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask, position_ids):
        normed = self.layer_norm(hidden_states)
        attn_out = self.self_attention(normed, mask=attention_mask, position_ids=position_ids)
        return hidden_states + self.dropout(attn_out)


class NativeGroupSelfAttention(nn.Module):
    def __init__(self, d_model, d_kv, num_heads, dropout, rope_theta, eps,
                 quant_config=None, prefix=""):
        super().__init__()
        self.self_attention = NativeMHA(d_model, d_kv, num_heads, dropout, rope_theta, use_rope=False,
                                        quant_config=quant_config, prefix=f"{prefix}.self_attention")
        self.layer_norm = RMSNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        normed = self.layer_norm(hidden_states)
        attn_out = self.self_attention(normed, mask=attention_mask)
        hidden_states = hidden_states + self.dropout(attn_out)
        return hidden_states.transpose(0, 1).contiguous()


class NativeFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, eps, quant_config=None, prefix=""):
        super().__init__()
        self.wi = ColumnParallelLinear(
            input_size=d_model, output_size=d_ff, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wi")
        self.wo = RowParallelLinear(
            input_size=d_ff, output_size=d_model, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.wo")
        self.act = nn.ReLU()
        self.layer_norm = RMSNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        normed = self.layer_norm(hidden_states)
        h, _ = self.wi(normed)
        h = self.act(h)
        h = self.dropout(h)
        h, _ = self.wo(h)
        return hidden_states + self.dropout(h)


class NativeEncoderBlock(nn.Module):
    def __init__(self, d_model, d_kv, d_ff, num_heads, dropout, rope_theta, eps,
                 quant_config=None, prefix=""):
        super().__init__()
        self.time_attn = NativeTimeSelfAttention(d_model, d_kv, num_heads, dropout, rope_theta, eps,
                                                  quant_config=quant_config, prefix=f"{prefix}.time_attn")
        self.group_attn = NativeGroupSelfAttention(d_model, d_kv, num_heads, dropout, rope_theta, eps,
                                                    quant_config=quant_config, prefix=f"{prefix}.group_attn")
        self.ff = NativeFeedForward(d_model, d_ff, dropout, eps,
                                     quant_config=quant_config, prefix=f"{prefix}.ff")

    def forward(self, hidden_states, position_ids, attention_mask, group_time_mask):
        hidden_states = self.time_attn(hidden_states, attention_mask, position_ids)
        hidden_states = self.group_attn(hidden_states, group_time_mask)
        hidden_states = self.ff(hidden_states)
        return hidden_states


class NativeEncoder(nn.Module):
    def __init__(self, num_layers, d_model, d_kv, d_ff, num_heads, dropout, rope_theta, eps,
                 quant_config=None, prefix=""):
        super().__init__()
        self.block = nn.ModuleList([
            NativeEncoderBlock(d_model, d_kv, d_ff, num_heads, dropout, rope_theta, eps,
                               quant_config=quant_config, prefix=f"{prefix}.block.{i}")
            for i in range(num_layers)
        ])
        self.final_layer_norm = RMSNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _expand_time_mask(mask, dtype):
        return (1.0 - mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min

    @staticmethod
    def _build_group_time_mask(group_ids, attention_mask, dtype):
        group_mask = group_ids[:, None] == group_ids[None, :]
        gtm = torch.einsum("qb, bt -> qbt", group_mask.to(dtype), attention_mask.to(dtype))
        gtm = gtm.permute(2, 0, 1).unsqueeze(1)  # q b t -> t 1 q b
        return (1.0 - gtm) * torch.finfo(dtype).min

    def forward(self, inputs_embeds, group_ids, attention_mask=None, position_ids=None):
        B, S = inputs_embeds.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(S, device=inputs_embeds.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(B, S, device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        ext_mask = self._expand_time_mask(attention_mask, inputs_embeds.dtype)
        gtm = self._build_group_time_mask(group_ids, attention_mask, inputs_embeds.dtype)

        h = self.dropout(inputs_embeds)
        for block in self.block:
            h = block(h, position_ids, ext_mask, gtm)
        h = self.final_layer_norm(h)
        return self.dropout(h)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = nn.ReLU()
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.output_layer(self.dropout(self.act(self.hidden_layer(x)))) + self.residual_layer(x)


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5, use_arcsinh=False):
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(self, x, loc_scale=None):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale
        scaled_x = (x - loc) / scale
        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)
        return scaled_x.to(orig_dtype), (loc, scale)

    def inverse(self, x, loc_scale):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale
        if self.use_arcsinh:
            x = torch.sinh(x)
        return (x * scale + loc).to(orig_dtype)


class Patch(nn.Module):
    def __init__(self, patch_size, patch_stride):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x):
        length = x.shape[-1]
        if length % self.patch_size != 0:
            pad_size = self.patch_size - (length % self.patch_size)
            padding = torch.full((*x.shape[:-1], pad_size), float('nan'), dtype=x.dtype, device=x.device)
            x = torch.cat((padding, x), dim=-1)
        return x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)


# ── Top-level model ───────────────────────────────────────────────────

@attn_type("attention_free")
@MULTIMODAL_REGISTRY.register_processor(
    Chronos2Processor, info=Chronos2ProcessingInfo, dummy_inputs=Chronos2DummyInputBuilder)
class Chronos2ForForecasting(nn.Module, IsAttentionFree, SupportsMultiModal):
    supports_multimodal_raw_input_only: ClassVar[bool] = True
    is_pooling_model: ClassVar[Literal[True]] = True
    is_attention_free: ClassVar[Literal[True]] = True

    @classmethod
    def get_placeholder_str(cls, modality, i):
        if modality.startswith("image"):
            return None
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = vllm_config.model_config.hf_config.to_dict()
        cc = cfg["chronos_config"]
        quant_config = vllm_config.quant_config

        self.d_model = cfg["d_model"]
        d_kv = cfg["d_kv"]
        d_ff = cfg["d_ff"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        dropout = cfg["dropout_rate"]
        eps = cfg["layer_norm_epsilon"]
        rope_theta = cfg.get("rope_theta", 10000.0)

        # Chronos config
        self.input_patch_size = cc["input_patch_size"]
        self.input_patch_stride = cc.get("input_patch_stride", self.input_patch_size)
        self.output_patch_size = cc["output_patch_size"]
        self.context_length = cc["context_length"]
        self.use_reg_token = cc.get("use_reg_token", False)
        self.use_arcsinh = cc.get("use_arcsinh", False)
        self.time_encoding_scale = cc.get("time_encoding_scale", self.context_length)
        self.quantiles = cc["quantiles"]
        self.num_quantiles = len(self.quantiles)
        self.model_output_patch_size = self.output_patch_size

        quantiles_t = torch.tensor(self.quantiles, dtype=torch.float32)
        self.register_buffer("quantiles_buf", quantiles_t, persistent=False)

        vocab_size = 2 if self.use_reg_token else 1
        self.reg_token_id = 1 if self.use_reg_token else None
        self.shared = nn.Embedding(vocab_size, self.d_model)

        self.input_patch_embedding = ResidualBlock(
            in_dim=self.input_patch_size * 3, h_dim=d_ff, out_dim=self.d_model, dropout_p=dropout)
        self.output_patch_embedding = ResidualBlock(
            in_dim=self.d_model, h_dim=d_ff,
            out_dim=self.num_quantiles * self.output_patch_size, dropout_p=dropout)

        self.patch = Patch(self.input_patch_size, self.input_patch_stride)
        self.instance_norm = InstanceNorm(use_arcsinh=self.use_arcsinh)

        self.encoder = NativeEncoder(
            num_layers, self.d_model, d_kv, d_ff, num_heads, dropout, rope_theta, eps,
            quant_config=quant_config, prefix=f"{prefix}.encoder" if prefix else "encoder")

        self.pooler = IdentityPooler()

    # ── Preprocessing (reimplemented from Chronos2Model) ──────────

    def _prepare_patched_context(self, context, context_mask=None):
        context_mask = (
            context_mask.to(context.dtype)
            if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )
        B, L = context.shape
        if L > self.context_length:
            context = context[..., -self.context_length:]
            context_mask = context_mask[..., -self.context_length:]

        # scaling in float32, then cast to model dtype
        context, loc_scale = self.instance_norm(context)
        dtype = self.shared.weight.dtype
        context = context.to(dtype)
        context_mask = context_mask.to(dtype)

        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        attention_mask = patched_mask.sum(dim=-1) > 0
        num_context_patches = attention_mask.shape[-1]

        final_len = num_context_patches * self.input_patch_size
        time_enc = torch.arange(-final_len, 0, device=context.device, dtype=torch.float32)
        time_enc = time_enc.view(num_context_patches, self.input_patch_size).unsqueeze(0).expand(B, -1, -1)
        time_enc = time_enc.div(self.time_encoding_scale).to(context.dtype)

        patched_context = torch.cat([time_enc, patched_context, patched_mask], dim=-1)
        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(self, future_covariates, future_covariates_mask, loc_scale, num_output_patches, B):
        ps = self.output_patch_size
        dtype = self.shared.weight.dtype
        device = self.shared.weight.device

        if future_covariates is not None:
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = future_covariates.to(dtype)
            if future_covariates_mask is None:
                future_covariates_mask = (~torch.isnan(future_covariates)).to(dtype)
            future_covariates = torch.where(future_covariates_mask > 0, future_covariates, 0.0)
            if num_output_patches * ps > future_covariates.shape[-1]:
                pad = num_output_patches * ps - future_covariates.shape[-1]
                future_covariates = nn.functional.pad(future_covariates, (0, pad))
                future_covariates_mask = nn.functional.pad(future_covariates_mask, (0, pad))
            pfc = future_covariates.view(B, num_output_patches, ps)
            pfm = future_covariates_mask.view(B, num_output_patches, ps)
        else:
            pfc = torch.zeros(B, num_output_patches, ps, device=device, dtype=dtype)
            pfm = torch.zeros(B, num_output_patches, ps, device=device, dtype=dtype)

        final_len = num_output_patches * ps
        fte = torch.arange(0, final_len, device=device, dtype=torch.float32)
        fte = fte.view(num_output_patches, ps).unsqueeze(0).expand(B, -1, -1)
        fte = fte.div(self.time_encoding_scale).to(dtype)

        return torch.cat([fte, pfc, pfm], dim=-1), pfm

    # ── Forward ───────────────────────────────────────────────────

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, **kw):
        return torch.zeros(input_ids.shape[0], self.d_model,
                           device=input_ids.device, dtype=torch.float16)

    @staticmethod
    def _squeeze(t):
        if t is not None and t.ndim >= 2 and t.shape[0] == 1:
            return t.squeeze(0)
        return t

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        context = kwargs.get("context")
        if context is None:
            return torch.zeros(1, 1, device=positions.device)

        context = self._squeeze(context).to(dtype=torch.float32)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        B = context.shape[0]

        nop_t = kwargs.get("num_output_patches")
        if nop_t is not None:
            nop = int(self._squeeze(nop_t).flatten()[0].item())
        else:
            nop = 1

        gids = kwargs.get("group_ids")
        if gids is not None:
            gids = self._squeeze(gids).to(dtype=torch.long)

        fc = kwargs.get("future_covariates")
        if fc is not None:
            fc = self._squeeze(fc).to(dtype=torch.float32)

        # Preprocessing
        patched_ctx, attn_mask, loc_scale = self._prepare_patched_context(context)
        num_ctx_patches = attn_mask.shape[-1]

        input_embeds = self.input_patch_embedding(patched_ctx)

        if self.use_reg_token:
            reg_ids = torch.full((B, 1), self.reg_token_id, device=input_embeds.device)
            reg_embeds = self.shared(reg_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attn_mask = torch.cat([attn_mask.to(input_embeds.dtype),
                                   torch.ones(B, 1, device=input_embeds.device, dtype=input_embeds.dtype)], dim=-1)

        patched_future, pfcm = self._prepare_patched_future(fc, None, loc_scale, nop, B)
        future_embeds = self.input_patch_embedding(patched_future)
        future_mask = torch.ones(B, nop, dtype=input_embeds.dtype, device=input_embeds.device)

        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attn_mask = torch.cat([attn_mask, future_mask], dim=-1)

        if gids is None:
            gids = torch.arange(B, dtype=torch.long, device=input_embeds.device)

        # Encoder
        hidden = self.encoder(input_embeds, gids, attn_mask)

        # Output head
        forecast_embeds = hidden[:, -nop:]
        qp = self.output_patch_embedding(forecast_embeds)
        qp = qp.view(B, nop, self.num_quantiles, self.output_patch_size).permute(0, 2, 1, 3).reshape(B, self.num_quantiles, nop * self.output_patch_size)

        # Unscale
        qp = qp.reshape(B, -1)
        qp = self.instance_norm.inverse(qp, loc_scale)
        qp = qp.view(B, self.num_quantiles, -1)

        return qp.reshape(1, -1)

    def load_weights(self, weights):
        # Mapping: checkpoint separate q/k/v → our merged qkv
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attention.qkv", ".self_attention.q", "q"),
            (".self_attention.qkv", ".self_attention.k", "k"),
            (".self_attention.qkv", ".self_attention.v", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded = set()

        for name, loaded_weight in weights:
            # Skip RoPE buffers (computed from config)
            if "rope_embed.inv_freq" in name:
                continue

            # Remap HF layer structure to our native structure
            # layer.0 → time_attn, layer.1 → group_attn, layer.2 → ff
            name = name.replace(".layer.0.self_attention.", ".time_attn.self_attention.")
            name = name.replace(".layer.0.layer_norm.", ".time_attn.layer_norm.")
            name = name.replace(".layer.1.self_attention.", ".group_attn.self_attention.")
            name = name.replace(".layer.1.layer_norm.", ".group_attn.layer_norm.")
            name = name.replace(".layer.2.mlp.wi.", ".ff.wi.")
            name = name.replace(".layer.2.mlp.wo.", ".ff.wo.")
            name = name.replace(".layer.2.layer_norm.", ".ff.layer_norm.")

            # Handle stacked QKV params
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Non-stacked params: direct load
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, loaded_weight)
                else:
                    param.data.copy_(loaded_weight)
            loaded.add(name)

        return loaded
