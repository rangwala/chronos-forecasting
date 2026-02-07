# Chronos-2 vLLM Integration

## Overview

vLLM-compatible serving layer for Chronos-2 time series forecasting models. Enables deployment via vLLM's serving infrastructure while maintaining output equivalence with the original `Chronos2Pipeline`.

## What Was Done

### 1. vLLM Wrapper Implementation (commit `8814178`)

Created `src/chronos/chronos2/vllm/` with 5 modules (3,126 lines):

| Module | Purpose |
|---|---|
| `__init__.py` | Package exports |
| `wrapper.py` | `Chronos2ForVLLM` — nn.Module wrapper exposing vLLM-compatible interfaces |
| `preprocessing.py` | `preprocess_for_chronos2()` — converts various input formats to standardized tensors |
| `postprocessing.py` | `postprocess_chronos2_output()` — converts model output to `ForecastOutput` / JSON |
| `register.py` | `Chronos2VLLMPlugin` — model registration with vLLM's `ModelRegistry` |

Unit tests in `test/test_vllm_*.py` (5 test files).

### 2. Ran All vLLM Unit & Integration Tests

```
test/test_vllm_preprocessing.py    31 passed
test/test_vllm_postprocessing.py   29 passed
test/test_vllm_register.py         25 passed
test/test_vllm_wrapper.py          24 passed
test/test_vllm_integration.py      15 passed
─────────────────────────────────────────────
Total                             124 passed
```

Key validations:
- Output equivalence between `Chronos2Pipeline` and `Chronos2ForVLLM` (bit-identical)
- Preprocessing handles NaN masking, variable-length batching, context truncation
- Postprocessing produces correct quantile dictionaries and JSON serialization
- Edge cases: minimal context, non-multiple-of-patch prediction lengths, numpy inputs

### 3. Airline Passengers Forecasting Demo (commit `80e5d5d`)

Created `forecast_airlines.py` — end-to-end demo using the classic Box-Jenkins airline passengers dataset (1949–1960).

**Part 1: Equivalence verification (dummy model)**
- Loaded dummy model via both original pipeline and vLLM wrapper
- Max absolute difference: **0.000000** (bit-identical)

**Part 2: Real forecasting with `amazon/chronos-2`**
- Held out 1960 (12 months), trained on 1949–1959
- Equivalence confirmed on real model (max diff: 0.000000)

Results (zero-shot, no fine-tuning):

| Metric | Value |
|---|---|
| MAE | 13.4 |
| RMSE | 19.6 |
| MAPE | **3.0%** |

Month-by-month forecast saved to `forecast_airlines.png`.

### 4. Mock vLLM Serving Tests (commit `54b9bdd`)

Identified that vLLM requires NVIDIA CUDA GPUs and cannot be installed on this machine (Intel Mac). Created `test/test_vllm_serving.py` with mock vLLM infrastructure to test the full serving lifecycle with real model inference.

**Mock classes:**

| Mock | Simulates |
|---|---|
| `MockModelRegistry` | `vllm.ModelRegistry` — register/resolve model architectures |
| `MockLLMEngine` | `vllm.LLMEngine` — add_request → step → real inference → response |
| `MockAsyncLLMEngine` | `vllm.AsyncLLMEngine` — async wrapper with generate() iterator |
| `MockRequestOutput` | `vllm.RequestOutput` — response object |
| `MockPoolingOutput` | `vllm.PoolingOutput` — quantile prediction carrier |
| `MockSamplingParams` | `vllm.SamplingParams` — forecasting parameters |
| `MockEngineArgs` | `vllm.EngineArgs` — engine configuration |

**Test coverage (44 tests):**

| Test Class | Tests | What It Covers |
|---|---|---|
| `TestMockModelRegistry` | 5 | Architecture registration, dynamic class resolution |
| `TestEngineModelLoading` | 4 | Model loading from registry, property validation |
| `TestRequestLifecycle` | 7 | Add/step/abort, output shapes, duplicate IDs |
| `TestBatchingBehavior` | 5 | Multi-request batching, prediction length grouping, max batch size |
| `TestAsyncServingFlow` | 5 | Async add/step, generate() iterator, sync/async equivalence |
| `TestPluginInterface` | 4 | Plugin discovery, setup_vllm_integration() |
| `TestErrorHandling` | 7 | Empty/NaN/3D contexts, invalid prediction lengths |
| `TestEndToEndServingPipeline` | 4 | Full flow to JSON, equivalence with direct wrapper |
| `TestOutputFormatConversion` | 3 | Postprocessing, JSON serialization, batch splitting |

### 5. Final Test Count

```
All vLLM tests: 168 passed (124 original + 44 serving)
```

## Architecture

```
Client Request (time series data)
    │
    ▼
MockLLMEngine.add_request()          ← queues request
    │
    ▼
MockLLMEngine.step()                 ← processes batch
    ├── preprocess_for_chronos2()    ← real preprocessing
    ├── Chronos2ForVLLM.forward()    ← real model inference
    └── postprocess_chronos2_output()← real postprocessing
    │
    ▼
MockRequestOutput                    ← response with quantile forecasts
    │
    ▼
format_for_json()                    ← JSON-serializable output
```

## Usage

```python
from chronos.chronos2.vllm import Chronos2ForVLLM, preprocess_for_chronos2, postprocess_chronos2_output

# Load model
model = Chronos2ForVLLM.from_pretrained("amazon/chronos-2", device_map="cpu")

# Direct forecast
context = torch.randn(4, 100)  # 4 series, 100 timesteps
output = model.forward_forecast(context, prediction_length=24)
print(output.quantile_preds.shape)  # (4, 21, 24)

# Full pipeline
inputs = preprocess_for_chronos2(context, prediction_length=24,
    output_patch_size=model.output_patch_size)
raw = model.forward(inputs.context, context_mask=inputs.context_mask,
    num_output_patches=inputs.num_output_patches)
forecast = postprocess_chronos2_output(raw, quantile_levels=model.quantiles,
    prediction_length=24)
```

## Running Tests

```bash
# All vLLM tests
python -m pytest test/test_vllm_*.py -v

# Just the serving tests
python -m pytest test/test_vllm_serving.py -v

# Just the unit/integration tests
python -m pytest test/test_vllm_preprocessing.py test/test_vllm_postprocessing.py \
    test/test_vllm_register.py test/test_vllm_wrapper.py test/test_vllm_integration.py -v
```

## Commits

| Hash | Description |
|---|---|
| `8814178` | vLLM wrapper implementation (5 source + 5 test files) |
| `80e5d5d` | Airline passengers forecasting demo |
| `54b9bdd` | Mock vLLM serving tests (44 tests) |
