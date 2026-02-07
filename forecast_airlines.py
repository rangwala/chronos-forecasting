"""
Airline Passengers Forecasting using Chronos-2 vLLM Wrapper.

Uses the classic Box-Jenkins airline passengers dataset (1949-1960)
to demonstrate forecasting with the new vLLM integration, comparing
the vLLM wrapper against the original Chronos-2 pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from chronos import BaseChronosPipeline
from chronos.chronos2.vllm import (
    Chronos2ForVLLM,
    preprocess_for_chronos2,
    postprocess_chronos2_output,
)

# --- Airline passengers data (monthly, 1949-01 to 1960-12) ---
# Classic Box-Jenkins dataset: thousands of international airline passengers
passengers = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,  # 1949
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,  # 1950
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,  # 1951
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,  # 1952
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,  # 1953
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,  # 1954
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,  # 1955
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,  # 1956
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,  # 1957
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,  # 1958
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,  # 1959
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,  # 1960
]

dates = pd.date_range(start="1949-01", periods=len(passengers), freq="MS")
df = pd.DataFrame({"date": dates, "passengers": passengers})

# Hold out last 12 months for evaluation
train = df.iloc[:-12]
test = df.iloc[-12:]
prediction_length = 12

print("=" * 70)
print("Airline Passengers Forecasting with Chronos-2 vLLM Wrapper")
print("=" * 70)
print(f"Training: {train['date'].iloc[0].strftime('%Y-%m')} to "
      f"{train['date'].iloc[-1].strftime('%Y-%m')} ({len(train)} months)")
print(f"Test:     {test['date'].iloc[0].strftime('%Y-%m')} to "
      f"{test['date'].iloc[-1].strftime('%Y-%m')} ({len(test)} months)")
print()

# =====================================================================
# PART 1: Verify vLLM equivalence with dummy model
# =====================================================================
print("-" * 70)
print("PART 1: vLLM Equivalence Check (dummy model)")
print("-" * 70)

from pathlib import Path
DUMMY_MODEL_PATH = Path("test/dummy-chronos2-model")

print("Loading dummy model via original pipeline...")
dummy_pipeline = BaseChronosPipeline.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")

print("Loading dummy model via vLLM wrapper...")
dummy_vllm = Chronos2ForVLLM.from_pretrained(DUMMY_MODEL_PATH, device_map="cpu")

context_values = torch.tensor(train["passengers"].values, dtype=torch.float32)
context_3d = context_values.unsqueeze(0).unsqueeze(0)  # (1, 1, 132)
context_2d = context_values.unsqueeze(0)  # (1, 132)

dummy_orig_out = dummy_pipeline.predict(context_3d, prediction_length=prediction_length)
dummy_vllm_out = dummy_vllm.forward_forecast(context_2d, prediction_length=prediction_length)

orig_preds = dummy_orig_out[0].squeeze(0)
vllm_preds = dummy_vllm_out.quantile_preds.squeeze(0)
max_diff = torch.max(torch.abs(orig_preds - vllm_preds)).item()
print(f"  Max absolute difference: {max_diff:.8f}")
print(f"  Outputs match: {'YES' if max_diff < 1e-4 else 'NO'}")

# Also test full pipeline path
inputs = preprocess_for_chronos2(
    context_2d,
    prediction_length=prediction_length,
    output_patch_size=dummy_vllm.output_patch_size,
    max_context_length=dummy_vllm.context_length,
)
raw_output = dummy_vllm.forward(
    inputs.context,
    context_mask=inputs.context_mask,
    num_output_patches=inputs.num_output_patches,
)
forecast = postprocess_chronos2_output(
    raw_output,
    quantile_levels=dummy_vllm.quantiles,
    prediction_length=prediction_length,
)
print(f"  Full pipeline output shape: {forecast.predictions.shape}")
print(f"  Full pipeline quantiles: {len(forecast.quantiles)} keys")
print()

# =====================================================================
# PART 2: Real forecasting with pretrained Chronos-2
# =====================================================================
print("-" * 70)
print("PART 2: Real Forecasting with Pretrained amazon/chronos-2")
print("-" * 70)

MODEL_ID = "amazon/chronos-2"

print(f"Loading {MODEL_ID} (original pipeline)...")
real_pipeline = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu")

print(f"Loading {MODEL_ID} (vLLM wrapper)...")
real_vllm = Chronos2ForVLLM.from_pretrained(MODEL_ID, device_map="cpu")
print()

# --- Run forecasts ---
print("Running original pipeline forecast...")
real_orig_out = real_pipeline.predict(context_3d, prediction_length=prediction_length)
real_orig_quantiles = real_orig_out[0].squeeze(0)  # (num_quantiles, pred_len)

print("Running vLLM wrapper forecast...")
real_vllm_out = real_vllm.forward_forecast(context_2d, prediction_length=prediction_length)
real_vllm_quantiles = real_vllm_out.quantile_preds.squeeze(0)

# --- Extract quantile indices ---
quantile_list = list(real_vllm.quantiles)
median_idx = min(range(len(quantile_list)), key=lambda i: abs(quantile_list[i] - 0.5))
low_idx = min(range(len(quantile_list)), key=lambda i: abs(quantile_list[i] - 0.1))
high_idx = min(range(len(quantile_list)), key=lambda i: abs(quantile_list[i] - 0.9))

# --- Equivalence check ---
orig_median = real_orig_quantiles[median_idx].detach().numpy()
vllm_median = real_vllm_quantiles[median_idx].detach().numpy()
actual = test["passengers"].values.astype(float)

max_diff_real = np.max(np.abs(orig_median - vllm_median))
print()
print(f"Equivalence (Original vs vLLM on real model):")
print(f"  Max absolute difference: {max_diff_real:.6f}")
print(f"  Outputs match: {'YES' if max_diff_real < 1e-3 else 'NO'}")

# --- Metrics ---
mae = np.mean(np.abs(actual - vllm_median))
mape = np.mean(np.abs((actual - vllm_median) / actual)) * 100
rmse = np.sqrt(np.mean((actual - vllm_median) ** 2))
print()
print("Forecast Accuracy (vs actual 1960 passengers):")
print(f"  MAE:  {mae:.1f}")
print(f"  RMSE: {rmse:.1f}")
print(f"  MAPE: {mape:.1f}%")
print()

# --- Forecast table ---
orig_low = real_orig_quantiles[low_idx].detach().numpy()
orig_high = real_orig_quantiles[high_idx].detach().numpy()

print("Month-by-Month Forecast:")
print(f"{'Month':<10} {'Actual':>8} {'Median':>8} {'Low(10%)':>10} {'High(90%)':>10} {'Error':>8}")
print("-" * 54)
for i in range(prediction_length):
    month = test["date"].iloc[i].strftime("%Y-%m")
    err = vllm_median[i] - actual[i]
    print(f"{month:<10} {actual[i]:>8.0f} {vllm_median[i]:>8.1f} "
          f"{orig_low[i]:>10.1f} {orig_high[i]:>10.1f} {err:>+8.1f}")
print()

# --- Plot ---
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full history + forecast
ax = axes[0]
ax.plot(train["date"], train["passengers"], "k-", label="Training data", linewidth=1)
ax.plot(test["date"], actual, "ko-", label="Actual (held out)", linewidth=2, markersize=5)
ax.plot(test["date"], vllm_median, "b-", label="Chronos-2 vLLM (median)", linewidth=2)
ax.fill_between(
    test["date"], orig_low, orig_high,
    alpha=0.25, color="blue", label="80% prediction interval"
)
ax.set_title("Airline Passengers: Chronos-2 vLLM Forecast", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Passengers (thousands)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# Plot 2: Zoomed forecast with equivalence comparison
ax = axes[1]
recent_train = train.iloc[-24:]
ax.plot(recent_train["date"], recent_train["passengers"], "k-",
        label="Recent history", linewidth=1.5)
ax.plot(test["date"], actual, "ko-", label="Actual", markersize=6, linewidth=2)
ax.plot(test["date"], orig_median, "b-", label="Original pipeline", linewidth=2)
ax.plot(test["date"], vllm_median, "r--", label="vLLM wrapper", linewidth=2)
ax.fill_between(
    test["date"], orig_low, orig_high,
    alpha=0.15, color="blue", label="80% interval (original)"
)
vllm_low = real_vllm_quantiles[low_idx].detach().numpy()
vllm_high = real_vllm_quantiles[high_idx].detach().numpy()
ax.fill_between(
    test["date"], vllm_low, vllm_high,
    alpha=0.15, color="red", label="80% interval (vLLM)"
)
ax.set_title("Forecast Detail: Original Pipeline vs vLLM Wrapper (should overlap)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Passengers (thousands)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("forecast_airlines.png", dpi=150, bbox_inches="tight")
print("Plot saved to forecast_airlines.png")
print("=" * 70)
print("DONE")
