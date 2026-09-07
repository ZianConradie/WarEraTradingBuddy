"""Load trained item models and create recursive daily price forecasts."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market_relationships import related_percentage_changes
from train import PricePredictor, load_price_history, make_date_features, safe_item_name


MODEL_DIR = Path("./price_models")
PREDICTION_DAYS = 3
SHOW_GRAPHS = False


def load_recent_history(history_days):
    """Load the latest consecutive calendar days from downloaded history."""

    if history_days < 1:
        raise ValueError("history_days must be at least 1")

    history = load_price_history()
    available_dates = sorted(pd.Timestamp(value).normalize() for value in history["Date"].unique())
    if len(available_dates) < history_days:
        raise RuntimeError(
            f"Need {history_days} history days, but only {len(available_dates)} are available."
        )

    selected_dates = available_dates[-history_days:]
    expected_dates = list(pd.date_range(selected_dates[0], periods=history_days, freq="D"))
    if selected_dates != expected_dates:
        raise RuntimeError(
            f"The latest {history_days} price-history days are not consecutive: "
            f"{selected_dates[0].date()} to {selected_dates[-1].date()}."
        )

    return history[history["Date"].isin(selected_dates)].copy()


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model_for_item(item):
    """Load one item checkpoint using the architecture saved during training."""

    model_path = MODEL_DIR / f"{safe_item_name(item)}.pt"
    if not model_path.exists():
        print(f"  No trained model: {model_path}")
        return None

    checkpoint = _load_checkpoint(model_path)
    feature_count = int(checkpoint.get("feature_count", 4))
    model = PricePredictor(
        feature_count=feature_count,
        hidden_size=int(checkpoint.get("gru_hidden_size", 128)),
        gru_layers=int(checkpoint.get("gru_layers", 2)),
        dropout=float(checkpoint.get("dropout", 0.10)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return (
        model,
        int(checkpoint["history_days"]),
        feature_count,
        float(checkpoint.get("price_scale", 1.0)),
    )


def _history_features(item_df, related_changes, price_scale, feature_count):
    frame = item_df.sort_values("Date").copy()
    frame["Price Change"] = frame["Market Midpoint"].diff().fillna(0.0)
    previous = frame["Market Midpoint"].shift(1).abs().clip(lower=1e-8)
    frame["Percentage Change"] = (frame["Price Change"] / previous).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    features = []
    for row in frame.to_dict("records"):
        values = [
            float(row["Market Midpoint"]) / price_scale,
            float(row["Spread"]) / price_scale,
            float(row["Price Change"]) / price_scale,
            float(row["Percentage Change"]),
        ]
        if feature_count >= 5:
            values.append(float(related_changes.get(pd.Timestamp(row["Date"]), 0.0)))
        if feature_count > len(values):
            values.extend([0.0] * (feature_count - len(values)))
        features.append(values[:feature_count])

    return frame, features


def predict_item(model, item, history_df, history_days, price_scale):
    """Predict future midpoints recursively from one item's latest history."""

    item_rows = history_df[history_df["Item"] == item].sort_values("Date").tail(history_days)
    if len(item_rows) != history_days:
        raise RuntimeError(
            f"{item} needs {history_days} consecutive observations; {len(item_rows)} are available."
        )

    expected_dates = list(pd.date_range(item_rows["Date"].iloc[0], periods=history_days, freq="D"))
    actual_dates = [pd.Timestamp(value).normalize() for value in item_rows["Date"]]
    if actual_dates != expected_dates:
        raise RuntimeError(f"{item} history is not consecutive.")

    feature_count = int(model.gru.input_size)
    related_changes = related_percentage_changes(history_df, item)
    frame, features = _history_features(item_rows, related_changes, price_scale, feature_count)

    current_price = float(frame["Market Midpoint"].iloc[-1])
    current_spread = float(frame["Spread"].iloc[-1])
    current_date = pd.Timestamp(frame["Date"].iloc[-1])
    predicted_dates = []
    predicted_prices = []

    for step in range(1, PREDICTION_DAYS + 1):
        prediction_date = current_date + pd.Timedelta(days=step)
        history_tensor = torch.tensor(
            np.asarray(features[-history_days:], dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(0)
        date_tensor = torch.tensor(
            np.asarray(make_date_features(prediction_date), dtype=np.float32),
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            predicted_change = float(model(history_tensor, date_tensor).item()) * price_scale

        next_price = max(0.0, current_price + predicted_change)
        percentage_change = predicted_change / max(abs(current_price), 1e-8)
        next_features = [
            next_price / price_scale,
            current_spread / price_scale,
            predicted_change / price_scale,
            percentage_change,
        ]
        if feature_count >= 5:
            # Future related-market movement is unknown; zero is the neutral, no-leakage input.
            next_features.append(0.0)
        if feature_count > len(next_features):
            next_features.extend([0.0] * (feature_count - len(next_features)))
        features.append(next_features[:feature_count])

        predicted_dates.append(prediction_date)
        predicted_prices.append(next_price)
        current_price = next_price

    return {
        "item": item,
        "historical_dates": list(frame["Date"]),
        "historical_prices": [float(value) for value in frame["Market Midpoint"]],
        "predicted_dates": predicted_dates,
        "predicted_prices": predicted_prices,
    }
