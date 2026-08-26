"""
Hourly WarEra Trading Alert System

Every hour on the hour, this script:

1. Downloads any missing price-history sheets.
2. Makes sure at least 30 consecutive days of history are available.
3. Loads each item's independently trained AI model.
4. Predicts the next 30 days.
5. Looks at the previous 2 real days to determine the current trend.
6. If the recent trend is upward:
       Follow the AI predictions until the first predicted decrease.
       The previous predicted day is considered the peak.
       Send a SELL notification.
7. If the recent trend is downward:
       Follow the AI predictions until the first predicted increase.
       The previous predicted day is considered the bottom.
       Send a BUY notification.
8. Includes the predicted turning-point price and confidence.
9. Only sends notifications when confidence is at least 95%.
10. Attaches a graph ending on the day after the turning point.
11. Sends ONLY ONE notification per signal.
12. Runs immediately when started, then every hour on the hour.
"""

from datetime import datetime, timedelta
from time import sleep
from pathlib import Path
import math

import numpy as np
import pandas as pd

import matplotlib

# Background process. Never open a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from download_history import main as download_history
from notify import send_notification

import predict


# ============================================================
# CONFIGURATION
# ============================================================

TOPIC = "WarEraTrading-0481958172"

NOTIFICATION_PRIORITY = 3 # popup, no vibration, no sound

# Number of future days to predict.
PREDICTION_DAYS = 2+1
MIN_HISTORY_DAYS = 30

# Number of real historical days used to determine trend.
TREND_DAYS = 2

# Only send notifications at or above this confidence.
MIN_CONFIDENCE = 0.95

# Temporary notification graphs.
GRAPH_DIR = Path("./notification_graphs")


# ============================================================
# ICON
# ============================================================

# This MUST be a publicly accessible direct image URL.
#
# If ntfy does not display this image, replace it with another
# direct HTTPS image URL.
#
# The URL must point directly to an image, not a webpage.
#
ICON_URL = (
    "https://i.postimg.cc/9FqSVCnQ/"
    "Gemini-Generated-Image-308f4308f4308f43.jpg"
)


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_prediction_confidence(
    predicted_prices,
    turning_point_index,
):
    """
    Estimate confidence in the predicted turning point.

    The neural network itself does NOT output probability
    confidence, so this is a heuristic.

    Factors:

        - Direction consistency
        - Movement strength
        - Prediction horizon

    Returns:
        float from 0.0 to 1.0
    """

    if turning_point_index <= 0:
        return 0.0

    prices = np.asarray(
        predicted_prices,
        dtype=np.float64,
    )

    if len(prices) < 2:
        return 0.0

    relevant = prices[
        :turning_point_index + 1
    ]

    changes = np.diff(relevant)

    if len(changes) == 0:
        return 0.0

    # ========================================================
    # Direction consistency
    # ========================================================

    if changes[-1] < 0:

        # Looking for an upward trend ending in a decrease.
        direction_changes = changes[:-1]

        if len(direction_changes) == 0:
            direction_consistency = 1.0

        else:
            direction_consistency = np.mean(
                direction_changes > 0
            )

    elif changes[-1] > 0:

        # Looking for a downward trend ending in an increase.
        direction_changes = changes[:-1]

        if len(direction_changes) == 0:
            direction_consistency = 1.0

        else:
            direction_consistency = np.mean(
                direction_changes < 0
            )

    else:

        direction_consistency = 0.0

    # ========================================================
    # Movement strength
    # ========================================================

    absolute_changes = np.abs(changes)

    mean_change = np.mean(
        absolute_changes
    )

    if mean_change <= 1e-12:

        movement_strength = 0.0

    else:

        turning_change = abs(
            changes[-1]
        )

        movement_strength = (
            turning_change
            /
            (mean_change + 1e-12)
        )

        movement_strength = min(
            movement_strength,
            1.0,
        )

    # ========================================================
    # Horizon confidence
    # ========================================================

    horizon = turning_point_index + 1

    horizon_confidence = (
        1.0
        /
        np.sqrt(horizon)
    )

    horizon_confidence = min(
        horizon_confidence * 1.5,
        1.0,
    )

    # ========================================================
    # Combine
    # ========================================================

    confidence = (
        direction_consistency * 0.55
        +
        movement_strength * 0.25
        +
        horizon_confidence * 0.20
    )

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )


# ============================================================
# FIND TURNING POINT
# ============================================================

def find_turning_point(
    recent_prices,
    predicted_prices,
):
    """
    Determine the recent real trend and find the first
    predicted reversal.

    Returns:
        dict | None
    """

    if len(recent_prices) < TREND_DAYS:
        return None

    if len(predicted_prices) < 2:
        return None

    # ========================================================
    # Recent real trend
    # ========================================================

    recent_start = float(
        recent_prices[-TREND_DAYS]
    )

    recent_end = float(
        recent_prices[-1]
    )

    if recent_end > recent_start:

        trend = "UP"

    elif recent_end < recent_start:

        trend = "DOWN"

    else:

        return None

    # ========================================================
    # Combine latest real price + predictions
    # ========================================================

    prices = (
        [recent_end]
        +
        [
            float(price)
            for price in predicted_prices
        ]
    )

    # ========================================================
    # UPWARD TREND -> SELL AT PEAK
    # ========================================================

    if trend == "UP":

        for i in range(
            1,
            len(prices),
        ):

            current_price = prices[i]
            previous_price = prices[i - 1]

            if current_price < previous_price:

                peak_index = i - 1

                # Turning point must be an AI prediction.
                if peak_index == 0:
                    return None

                peak_price = prices[
                    peak_index
                ]

                prediction_index = (
                    peak_index - 1
                )

                confidence = (
                    calculate_prediction_confidence(
                        predicted_prices=predicted_prices,
                        turning_point_index=prediction_index,
                    )
                )

                return {
                    "action": "SELL",
                    "price": peak_price,
                    "prediction_index": prediction_index,
                    "confidence": confidence,
                    "trend": trend,
                }

    # ========================================================
    # DOWNWARD TREND -> BUY AT BOTTOM
    # ========================================================

    elif trend == "DOWN":

        for i in range(
            1,
            len(prices),
        ):

            current_price = prices[i]
            previous_price = prices[i - 1]

            if current_price > previous_price:

                bottom_index = i - 1

                # Turning point must be an AI prediction.
                if bottom_index == 0:
                    return None

                bottom_price = prices[
                    bottom_index
                ]

                prediction_index = (
                    bottom_index - 1
                )

                confidence = (
                    calculate_prediction_confidence(
                        predicted_prices=predicted_prices,
                        turning_point_index=prediction_index,
                    )
                )

                return {
                    "action": "BUY",
                    "price": bottom_price,
                    "prediction_index": prediction_index,
                    "confidence": confidence,
                    "trend": trend,
                }

    return None


# ============================================================
# DIRECTIONAL 3-DECIMAL ROUNDING
# ============================================================

def round_price_three_decimals(
    value,
    direction,
):
    """
    Round to exactly 3 digits after the decimal point.

    BUY / bottom:
        Round UP.

    SELL / peak:
        Round DOWN.

    Examples:

        BUY:
            123.4561 -> 123.457

        SELL:
            123.4569 -> 123.456

        BUY:
            5.10001 -> 5.101

        SELL:
            5.10099 -> 5.100
    """

    value = float(value)

    if not math.isfinite(value):
        return value

    multiplier = 1000.0

    if direction == "UP":

        rounded = (
            math.ceil(
                value * multiplier - 1e-12
            )
            /
            multiplier
        )

    elif direction == "DOWN":

        rounded = (
            math.floor(
                value * multiplier + 1e-12
            )
            /
            multiplier
        )

    else:

        raise ValueError(
            "direction must be 'UP' or 'DOWN'"
        )

    return rounded


def format_price(
    price,
    action,
):
    """
    Format a price with exactly 3 decimal places.

    BUY:
        Round upward.

    SELL:
        Round downward.
    """

    if action == "BUY":

        rounded = round_price_three_decimals(
            price,
            direction="UP",
        )

    elif action == "SELL":

        rounded = round_price_three_decimals(
            price,
            direction="DOWN",
        )

    else:

        raise ValueError(
            f"Unknown action: {action}"
        )

    return f"{rounded:.3f}"


# ============================================================
# LOAD / PREDICT ONE ITEM
# ============================================================

def process_item(
    item,
    history_df,
):
    """
    Load the item's independent model and generate
    the prediction.
    """

    loaded = predict.load_model_for_item(
        item
    )

    if loaded is None:
        return None

    (
        model,
        history_days,
        feature_count,
        price_scale,
    ) = loaded

    available_history_days = len(
        history_df["Date"].unique()
    )

    if history_days != available_history_days:

        item_history_df = (
            predict.load_recent_history(
                history_days
            )
        )

    else:

        item_history_df = history_df

    result = predict.predict_item(
        model=model,
        item=item,
        history_df=item_history_df,
        history_days=history_days,
        price_scale=price_scale,
    )

    return result


# ============================================================
# GRAPH
# ============================================================

def create_notification_graph(
    result,
    signal,
):
    """
    Create the graph used by the notification.

    Blue:
        Real historical prices.

    Red:
        AI predicted prices.

    Green:
        Predicted turning point.

    The graph ends on the first predicted reversal day,
    allowing the viewer to see the peak/bottom followed
    by the reversal.

    Returns:
        Path
    """

    item = result["item"]

    historical_dates = [
        pd.Timestamp(date)
        for date in result[
            "historical_dates"
        ]
    ]

    historical_prices = [
        float(price)
        for price in result[
            "historical_prices"
        ]
    ]

    predicted_dates = [
        pd.Timestamp(date)
        for date in result[
            "predicted_dates"
        ]
    ]

    predicted_prices = [
        float(price)
        for price in result[
            "predicted_prices"
        ]
    ]

    prediction_index = int(
        signal["prediction_index"]
    )

    # ========================================================
    # Show turning point + first reversal day.
    #
    # prediction_index:
    #
    #     0 = first prediction
    #     1 = second prediction
    #     ...
    #
    # Therefore:
    #
    #     index + 2
    #
    # includes the turning point and the next day.
    # ========================================================

    plot_prediction_count = min(
        prediction_index + 2,
        len(predicted_prices),
    )

    visible_predicted_dates = (
        predicted_dates[
            :plot_prediction_count
        ]
    )

    visible_predicted_prices = (
        predicted_prices[
            :plot_prediction_count
        ]
    )

    # ========================================================
    # GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    # ========================================================
    # HISTORICAL
    # ========================================================

    ax.plot(
        historical_dates,
        historical_prices,
        color="blue",
        marker="o",
        label="Historical",
    )

    # ========================================================
    # PREDICTION
    #
    # Include final real point so red starts exactly where
    # blue ends.
    # ========================================================

    prediction_dates_for_plot = (
        [historical_dates[-1]]
        +
        visible_predicted_dates
    )

    prediction_prices_for_plot = (
        [historical_prices[-1]]
        +
        visible_predicted_prices
    )

    ax.plot(
        prediction_dates_for_plot,
        prediction_prices_for_plot,
        color="red",
        marker="o",
        label="Predicted",
    )

    # ========================================================
    # TURNING POINT
    # ========================================================

    turning_point_date = (
        visible_predicted_dates[
            prediction_index
        ]
    )

    turning_point_price = (
        visible_predicted_prices[
            prediction_index
        ]
    )

    ax.scatter(
        [turning_point_date],
        [turning_point_price],
        color="green",
        s=100,
        zorder=5,
        label=(
            "Peak"
            if signal["action"] == "SELL"
            else "Bottom"
        ),
    )

    # ========================================================
    # TITLE
    # ========================================================

    action_name = (
        "Predicted Peak"
        if signal["action"] == "SELL"
        else "Predicted Bottom"
    )

    ax.set_title(
        f"{item} - {action_name}"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Market Midpoint"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    fig.tight_layout()

    # ========================================================
    # SAVE
    # ========================================================

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_item = "".join(
        character
        if (
            character.isalnum()
            or character in "._-"
        )
        else "_"
        for character in str(item)
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_path = (
        GRAPH_DIR
        /
        f"{safe_item}_{timestamp}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        format="png",
    )

    plt.close(fig)

    print(
        f"  Graph saved: {output_path}"
    )

    return output_path


# ============================================================
# SEND NOTIFICATION
# ============================================================

def send_trade_notification(
    item,
    signal,
    graph_path,
):
    """
    Send exactly ONE BUY/SELL notification.

    The graph is attached directly to that notification.

    The notification contains:

        - Title
        - Predicted peak/bottom
        - Prediction horizon
        - Confidence
        - Graph attachment
        - Icon
    """

    action = signal["action"]
    price = signal["price"]
    confidence = signal["confidence"]

    confidence_percent = (
        confidence * 100.0
    )

    prediction_day = (
        signal["prediction_index"] + 1
    )

    # ========================================================
    # ROUND PRICE
    # ========================================================

    display_price = format_price(
        price,
        action,
    )

    # ========================================================
    # BUILD MESSAGE
    # ========================================================

    if action == "SELL":

        title = (
            f"SELL: {item}"
        )

        message = (
            f"Predicted peak: {display_price}\n"
            f"Set up a SELL order around this price.\n\n"
            f"Predicted turning point: "
            f"{prediction_day} day(s) from now\n"
            f"Overall forecast confidence: "
            f"{confidence_percent:.1f}%"
        )

    else:

        title = (
            f"BUY: {item}"
        )

        message = (
            f"Predicted bottom: {display_price}\n"
            f"Set up a BUY order around this price.\n\n"
            f"Predicted turning point: "
            f"{prediction_day} day(s) from now\n"
            f"Overall forecast confidence: "
            f"{confidence_percent:.1f}%"
        )

    # ========================================================
    # VALIDATE GRAPH
    # ========================================================

    graph_path = Path(
        graph_path
    )

    if not graph_path.exists():

        print(
            f"  Graph does not exist: "
            f"{graph_path}"
        )

        return None

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(
        "NOTIFICATION"
    )
    print(
        "-" * 70
    )
    print(title)
    print(message)
    print(
        f"Graph: {graph_path}"
    )
    print(
        f"Icon: {ICON_URL}"
    )
    print(
        "-" * 70
    )

    # ========================================================
    # SEND ONE NOTIFICATION
    # ========================================================

    try:

        response = send_notification(
            title=title,
            message=message,
            topic=TOPIC,
            priority=NOTIFICATION_PRIORITY,

            # The graph is attached directly.
            attach=graph_path,

            filename=(
                f"{item}_prediction.png"
            ),

            icon=ICON_URL,
        )

        print(
            "Notification sent successfully."
        )

        return response

    except Exception as e:

        print(
            f"Failed to send notification: {e}"
        )

        return None
# ============================================================
# ONE COMPLETE RUN
# ============================================================

def run_once():
    """
    Perform one complete prediction/notification cycle.
    """

    print()
    print(
        "=" * 70
    )
    print(
        "WARERA TRADING CHECK"
    )
    print(
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )
    print(
        "=" * 70
    )

    # ========================================================
    # UPDATE HISTORY
    # ========================================================

    print()
    print(
        "Updating price history..."
    )

    try:

        download_history()

    except Exception as e:

        print(
            f"Price-history update failed: {e}"
        )

        return

    # ========================================================
    # MODEL DIRECTORY
    # ========================================================

    if not predict.MODEL_DIR.exists():

        print(
            f"Model directory does not exist: "
            f"{predict.MODEL_DIR}"
        )

        return

    # ========================================================
    # LOAD HISTORY
    # ========================================================

    try:

        history_df = (
            predict.load_recent_history(
                MIN_HISTORY_DAYS
            )
        )

    except Exception as e:

        print(
            f"Could not load enough history: {e}"
        )

        return

    # ========================================================
    # ITEMS
    # ========================================================

    items = sorted(
        history_df["Item"].unique()
    )

    print()
    print(
        f"Found {len(items)} items."
    )

    notifications_sent = 0

    # ========================================================
    # PROCESS ITEMS
    # ========================================================

    for item in items:

        print()
        print(
            f"Processing: {item}"
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        try:

            result = process_item(
                item=item,
                history_df=history_df,
            )

        except Exception as e:

            print(
                f"  Prediction failed: {e}"
            )

            continue

        if result is None:
            continue

        # ----------------------------------------------------
        # Historical prices
        # ----------------------------------------------------

        historical_prices = result[
            "historical_prices"
        ]

        if len(historical_prices) < TREND_DAYS:

            print(
                "  Not enough historical prices "
                "for trend detection."
            )

            continue

        recent_prices = historical_prices[
            -TREND_DAYS:
        ]

        # ----------------------------------------------------
        # Predicted prices
        # ----------------------------------------------------

        predicted_prices = result[
            "predicted_prices"
        ]

        # ----------------------------------------------------
        # Find turning point
        # ----------------------------------------------------

        signal = find_turning_point(
            recent_prices=recent_prices,
            predicted_prices=predicted_prices,
        )

        if signal is None:

            print(
                "  No predicted turning point "
                "within forecast."
            )

            continue

        # ====================================================
        # PRINT SIGNAL
        # ====================================================

        print()
        print(
            f"  SIGNAL: {signal['action']}"
        )

        print(
            f"  Raw price: "
            f"{signal['price']:.12g}"
        )

        print(
            f"  Rounded price: "
            f"{format_price(signal['price'], signal['action'])}"
        )

        print(
            f"  Prediction day: "
            f"{signal['prediction_index'] + 1}"
        )

        confidence = signal[
            "confidence"
        ]

        print(
            f"  Confidence: "
            f"{confidence * 100:.1f}%"
        )

        # ====================================================
        # CONFIDENCE FILTER
        # ====================================================

        if confidence < MIN_CONFIDENCE:

            print(
                f"  Skipping notification: "
                f"confidence "
                f"{confidence * 100:.1f}% is below "
                f"the "
                f"{MIN_CONFIDENCE * 100:.1f}% "
                f"threshold."
            )

            continue

        # ====================================================
        # CREATE GRAPH
        # ====================================================

        try:

            graph_path = (
                create_notification_graph(
                    result=result,
                    signal=signal,
                )
            )

        except Exception as e:

            print(
                f"  Failed to create graph: {e}"
            )

            continue

        # ====================================================
        # SEND ONE NOTIFICATION
        # ====================================================

        response = (
            send_trade_notification(
                item=item,
                signal=signal,
                graph_path=graph_path,
            )
        )

        if response is not None:

            notifications_sent += 1

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print(
        "=" * 70
    )
    print(
        "CHECK COMPLETE"
    )
    print(
        f"Signals sent: "
        f"{notifications_sent}"
    )
    print(
        "=" * 70
    )


# ============================================================
# WAIT UNTIL NEXT HOUR
# ============================================================

def seconds_until_next_hour():
    """
    Return seconds until the next exact hour.
    """

    now = datetime.now()

    next_hour = (
        now.replace(
            minute=0,
            second=0,
            microsecond=0,
        )
        +
        timedelta(hours=1)
    )

    return max(
        1,
        int(
            (
                next_hour - now
            ).total_seconds()
        ),
    )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print(
        "=" * 70
    )
    print(
        "WARERA TRADING NOTIFIER"
    )
    print(
        "=" * 70
    )

    print(
        f"Prediction horizon: "
        f"{PREDICTION_DAYS} days"
    )

    print(
        f"Required history: "
        f"{MIN_HISTORY_DAYS} days"
    )

    print(
        f"Notification confidence threshold: "
        f"{MIN_CONFIDENCE * 100:.1f}%"
    )

    print(
        f"Notification topic: "
        f"{TOPIC}"
    )

    print(
        "Schedule: immediately, then every hour"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Prediction configuration
    # ========================================================

    predict.PREDICTION_DAYS = (
        PREDICTION_DAYS
    )

    # Never show graphs from predict.py.
    predict.SHOW_GRAPHS = False

    # ========================================================
    # Run immediately
    # ========================================================

    run_once()

    # ========================================================
    # Hourly loop
    # ========================================================

    while True:

        wait_seconds = (
            seconds_until_next_hour()
        )

        next_run = (
            datetime.now()
            +
            timedelta(
                seconds=wait_seconds
            )
        )

        print()
        print(
            f"Next check at "
            f"{next_run.strftime('%H:%M:%S')}"
        )

        sleep(
            wait_seconds
        )

        run_once()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()