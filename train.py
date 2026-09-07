from pathlib import Path
import re
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from market_relationships import related_percentage_changes


# ============================================================
# CONFIG
# ============================================================

PRICE_HISTORY_DIR = Path("./price_history")

# Directory where one model per item will be saved.
MODEL_DIR = Path("./price_models")

# Number of previous calendar days given to each model.
HISTORY_DAYS = 30

# All price-related values are represented in thousandths.
#
# Example:
#
#     0.0825 -> 82.5
#
# This is simple scaling, NOT z-score normalization.
PRICE_SCALE = 1.0

# Training settings.
BATCH_SIZE = 64
MAX_EPOCHS = 300
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5

# Percentage of chronological samples used for training.
TRAIN_RATIO = 0.8

# Stop training when validation loss stops improving.
PATIENCE = 30

# Maximum gradient norm.
GRADIENT_CLIP = 1.0

# GRU settings.
GRU_HIDDEN_SIZE = 128
GRU_LAYERS = 2

# Dropout.
DROPOUT = 0.10

# Reproducibility.
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL
# ============================================================

class PricePredictor(nn.Module):
    """
    One independent model for one item.

    Historical input for each day:

        Market Midpoint / PRICE_SCALE
        Spread / PRICE_SCALE
        Price Change / PRICE_SCALE
        Percentage Price Change
        Mean percentage change of directly related markets

    The 30-day sequence is processed by a GRU.

    Additional inputs:

        Six calendar features

    Output:

        Predicted next-day price change / PRICE_SCALE

    There is intentionally NO item embedding.

    Every saved model belongs to exactly one item.
    """

    def __init__(
        self,
        feature_count,
        hidden_size=GRU_HIDDEN_SIZE,
        gru_layers=GRU_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_count,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        combined_size = hidden_size + 6

        self.head = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1),
        )

    def forward(
        self,
        history,
        date_features,
    ):
        """
        history:
            [batch, HISTORY_DAYS, feature_count]

        date_features:
            [batch, 6]

        Returns:
            [batch, 1]
        """

        gru_output, _ = self.gru(history)

        # Representation of the most recent day.
        temporal_features = gru_output[:, -1, :]

        x = torch.cat(
            [
                temporal_features,
                date_features,
            ],
            dim=1,
        )

        return self.head(x)


# ============================================================
# LOAD PRICE HISTORY
# ============================================================

def load_price_history():
    """
    Load all YYYY-MM-DD.csv files from ./price_history/.

    Required columns:

        Item
        Spread
        Market Midpoint

    If multiple rows exist for the same Item on the same day,
    the LAST row in the CSV is used.
    """

    if not PRICE_HISTORY_DIR.exists():
        raise RuntimeError(
            f"Directory does not exist: {PRICE_HISTORY_DIR}"
        )

    pattern = re.compile(
        r"^\d{4}-\d{2}-\d{2}\.csv$"
    )

    files = sorted(
        path
        for path in PRICE_HISTORY_DIR.iterdir()
        if path.is_file()
        and pattern.match(path.name)
    )

    if not files:
        raise RuntimeError(
            f"No YYYY-MM-DD.csv files found in "
            f"{PRICE_HISTORY_DIR}"
        )

    print(f"Found {len(files)} CSV files.")

    daily_data = []

    for path in files:
        print(f"Loading {path}")

        # ----------------------------------------------------
        # Date
        # ----------------------------------------------------

        try:
            date = pd.to_datetime(
                path.stem,
                format="%Y-%m-%d",
            )
        except ValueError:
            raise ValueError(
                f"Invalid date filename: {path.name}"
            )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        df = pd.read_csv(path)

        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        required_columns = {
            "Item",
            "Spread",
            "Market Midpoint",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{path} is missing columns: "
                f"{sorted(missing)}"
            )

        df = df[
            [
                "Item",
                "Market Midpoint",
                "Spread",
            ]
        ].copy()

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        df["Item"] = (
            df["Item"]
            .astype(str)
            .str.strip()
        )

        df["Market Midpoint"] = pd.to_numeric(
            df["Market Midpoint"],
            errors="coerce",
        )

        df["Spread"] = pd.to_numeric(
            df["Spread"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "Item",
                "Market Midpoint",
                "Spread",
            ]
        )

        df = df[df["Item"] != ""]

        df = df[
            df["Market Midpoint"] >= 0
        ]

        df = df[
            df["Spread"] >= 0
        ]

        # ----------------------------------------------------
        # Add date
        # ----------------------------------------------------

        df["Date"] = date

        # ----------------------------------------------------
        # Multiple rows per item/day
        # ----------------------------------------------------

        before = len(df)

        df = (
            df
            .drop_duplicates(
                subset=["Item"],
                keep="last",
            )
        )

        removed = before - len(df)

        if removed:
            print(
                f"  Removed {removed:,} duplicate "
                f"item observations."
            )

        daily_data.append(df)

    # --------------------------------------------------------
    # Combine all days
    # --------------------------------------------------------

    all_data = pd.concat(
        daily_data,
        ignore_index=True,
    )

    all_data = all_data[
        [
            "Date",
            "Item",
            "Market Midpoint",
            "Spread",
        ]
    ]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    duplicates = all_data.duplicated(
        subset=[
            "Date",
            "Item",
        ],
        keep=False,
    )

    if duplicates.any():
        count = int(duplicates.sum())

        print(
            f"WARNING: {count:,} duplicate "
            f"item/day rows remain."
        )

        all_data = (
            all_data
            .drop_duplicates(
                subset=[
                    "Date",
                    "Item",
                ],
                keep="last",
            )
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    all_data = (
        all_data
        .sort_values(
            [
                "Item",
                "Date",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print(
        f"Loaded "
        f"{len(all_data):,} item/day observations."
    )

    print(
        f"Items: "
        f"{all_data['Item'].nunique()}"
    )

    print(
        f"Dates: "
        f"{all_data['Date'].min().date()} "
        f"-> "
        f"{all_data['Date'].max().date()}"
    )

    print()

    return all_data


# ============================================================
# DATE FEATURES
# ============================================================

def make_date_features(date):
    """
    Convert a date into six features:

        1. sin(day of year)
        2. cos(day of year)
        3. sin(day of week)
        4. cos(day of week)
        5. normalized year
        6. normalized day of month
    """

    date = pd.Timestamp(date)

    day_of_year = date.dayofyear
    day_of_week = date.dayofweek

    annual_angle = (
        2
        * np.pi
        * day_of_year
        / 365.25
    )

    sin_day = np.sin(annual_angle)
    cos_day = np.cos(annual_angle)

    weekly_angle = (
        2
        * np.pi
        * day_of_week
        / 7.0
    )

    sin_week = np.sin(weekly_angle)
    cos_week = np.cos(weekly_angle)

    normalized_year = (
        date.year - 2020
    ) / 10.0

    normalized_day_of_month = (
        (date.day - 1) / 30.0
    )

    return [
        sin_day,
        cos_day,
        sin_week,
        cos_week,
        normalized_year,
        normalized_day_of_month,
    ]


# ============================================================
# CREATE SAMPLES FOR ONE ITEM
# ============================================================

def create_samples_for_item(item_df):
    """
    Create chronological training samples for ONE item.

    Historical features:

        Market Midpoint / PRICE_SCALE
        Spread / PRICE_SCALE
        Price Change / PRICE_SCALE
        Percentage Price Change
        Mean percentage change of directly related markets

    Target:

        Next-day Price Change / PRICE_SCALE

    Returns:

        X_history
        X_dates
        y
        sample_dates
    """

    item_df = (
        item_df
        .sort_values("Date")
        .copy()
    )

    # --------------------------------------------------------
    # Calculate movement
    # --------------------------------------------------------

    item_df["Price Change"] = (
        item_df["Market Midpoint"].diff()
    )

    previous_price = (
        item_df["Market Midpoint"].shift(1)
    )

    item_df["Percentage Change"] = (
        item_df["Price Change"]
        / previous_price.abs().clip(
            lower=1e-8
        )
    )

    item_df["Percentage Change"] = (
        item_df["Percentage Change"]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    lookup = {}

    for row in item_df.to_dict("records"):
        lookup[
            pd.Timestamp(row["Date"])
        ] = row

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    X_history = []
    X_dates = []
    y = []
    sample_dates = []

    all_dates = sorted(
        pd.Timestamp(d)
        for d in item_df["Date"].unique()
    )

    # --------------------------------------------------------
    # Generate samples
    # --------------------------------------------------------

    for target_date in all_dates:

        history = []
        valid = True

        # ----------------------------------------------------
        # Previous HISTORY_DAYS calendar days
        # ----------------------------------------------------

        for days_back in range(
            HISTORY_DAYS,
            0,
            -1,
        ):
            history_date = (
                target_date
                - pd.Timedelta(days=days_back)
            )

            row = lookup.get(history_date)

            if row is None:
                valid = False
                break

            midpoint = float(
                row["Market Midpoint"]
            )

            spread = float(
                row["Spread"]
            )

            price_change = row[
                "Price Change"
            ]

            percentage_change = row[
                "Percentage Change"
            ]

            related_change = row.get(
                "Related Percentage Change",
                0.0,
            )

            if pd.isna(price_change):
                price_change = 0.0

            if pd.isna(percentage_change):
                percentage_change = 0.0

            if pd.isna(related_change):
                related_change = 0.0

            history.append(
                [
                    midpoint / PRICE_SCALE,
                    spread / PRICE_SCALE,
                    float(price_change) / PRICE_SCALE,
                    float(percentage_change),
                    float(related_change),
                ]
            )

        if not valid:
            continue

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target_row = lookup.get(target_date)

        if target_row is None:
            continue

        target_change = target_row[
            "Price Change"
        ]

        if pd.isna(target_change):
            continue

        target_change = (
            float(target_change)
            / PRICE_SCALE
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        X_history.append(
            np.asarray(
                history,
                dtype=np.float32,
            )
        )

        X_dates.append(
            make_date_features(target_date)
        )

        y.append(target_change)

        sample_dates.append(target_date)

    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    X_history = np.asarray(
        X_history,
        dtype=np.float32,
    )

    X_dates = np.asarray(
        X_dates,
        dtype=np.float32,
    )

    y = np.asarray(
        y,
        dtype=np.float32,
    )

    sample_dates = np.asarray(
        sample_dates,
        dtype="datetime64[ns]",
    )

    return (
        X_history,
        X_dates,
        y,
        sample_dates,
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    device,
):
    """
    Evaluate one item's model.
    """

    model.eval()

    loss_function = nn.SmoothL1Loss()

    total_loss = 0.0

    predictions = []
    actuals = []

    with torch.no_grad():

        for (
            history,
            date_features,
            target,
        ) in loader:

            history = history.to(device)
            date_features = date_features.to(device)
            target = target.to(device)

            prediction = (
                model(
                    history,
                    date_features,
                )
                .squeeze(1)
            )

            loss = loss_function(
                prediction,
                target,
            )

            total_loss += (
                loss.item()
                * len(target)
            )

            predictions.extend(
                prediction.cpu().numpy()
            )

            actuals.extend(
                target.cpu().numpy()
            )

    loss = (
        total_loss
        / len(loader.dataset)
    )

    predictions = (
        np.asarray(predictions)
        * PRICE_SCALE
    )

    actuals = (
        np.asarray(actuals)
        * PRICE_SCALE
    )

    mae = np.mean(
        np.abs(
            predictions
            - actuals
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                predictions
                - actuals
            ) ** 2
        )
    )

    return loss, mae, rmse


# ============================================================
# SAFE MODEL NAME
# ============================================================

def safe_item_name(item):
    """
    Convert an item name into a safe filename.
    """

    return re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        str(item),
    )


# ============================================================
# TRAIN ONE ITEM
# ============================================================

def train_item(
    item,
    item_df,
    all_history_df,
    device,
):
    """
    Train one completely independent model for one item.
    """

    print()
    print("=" * 70)
    print(f"TRAINING ITEM: {item}")
    print("=" * 70)

    # --------------------------------------------------------
    # Add same-day movements from direct production-chain neighbours. The
    # target remains the following day, so this context introduces no future
    # price leakage.
    # --------------------------------------------------------

    related_changes = related_percentage_changes(
        all_history_df,
        item,
    )

    item_df = item_df.copy()
    item_df["Related Percentage Change"] = (
        pd.to_datetime(item_df["Date"])
        .map(related_changes)
        .fillna(0.0)
    )

    (
        X_history,
        X_dates,
        y,
        sample_dates,
    ) = create_samples_for_item(item_df)

    print(
        f"Created {len(y):,} samples."
    )

    if len(y) < 2:
        print(
            f"Skipping {item}: not enough samples."
        )
        return None

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    order = np.argsort(sample_dates)

    X_history = X_history[order]
    X_dates = X_dates[order]
    y = y[order]
    sample_dates = sample_dates[order]

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    split_index = int(
        len(y)
        * TRAIN_RATIO
    )

    if split_index <= 0:
        print(
            f"Skipping {item}: no training samples."
        )
        return None

    if split_index >= len(y):
        print(
            f"Skipping {item}: no validation samples."
        )
        return None

    X_history_train = X_history[
        :split_index
    ]

    X_history_val = X_history[
        split_index:
    ]

    X_dates_train = X_dates[
        :split_index
    ]

    X_dates_val = X_dates[
        split_index:
    ]

    y_train = y[
        :split_index
    ]

    y_val = y[
        split_index:
    ]

    print(
        f"Training samples:   {len(y_train):,}"
    )

    print(
        f"Validation samples: {len(y_val):,}"
    )

    print(
        f"Training through:   "
        f"{pd.Timestamp(sample_dates[split_index - 1]).date()}"
    )

    print(
        f"Validation from:    "
        f"{pd.Timestamp(sample_dates[split_index]).date()}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_history_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            X_dates_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32,
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_history_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            X_dates_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_val,
            dtype=torch.float32,
        ),
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    feature_count = int(X_history.shape[2])

    model = PricePredictor(
        feature_count=feature_count,
    ).to(device)

    print(
        f"GRU hidden size:    {GRU_HIDDEN_SIZE}"
    )

    print(
        f"GRU layers:         {GRU_LAYERS}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.SmoothL1Loss()

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_loss = 0.0

        for (
            history,
            date_features,
            target,
        ) in train_loader:

            history = history.to(device)
            date_features = date_features.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            prediction = (
                model(
                    history,
                    date_features,
                )
                .squeeze(1)
            )

            loss = loss_function(
                prediction,
                target,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP,
            )

            optimizer.step()

            train_loss += (
                loss.item()
                * len(target)
            )

        train_loss /= len(
            train_dataset
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for (
                history,
                date_features,
                target,
            ) in val_loader:

                history = history.to(device)
                date_features = date_features.to(device)
                target = target.to(device)

                prediction = (
                    model(
                        history,
                        date_features,
                    )
                    .squeeze(1)
                )

                loss = loss_function(
                    prediction,
                    target,
                )

                val_loss += (
                    loss.item()
                    * len(target)
                )

        val_loss /= len(
            val_dataset
        )

        scheduler.step(val_loss)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        (
            _,
            mae,
            rmse,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        # ----------------------------------------------------
        # Save best state
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            epochs_without_improvement = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

        else:
            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == MAX_EPOCHS
            or epochs_without_improvement == 0
        ):
            print(
                f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
                f"Train: {train_loss:.5f} | "
                f"Val: {val_loss:.5f} | "
                f"MAE: {mae:.6f} | "
                f"RMSE: {rmse:.6f} | "
                f"LR: {current_lr:.2e}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                f"Early stopping at epoch {epoch}."
            )
            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is None:
        print(
            f"Training failed for {item}."
        )
        return None

    model.load_state_dict(best_state)

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    (
        _,
        final_mae,
        final_rmse,
    ) = evaluate(
        model,
        val_loader,
        device,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        MODEL_DIR
        / f"{safe_item_name(item)}.pt"
    )

    torch.save(
        {
            "model_state": best_state,

            "item": item,

            "history_days": HISTORY_DAYS,

            "feature_count": feature_count,

            "price_scale": PRICE_SCALE,

            "gru_hidden_size": GRU_HIDDEN_SIZE,

            "gru_layers": GRU_LAYERS,

            "dropout": DROPOUT,

            "model_type": "GRU_DELTA_SCALED_PER_ITEM",
        },
        model_path,
    )

    print()
    print(
        f"Finished {item}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Validation MAE:       "
        f"{final_mae:.6f}"
    )

    print(
        f"Validation RMSE:      "
        f"{final_rmse:.6f}"
    )

    print(
        f"Model saved to:       "
        f"{model_path}"
    )

    return {
        "item": item,
        "samples": len(y),
        "mae": final_mae,
        "rmse": final_rmse,
        "path": str(model_path),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PER-ITEM PRICE PREDICTOR TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Price scale: 1 / {PRICE_SCALE}"
    )

    print(
        f"History days: {HISTORY_DAYS}"
    )

    print(
        f"Models will be saved to: {MODEL_DIR}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_price_history()

    # --------------------------------------------------------
    # Train one model per item
    # --------------------------------------------------------

    results = []

    items = sorted(
        df["Item"].unique()
    )

    print()
    print(
        f"Found {len(items)} unique items."
    )

    for index, item in enumerate(items, 1):

        print()
        print(
            f"[{index}/{len(items)}] "
            f"Starting {item}"
        )

        item_df = df[
            df["Item"] == item
        ].copy()

        result = train_item(
            item=item,
            item_df=item_df,
            all_history_df=df,
            device=device,
        )

        if result is not None:
            results.append(result)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL TRAINING COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Models successfully trained: "
        f"{len(results)}/{len(items)}"
    )

    print(
        f"Models saved to: {MODEL_DIR}/"
    )

    print()

    if results:
        print(
            f"{'Item':<35} "
            f"{'Samples':>10} "
            f"{'MAE':>12} "
            f"{'RMSE':>12}"
        )

        print("-" * 75)

        for result in results:
            print(
                f"{result['item'][:35]:<35} "
                f"{result['samples']:>10,} "
                f"{result['mae']:>12.6f} "
                f"{result['rmse']:>12.6f}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
