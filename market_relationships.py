"""Direct WarEra production-chain relationships used as lagged model context."""

import re

import pandas as pd


RELATED_ITEMS = {
    "ammo": ("lead", "lightammo", "heavyammo"),
    "bread": ("grain",),
    "concrete": ("limestone",),
    "cookedfish": ("fish",),
    "fish": ("cookedfish",),
    "grain": ("bread",),
    "heavyammo": ("lead", "ammo", "lightammo"),
    "iron": ("steel",),
    "lead": ("lightammo", "ammo", "heavyammo"),
    "lightammo": ("lead", "ammo", "heavyammo"),
    "limestone": ("concrete",),
    "livestock": ("steak",),
    "mysteriousplant": ("pills",),
    "oil": ("petroleum",),
    "paper": ("wood",),
    "petroleum": ("oil",),
    "pills": ("mysteriousplant",),
    "steak": ("livestock",),
    "steel": ("iron",),
    "wood": ("paper",),
}


def normalize_item_name(value):
    """Return a punctuation- and whitespace-independent item key."""

    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def related_percentage_changes(history_df, item):
    """Return the mean same-day percentage movement of directly related items.

    Only movements already known on each historical day are used. This keeps the
    feature useful to training without exposing the model to future prices.
    """

    related = set(RELATED_ITEMS.get(normalize_item_name(item), ()))
    if not related:
        return pd.Series(dtype="float64")

    frame = history_df[["Date", "Item", "Market Midpoint"]].copy()
    frame["Item Key"] = frame["Item"].map(normalize_item_name)
    frame = frame[frame["Item Key"].isin(related)]
    if frame.empty:
        return pd.Series(dtype="float64")

    frame["Date"] = pd.to_datetime(frame["Date"])
    daily = frame.pivot_table(
        index="Date",
        columns="Item Key",
        values="Market Midpoint",
        aggfunc="last",
    ).sort_index()

    return daily.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
