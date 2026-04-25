"""Stage 1 cleaning — meta/review cleaning and 5-core filtering."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _clean_list_field(val: Any) -> str:
    """Flatten a list field to a '. '-joined string; None/NaN → ''."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, list):
        return ". ".join(str(v).strip() for v in val if v)
    return str(val).strip()


def _normalise_whitespace(text: str) -> str:
    """Collapse multiple whitespace/newline characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def clean_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, deduplicate by parent_asin, flatten list fields, normalise whitespace."""
    df = df.copy()

    # 1. Drop null identifiers
    before = len(df)
    df = df.dropna(subset=["parent_asin"])
    df["parent_asin"] = df["parent_asin"].str.strip()
    df = df[df["parent_asin"] != ""]
    logger.info("Dropped %d rows with null/empty parent_asin", before - len(df))

    # 2. Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["parent_asin"], keep="first")
    logger.info("Removed %d duplicate meta entries", before - len(df))

    # 3. Flatten list fields
    for col in ["description", "features", "categories"]:
        if col in df.columns:
            df[col] = df[col].apply(_clean_list_field)

    # 4. Normalise whitespace in all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].apply(
            lambda v: _normalise_whitespace(str(v)) if isinstance(v, str) else v
        )

    # 5. Ensure title is never null (fallback to parent_asin)
    if "title" in df.columns:
        mask = df["title"].isna() | (df["title"] == "")
        df.loc[mask, "title"] = df.loc[mask, "parent_asin"]

    # 6. Normalise price column: store as float (extract first number from strings
    #    like "from 14.99", "$29.99", etc.); keep NaN for unparseable values.
    if "price" in df.columns:
        def _parse_price(val: Any) -> float | None:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            # Extract first numeric value from the string
            match = re.search(r"[\d]+(?:\.\d+)?", str(val).replace(",", ""))
            return float(match.group()) if match else None

        df["price"] = df["price"].apply(_parse_price)

    df = df.reset_index(drop=True)
    logger.info("Cleaned meta: %d items", len(df))
    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, deduplicate per user-item pair, normalise text and ratings."""
    df = df.copy()

    # 1. Drop null identifiers / ratings
    before = len(df)
    df = df.dropna(subset=["user_id", "parent_asin", "rating"])
    df["user_id"] = df["user_id"].str.strip()
    df["parent_asin"] = df["parent_asin"].str.strip()
    logger.info("Dropped %d rows with null user_id/parent_asin/rating", before - len(df))

    # Cast rating to float, drop out-of-range values
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    before = len(df)
    df = df[df["rating"].between(1.0, 5.0)]
    logger.info("Dropped %d rows with invalid rating", before - len(df))

    # 2. Deduplicate: keep the most recent review per user–item pair
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp", ascending=False)
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "parent_asin"], keep="first")
    logger.info("Removed %d duplicate user–item reviews", before - len(df))

    # 3. Fill missing text
    if "text" in df.columns:
        df["text"] = df["text"].fillna("").apply(
            lambda v: _normalise_whitespace(str(v))
        )

    df = df.reset_index(drop=True)
    logger.info("Cleaned reviews: %d records", len(df))
    return df


def apply_k_core(
    df: pd.DataFrame,
    k: int = 5,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
    max_passes: int = 20,
) -> pd.DataFrame:
    """Iterative k-core pruning — removes sparse users/items until stable."""
    df = df.copy()
    for pass_num in range(1, max_passes + 1):
        before = len(df)

        # Remove items with < k interactions
        item_counts = df[item_col].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df[item_col].isin(valid_items)]

        # Remove users with < k interactions
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df[user_col].isin(valid_users)]

        after = len(df)
        dropped = before - after
        logger.info(
            "5-core pass %d: removed %d rows → %d remaining "
            "(%d users, %d items)",
            pass_num, dropped, after,
            df[user_col].nunique(), df[item_col].nunique(),
        )
        if dropped == 0:
            logger.info("5-core converged after %d pass(es).", pass_num)
            break
    else:
        logger.warning("5-core did not fully converge within %d passes.", max_passes)

    return df.reset_index(drop=True)


# fallback: derive train from reviews when the official CSV isn't available
def derive_train_from_reviews(
    reviews: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "parent_asin",
) -> pd.DataFrame:
    """Build train split from reviews by excluding valid/test interactions."""
    # Build set of (user, item) pairs that are in valid or test
    held_out: set[tuple[str, str]] = set()
    for split_df in (valid, test):
        held_out.update(zip(split_df[user_col], split_df[item_col]))

    # Sort reviews by timestamp so we can build ordered histories
    rev = reviews.copy()
    if "timestamp" in rev.columns:
        rev = rev.sort_values(["user_id", "timestamp"])

    # Build history map: user → ordered list of parent_asin values
    history_map: dict[str, list[str]] = (
        rev.groupby(user_col)[item_col].apply(list).to_dict()
    )

    # Filter out held-out interactions
    mask = [
        (row[user_col], row[item_col]) not in held_out
        for _, row in rev.iterrows()
    ]
    train = rev[mask].copy()

    # Attach history column: items the user interacted with *before* this row
    def _build_history(group: pd.DataFrame) -> pd.Series:
        items = group[item_col].tolist()
        histories = []
        seen: list[str] = []
        for item in items:
            histories.append(" ".join(seen))  # history = everything seen so far
            seen.append(item)
        return pd.Series(histories, index=group.index)

    train["history"] = train.groupby(user_col, group_keys=False).apply(
        _build_history
    )

    # Ensure schema matches valid/test
    keep_cols = [user_col, item_col, "rating", "timestamp", "history"]
    keep_cols = [c for c in keep_cols if c in train.columns]
    train = train[keep_cols].reset_index(drop=True)

    logger.info(
        "Derived train split: %d interactions, %d users, %d items",
        len(train), train[user_col].nunique(), train[item_col].nunique(),
    )
    return train
