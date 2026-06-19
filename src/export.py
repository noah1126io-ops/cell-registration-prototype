from __future__ import annotations

import pandas as pd


def matches_to_csv(matches: pd.DataFrame) -> bytes:
    """Serialize a future match table to CSV bytes."""
    return matches.to_csv(index=False).encode("utf-8")


# TODO: Add export for transforms, paired-cell tables, and rendered quality-control images.
