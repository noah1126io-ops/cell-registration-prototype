from __future__ import annotations

import pandas as pd


def initialize_empty_matches() -> pd.DataFrame:
    """Return the table schema planned for future cell correspondence results."""
    return pd.DataFrame(columns=["fixed_label", "moving_label", "score"])


# TODO: Implement candidate generation and scoring for fixed/moving cell correspondences.
