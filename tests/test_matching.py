from src.matching import initialize_empty_matches


def test_initialize_empty_matches_schema():
    matches = initialize_empty_matches()

    assert list(matches.columns) == ["fixed_label", "moving_label", "score"]
    assert matches.empty
