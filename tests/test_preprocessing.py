import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_dataframe, split_and_transform


def make_raw_dataframe(n_rows: int = 100) -> pd.DataFrame:
    row = np.arange(n_rows)

    return pd.DataFrame(
        {
            "id": row,
            "age": 15_000 + row * 20,
            "gender": (row % 2) + 1,
            "height": 150 + (row % 30),
            "weight": 50 + ((row * 7) % 70),
            "ap_hi": 110 + (row % 50),
            "ap_lo": 60 + (row % 30),
            "cholesterol": (row % 3) + 1,
            "gluc": ((row + 1) % 3) + 1,
            "smoke": row % 2,
            "alco": (row // 2) % 2,
            "active": (row + 1) % 2,
            "cardio": row % 2,
        }
    )


def test_clean_dataframe_applies_filters_and_creates_features():
    raw = make_raw_dataframe()
    raw.loc[0, "ap_hi"] = 0
    raw.loc[1, "ap_hi"] = 100
    raw.loc[1, "ap_lo"] = 120
    raw.loc[2, "height"] = 230
    raw.loc[3, "weight"] = 20
    original = raw.copy(deep=True)

    cleaned = clean_dataframe(raw)

    assert len(cleaned) == len(raw) - 4
    assert "id" not in cleaned.columns
    assert "age" not in cleaned.columns
    assert "age_years" in cleaned.columns
    assert "bmi" in cleaned.columns
    assert "cardio" in cleaned.columns

    assert cleaned.loc[4, "age_years"] == pytest.approx(
        raw.loc[4, "age"] / 365.25
    )
    expected_bmi = raw.loc[4, "weight"] / (raw.loc[4, "height"] / 100) ** 2
    assert cleaned.loc[4, "bmi"] == pytest.approx(expected_bmi)

    # Cleaning must not mutate the raw DataFrame supplied by the caller.
    pd.testing.assert_frame_equal(raw, original)


def test_clean_dataframe_rejects_missing_required_column():
    raw = make_raw_dataframe().drop(columns=["smoke"])

    with pytest.raises(ValueError, match="Kolom wajib tidak ditemukan"):
        clean_dataframe(raw)


def test_clean_dataframe_rejects_missing_values():
    raw = make_raw_dataframe()
    raw.loc[0, "weight"] = np.nan

    with pytest.raises(ValueError, match="Nilai kosong ditemukan"):
        clean_dataframe(raw)


def test_split_is_stratified_and_scaler_fits_train_only():
    cleaned = clean_dataframe(make_raw_dataframe())

    X_train_processed, X_test_processed, y_train, y_test, preprocessor = (
        split_and_transform(cleaned, test_size=0.2, random_state=42)
    )

    assert X_train_processed.shape == (80, 17)
    assert X_test_processed.shape == (20, 17)
    assert y_train.shape == (80,)
    assert y_test.shape == (20,)
    assert y_train.value_counts().to_dict() == {0: 40, 1: 40}
    assert y_test.value_counts().to_dict() == {0: 10, 1: 10}

    X = cleaned.drop(columns=["cardio"])
    y = cleaned["cardio"]
    X_train_reference, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    expected_train_age_mean = X_train_reference["age_years"].mean()
    fitted_train_age_mean = preprocessor.named_transformers_["num"].mean_[0]

    assert fitted_train_age_mean == pytest.approx(expected_train_age_mean)
