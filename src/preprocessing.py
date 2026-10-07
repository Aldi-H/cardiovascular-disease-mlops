import pandas as pd

CANDIDATE_BOUNDS = {
    "ap_hi": (70, 250),
    "ap_lo": (40, 150),
    "height": (120, 220),
    "weight": (30, 250),
}

REQUIRED_COLUMNS = {
    "id",
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
}


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(raw.columns)
    
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}")

    missing_values = raw[list(REQUIRED_COLUMNS)].isna().sum()
    columns_with_missing = missing_values[missing_values > 0]
    
    if not columns_with_missing.empty:
        raise ValueError(
            "Nilai kosong ditemukan; tentukan kebijakan penanganannya terlebih dahulu: "
            f"{columns_with_missing.to_dict()}"
        )

    valid_mask = (
        raw["ap_hi"].gt(0)
        & raw["ap_lo"].gt(0)
        & raw["ap_hi"].ge(raw["ap_lo"])
    )

    for column, (lower, upper) in CANDIDATE_BOUNDS.items():
        valid_mask &= raw[column].between(lower, upper, inclusive="both")

    cleaned = raw.loc[valid_mask].copy()
    cleaned["age_years"] = cleaned["age"] / 365.25
    cleaned["bmi"] = cleaned["weight"] / (cleaned["height"] / 100) ** 2

    return cleaned.drop(columns=["id", "age"])
