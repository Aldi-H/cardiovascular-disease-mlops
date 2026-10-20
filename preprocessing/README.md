# Otomasi preprocessing

Notebook eksperimen manual berbasis template ada di
[`../notebooks/Cardiovascular_Disease_MSML.ipynb`](../notebooks/Cardiovascular_Disease_MSML.ipynb).
Entry point sesuai nama siswa adalah `automate_Aldi-Himawan.py`; implementasi
preprocessing kanonis dan pengujian berada di `../src/preprocessing.py` dan
`../tests/test_preprocessing.py`.

Workflow `.github/workflows/mlflow-train-publish.yml` mengunduh dataset raw dari
B2, menjalankan entry point ini, lalu mengunggah seluruh output `data/clean/`
ke bucket B2 private. File disimpan pada prefix versi
`datasets/preprocessed/runs/<run-id>-<commit-sha>/`; manifest
`datasets/preprocessed/latest.json` menunjuk ke hasil terbaru. Dataset tidak
diunggah sebagai GitHub Actions artifact atau di-commit ke Git.
