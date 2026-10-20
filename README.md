# Cardiovascular Disease — MSML MLOps

> **Tujuan:** membangun sistem machine learning yang memprediksi ada/tidaknya penyakit kardiovaskular (`cardio`) dari data klinis dan gaya hidup, sebagai **alat skrining awal**. Prediksi model bukan diagnosis dan tidak menggantikan tenaga kesehatan.

- GitHub: [Aldi-H/cardiovascular-disease-mlops](https://github.com/Aldi-H/cardiovascular-disease-mlops)
- DagsHub / MLflow: [Eksperimen Cardiovascular Disease](https://dagshub.com/aldihimawan88/cardiovascular-disease-mlops/experiments#/)
- Docker Hub: [cardiovascular-disease-api](https://hub.docker.com/repository/docker/kuro1905/cardiovascular-disease-api/general) dan [cardiovascular-disease-mlflow](https://hub.docker.com/repository/docker/kuro1905/cardiovascular-disease-mlflow/general).

## Sumber dan lisensi dataset

- Dataset: **Cardiovascular Disease dataset**, diterbitkan oleh Svetlana Ulianova (`sulianova`) di [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
- Berisi sekitar 70.000 pemeriksaan pasien; file sumber Kaggle bernama `cardio_train.csv` dan menggunakan delimiter titik koma (`;`).
- **Lisensi pada halaman Kaggle: Unknown.** Pastikan syarat penggunaan/redistribusi terkini dengan pemilik dataset sebelum memublikasikan data atau turunannya. Data mentah tidak boleh di-commit ke Git.
- Salinan lokal yang ditemukan di repo: `data/raw/cardiovascular-disease.csv`. Data mentah di `data/raw/` dan hasil preprocessing di `data/clean/` diabaikan Git; path data dapat diubah lewat `CARDIO_DATA_PATH`.

### Kolom dataset

| Kolom | Arti | Catatan |
|---|---|---|
| `id` | ID baris/pasien pada dataset | Identifier; bukan prediktor yang sebaiknya dipakai model |
| `age` | Umur | Integer dalam **hari**, bukan tahun |
| `gender` | Kode kategori gender | Kode kategori; jangan dianggap ordinal tanpa alasan |
| `height` | Tinggi badan | cm; periksa nilai ekstrem |
| `weight` | Berat badan | kg; periksa nilai ekstrem |
| `ap_hi` | Tekanan darah sistolik | Periksa nilai tidak masuk akal dan ekstrem |
| `ap_lo` | Tekanan darah diastolik | Periksa nilai tidak masuk akal dan ekstrem; validasi relasinya dengan `ap_hi` |
| `cholesterol` | Kategori kolesterol | 1 normal, 2 di atas normal, 3 jauh di atas normal |
| `gluc` | Kategori glukosa | 1 normal, 2 di atas normal, 3 jauh di atas normal |
| `smoke` | Kebiasaan merokok | Biner (0/1) |
| `alco` | Konsumsi alkohol | Biner (0/1) |
| `active` | Aktivitas fisik | Biner (0/1) |
| `cardio` | Target ada/tidaknya penyakit kardiovaskular | Biner (0/1) |

Definisi dan pengkodean kolom perlu dicocokkan kembali dengan data card sumber saat EDA. Kolom `id` tidak boleh masuk ke fitur model. Dataset ini observasional; korelasi/prediksi tidak membuktikan sebab-akibat.

## Struktur proyek

```text
.
├── data/
│   ├── raw/                  # Data sumber lokal; diabaikan Git
│   └── clean/                # Hasil preprocessing lokal; diabaikan Git
├── notebooks/                # Notebook eksplorasi
├── src/                      # Preprocessing dan training
├── preprocessing/             # Entry point automasi Kriteria 1
│   ├── automate_Aldi-Himawan.py
│   └── README.md
├── serving/                  # API inferensi
├── monitoring/               # Prometheus, Grafana, simulator
├── tests/                    # Pengujian
├── .github/workflows/        # Workflow GitHub Actions
├── MLproject                 # Entry point MLflow di root repository
├── MLProject/                # MLflow Project untuk workflow submission
│   ├── MLproject
│   ├── conda.yaml
│   ├── download_dataset.py
│   └── modelling.py
├── conda.yaml                # Environment conda Python 3.12.13
├── requirements.txt          # Versi dependensi Python yang dipin
├── .env.example              # Template variabel environment, tanpa secret
├── Membangun_model/
│   └── screenshots/           # Bukti screenshot Kriteria 2
├── Monitoring dan Logging/    # Folder bukti screenshot Kriteria 4
└── README.md
```

Notebook eksperimen berbasis template: [`notebooks/Cardiovascular_Disease_MSML.ipynb`](notebooks/Cardiovascular_Disease_MSML.ipynb). Notebook tersebut memuat data loading, EDA, dan eksplorasi preprocessing manual.

### Pemetaan ke kriteria Dicoding

- **Kriteria 1 — eksperimen dan preprocessing:** notebook eksperimen ada di `notebooks/`; entry point `preprocessing/automate_Aldi-Himawan.py` memakai implementasi kanonis `src/preprocessing.py`. Workflow retraining mengunggah hasil `data/clean/` ke B2 private di `datasets/preprocessed/runs/<run-id>-<commit>/`; `datasets/preprocessed/latest.json` menunjuk hasil terbaru.
- **Kriteria 2 — model:** baseline MLflow autolog ada di `src/train_baseline.py`; tuning dan manual logging ada di `src/train_tuning.py`. Bukti screenshot dapat dimasukkan ke `Membangun_model/screenshots/` dengan nama `screenshoot_dashboard.jpg` dan `screenshoot_artifak.jpg`.
- **Kriteria 3 — CI:** MLflow Project ada di `MLProject/`; workflow retraining dan publikasi ada di `.github/workflows/mlflow-train-publish.yml`. Artifact workflow disimpan sebagai GitHub Actions artifact.
- **Kriteria 4 — monitoring:** Prometheus config ada di `monitoring/prometheus.yml`; endpoint serving, inference, dan ekspor metrik Prometheus ada di `serving/app.py`; provisioning Grafana ada di `monitoring/grafana/provisioning/`. Folder screenshot tersedia di `Monitoring dan Logging/`.

### Bukti screenshot

Masukkan screenshot sendiri—folder di bawah sengaja disiapkan kosong. Hapus `.gitkeep` setelah menambahkan screenshot. Jangan memasukkan `.env`, password, token, atau URL webhook Discord ke screenshot.

```text
Membangun_model/
└── screenshots/
    ├── screenshoot_dashboard.jpg
    └── screenshoot_artifak.jpg

Monitoring dan Logging/
├── 1.bukti_serving/
├── 4.bukti monitoring Prometheus/
├── 5.bukti monitoring Grafana/
└── 6.bukti alerting Grafana/
    ├── rules/
    └── notifications/
```

Workflow membuat repo `kuro1905/cardiovascular-disease-mlflow` dan `kuro1905/cardiovascular-disease-api`; tag image menggunakan 12 karakter awal commit SHA. Buka tab **Tags** pada masing-masing halaman untuk melihat image yang dipublikasikan.

## Persiapan environment (WSL2 Ubuntu)

Gunakan Conda/Mamba dan Python **3.12.13**. Versi aktif mesin pengembang dapat berbeda; verifikasi versi environment setelah aktivasi.

```bash
conda env create -f conda.yaml
conda activate cardiovascular-mlops
python --version
python -m pip --version
python -m pip check
```

Versi yang diharapkan: `Python 3.12.13`, pip dapat dijalankan, dan `pip check` berakhir tanpa konflik (`No broken requirements found.`). Untuk membuka notebook:

```bash
jupyter lab
```

> Jika environment dengan nama sama sudah ada, jangan hapus environment tersebut tanpa memeriksa isinya; pilih nama environment baru atau kelola secara manual.

## Konfigurasi lokal

```bash
cp .env.example .env
```

Isi nilai bertanda `<ISI_SENDIRI>` di `.env` lokal saja. File `.env` diabaikan Git. Jangan menaruh token, password, application key, atau credential di source code, notebook, README, maupun commit; untuk CI gunakan GitHub Secrets. URI DagsHub yang disiapkan adalah `https://dagshub.com/aldihimawan88/cardiovascular-disease-mlops.mlflow`; autentikasi memerlukan akun/repo yang dapat diakses.

**Registry image:** Docker Hub digunakan untuk memenuhi kriteria CI/CD Dicoding. Credential Docker Hub hanya disimpan sebagai GitHub Secrets, bukan di repository.

## Rancangan evaluasi dan logging

- Evaluasi menggunakan stratified train/test split dan **5-fold Stratified CV**.
- Metrik utama: **Recall** dan **F1**; false negative dianggap lebih berisiko untuk konteks skrining. Metrik pendukung: ROC-AUC, PR-AUC, accuracy, precision, log loss, dan waktu training.
- Preprocessing harus di-fit hanya pada fold/train yang sesuai (pipeline CV dan train split) untuk mencegah data leakage. Test set hanya untuk evaluasi akhir.
- MLflow Tracking UI online menggunakan DagsHub. Logging manual tetap wajib selain autolog, termasuk metrik tambahan dan artefak evaluasi.
- Artifact model/hasil eksperimen dan dataset preprocessing disimpan di Backblaze B2 melalui endpoint S3-compatible; credential dibaca dari environment. Dataset preprocessing disimpan per-run dan dirujuk oleh manifest `datasets/preprocessed/latest.json`.
- Serving direncanakan dengan FastAPI dalam Docker, dipantau Prometheus dan Grafana.

## Keputusan data yang masih perlu dibuktikan saat Tahap 1

Nilai tekanan darah, tinggi, dan berat dapat memiliki outlier atau nilai tidak valid. Ambang pembersihan tidak akan dipilih diam-diam: distribusi dan jumlah data yang terdampak akan diperiksa, aturan dijelaskan, serta dampaknya pada split/train dievaluasi. Konversi umur hari ke tahun dan fitur BMI juga akan didokumentasikan. Untuk prediksi skrining, pemilihan threshold dan trade-off Recall/precision harus dilaporkan.

## Privasi dan batasan

Dataset publik bukan berarti bebas digunakan untuk segala tujuan. Jangan unggah data mentah ke Git atau image Docker tanpa dasar izin yang sesuai. Dataset mentah dan hasil preprocessing disimpan di bucket B2 private, bukan di repository GitHub. Sistem ini proyek pembelajaran, tidak divalidasi untuk pemakaian klinis, dan berpotensi bias terhadap populasi di luar distribusi dataset.
