#!/bin/bash
# Download script for Time Series Forecasting Benchmark Datasets
# Used in: Informer, Autoformer, PatchTST, iTransformer, etc.
#
# Datasets:
#   ETTh1, ETTh2    - Electricity Transformer Temperature (hourly)
#   ETTm1, ETTm2    - Electricity Transformer Temperature (15-min)
#   Weather          - 21 meteorological indicators
#   Electricity      - 321 clients hourly consumption
#   Traffic          - 862 sensors hourly road occupancy
#   ILI              - Influenza-like illness weekly ratios
#   Exchange         - Daily exchange rates of 8 countries

set -e

DATA_DIR="$(cd "$(dirname "$0")" && pwd)/data"
mkdir -p "$DATA_DIR"

echo "============================================"
echo " Time Series Benchmark Dataset Downloader"
echo "============================================"
echo ""
echo "Target directory: $DATA_DIR"
echo ""

# ---------- Method 1: Clone from Autoformer repo (most reliable) ----------
REPO_URL="https://github.com/zhouhaoyi/ETDataset.git"
TSLIB_URL="https://github.com/thuml/Time-Series-Library.git"

download_from_gdrive() {
    FILE_ID=$1
    OUTPUT=$2
    # Using gdown for Google Drive downloads
    if command -v gdown &> /dev/null; then
        gdown "$FILE_ID" -O "$OUTPUT"
    else
        echo "[!] gdown not found. Installing..."
        pip install gdown -q
        gdown "$FILE_ID" -O "$OUTPUT"
    fi
}

# ---- ETT Datasets (from original ETDataset repo) ----
echo "[1/7] Downloading ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2)..."
ETT_DIR="$DATA_DIR/ETT-small"
mkdir -p "$ETT_DIR"

for name in ETTh1 ETTh2 ETTm1 ETTm2; do
    if [ -f "$ETT_DIR/${name}.csv" ]; then
        echo "  - ${name}.csv already exists, skipping."
    else
        wget -q --show-progress \
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/${name}.csv" \
            -O "$ETT_DIR/${name}.csv"
        echo "  - ${name}.csv downloaded."
    fi
done

# ---- Weather Dataset ----
# echo ""
# echo "[2/7] Downloading Weather dataset..."
# WEATHER_DIR="$DATA_DIR/weather"
# mkdir -p "$WEATHER_DIR"

# if [ -f "$WEATHER_DIR/weather.csv" ]; then
#     echo "  - weather.csv already exists, skipping."
# else
#     wget -q --show-progress \
#         "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/weather.csv" \
#         -O "$WEATHER_DIR/weather.csv" 2>/dev/null || {
#         # Fallback: download from Time-Series-Library or autoformer
#         echo "  - Primary source failed. Trying alternative source..."
#         download_from_gdrive "1aKEBqFSqGBGwFq4zzPMOEB0TfLHOYmDv" "$WEATHER_DIR/weather.csv.zip"
#         if [ -f "$WEATHER_DIR/weather.csv.zip" ]; then
#             unzip -o "$WEATHER_DIR/weather.csv.zip" -d "$WEATHER_DIR/" && rm "$WEATHER_DIR/weather.csv.zip"
#         fi
#     }
#     echo "  - weather.csv downloaded."
# fi

# ---- Electricity Dataset ----
echo ""
echo "[3/7] Downloading Electricity dataset..."
ELEC_DIR="$DATA_DIR/electricity"
mkdir -p "$ELEC_DIR"

if [ -f "$ELEC_DIR/electricity.csv" ]; then
    echo "  - electricity.csv already exists, skipping."
else
    wget -q --show-progress \
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz" \
        -O "$ELEC_DIR/electricity.txt.gz" 2>/dev/null && {
        gunzip -f "$ELEC_DIR/electricity.txt.gz"
        echo "  - electricity.txt downloaded."
    } || {
        echo "  - Primary source failed. Trying Google Drive..."
        download_from_gdrive "1rUPdR7R2iWFW-LMZqo8nqAE1S1f3Jnc-" "$ELEC_DIR/electricity.csv"
        echo "  - electricity.csv downloaded."
    }
fi

# ---- Traffic Dataset ----
echo ""
echo "[4/7] Downloading Traffic dataset..."
TRAFFIC_DIR="$DATA_DIR/traffic"
mkdir -p "$TRAFFIC_DIR"

if [ -f "$TRAFFIC_DIR/traffic.csv" ]; then
    echo "  - traffic.csv already exists, skipping."
else
    wget -q --show-progress \
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/traffic/traffic.txt.gz" \
        -O "$TRAFFIC_DIR/traffic.txt.gz" 2>/dev/null && {
        gunzip -f "$TRAFFIC_DIR/traffic.txt.gz"
        echo "  - traffic.txt downloaded."
    } || {
        echo "  - Primary source failed. Trying Google Drive..."
        download_from_gdrive "1LE_hhJgemKbGjQXGOSY7AV7sSlgQjGZl" "$TRAFFIC_DIR/traffic.csv"
        echo "  - traffic.csv downloaded."
    }
fi

# ---- ILI (Influenza-Like Illness) Dataset ----
# echo ""
# echo "[5/7] Downloading ILI dataset..."
# ILI_DIR="$DATA_DIR/illness"
# mkdir -p "$ILI_DIR"

# if [ -f "$ILI_DIR/national_illness.csv" ]; then
#     echo "  - national_illness.csv already exists, skipping."
# else
#     wget -q --show-progress \
#         "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/national_illness.csv" \
#         -O "$ILI_DIR/national_illness.csv" 2>/dev/null || {
#         echo "  - Primary source failed. Trying Google Drive..."
#         download_from_gdrive "1UbDd8klLd7Y_h8nI_D1R5-xJPPUvuiRR" "$ILI_DIR/national_illness.csv"
#     }
#     echo "  - national_illness.csv downloaded."
# fi

# ---- Exchange Rate Dataset ----
echo ""
echo "[6/7] Downloading Exchange Rate dataset..."
EXCHANGE_DIR="$DATA_DIR/exchange_rate"
mkdir -p "$EXCHANGE_DIR"

if [ -f "$EXCHANGE_DIR/exchange_rate.csv" ]; then
    echo "  - exchange_rate.csv already exists, skipping."
else
    wget -q --show-progress \
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz" \
        -O "$EXCHANGE_DIR/exchange_rate.txt.gz" 2>/dev/null && {
        gunzip -f "$EXCHANGE_DIR/exchange_rate.txt.gz"
        echo "  - exchange_rate.txt downloaded."
    } || {
        echo "  - Primary source failed. Trying Google Drive..."
        download_from_gdrive "1TP4ycd4WvQxnrHV8OLKTdPYjGb_UpJPL" "$EXCHANGE_DIR/exchange_rate.csv"
        echo "  - exchange_rate.csv downloaded."
    }
fi

# ---- Alternative: All-in-one from Time-Series-Library ----
echo ""
echo "[7/7] Downloading all-in-one dataset archive (backup)..."
ALL_IN_ONE="$DATA_DIR/all_six_datasets.zip"

if [ -f "$ALL_IN_ONE" ] || [ -f "$DATA_DIR/.all_downloaded" ]; then
    echo "  - All-in-one archive already exists, skipping."
else
    echo "  - Attempting Google Drive download (all datasets bundled)..."
    download_from_gdrive "1CC4ZrUD4EKncndzgy99FKz1iu8IExflR" "$ALL_IN_ONE" 2>/dev/null && {
        echo "  - Extracting..."
        unzip -o "$ALL_IN_ONE" -d "$DATA_DIR/"
        touch "$DATA_DIR/.all_downloaded"
        rm -f "$ALL_IN_ONE"
        echo "  - All-in-one archive extracted."
    } || {
        echo "  - [SKIP] Google Drive download failed. Individual downloads above should suffice."
    }
fi

# ---- Summary ----
echo ""
echo "============================================"
echo " Download Summary"
echo "============================================"
echo ""
echo "Directory structure:"
find "$DATA_DIR" -type f -name "*.csv" -o -name "*.txt" | sort | while read f; do
    SIZE=$(du -h "$f" | cut -f1)
    echo "  $SIZE  $f"
done

echo ""
echo "Dataset details:"
echo "  ETTh1/h2  - 7 features,  ~17,420 rows (hourly, 2y)"
echo "  ETTm1/m2  - 7 features,  ~69,680 rows (15-min, 2y)"
echo "  Weather   - 21 features, ~52,696 rows (10-min, 1y)"
echo "  Electricity - 321 features, ~26,304 rows (hourly, 3y)"
echo "  Traffic   - 862 features, ~17,544 rows (hourly, 2y)"
echo "  ILI       - 7 features,  ~966 rows (weekly, 18y)"
echo "  Exchange  - 8 features,  ~7,588 rows (daily, 30y)"
echo ""
echo "Done! Ready for LIV-TiDAR time series experiments."