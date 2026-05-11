"""
download_data.py
----------------
Downloads the UCI Student Performance dataset (student-mat.csv).

This dataset records 395 Portuguese students taking a Math course,
along with their grades and 30+ background features (study time,
family info, absences, etc.).

Source: https://archive.ics.uci.edu/dataset/320/student+performance
Run this ONCE before running analysis.py or app.py.
"""

import os
import urllib.request
import zipfile

# 1) Make sure the "data" folder exists. If not, create it.
os.makedirs("data", exist_ok=True)

CSV_PATH = os.path.join("data", "student-mat.csv")

# 2) If the CSV already exists, do nothing (saves time).
if os.path.exists(CSV_PATH):
    print(f"Dataset already exists at {CSV_PATH}. Skipping download.")
else:
    # 3) Download the ZIP file from UCI's official archive.
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    zip_path = os.path.join("data", "student.zip")

    print("Downloading dataset from UCI...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")

    # 4) Unzip it. The zip contains another zip ("student.zip") with the CSVs.
    print("Extracting outer zip...")
    with zipfile.ZipFile(zip_path, "r") as outer:
        outer.extractall("data")

    inner_zip = os.path.join("data", "student.zip")
    if os.path.exists(inner_zip):
        print("Extracting inner zip...")
        with zipfile.ZipFile(inner_zip, "r") as inner:
            inner.extractall("data")

    # 5) The CSV uses ';' as the separator (semicolons), not commas.
    #    We'll handle that when we read it in analysis.py and app.py.
    if os.path.exists(CSV_PATH):
        print(f"Dataset ready at {CSV_PATH}")
    else:
        print("Something went wrong. Please download manually from:")
        print("https://archive.ics.uci.edu/dataset/320/student+performance")
