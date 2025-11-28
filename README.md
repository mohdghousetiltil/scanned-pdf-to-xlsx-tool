# 📄 Scanned PDF → Excel Extraction Engine (v2.0)
### AI-Powered Modular OCR Pipeline for Converting Messy PDFs into Structured Excel Data

This upgraded version of the extraction engine brings a fully **modular architecture**, improved preprocessing, enhanced OCR accuracy, and a cleaner user experience. Designed for real-world, messy scanned PDFs, the system outputs clean, analysis-ready Excel files and searchable PDFs.

## Interface Overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/e168ba3b-4517-46dc-90ee-9bae020cf3ac" width="49%" />
  <img src="https://github.com/user-attachments/assets/4be7bbc0-949a-456f-a2aa-d43fafe85be9" width="49%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6748f9ea-03a0-4112-b7be-ceba2b2163e6" width="80%" />
</p>

---

## 🚀 What’s New in v2.0
- Fully modular pipeline (`pipeline/` directory)
- Improved preprocessing and orientation correction
- Enhanced docTR OCR using modern detection/recognition models
- Stronger JSON cleanup + line grouping logic
- Better handling of messy lines and fragmented text
- Cleaner web UI (`static/index.html`)
- Organized output directory (`runs/`)
- New `main.py` entry point for improved simplicity

---

## 🔍 What This System Does
- Accepts **any scanned PDF** (multi-page, rotated, skewed, noisy)
- Applies deep preprocessing: deskew, thresholding, denoising
- Uses **Tesseract OSD** for orientation detection
- Performs high-accuracy OCR using **docTR (DB + PARSeq)**
- Cleans and groups OCR JSON into meaningful rows
- Converts raw OCR results into structured Excel sheets
- Generates:
  - `processed_output.xlsx` — structured tabular output
  - `searchable_output.pdf` — OCR-enhanced PDF
- Lightweight browser-based uploader for simple usage

---

## 🧩 High-Level Pipeline
```
User Upload (index.html)
        │
        ▼
Preprocessing
 • Orientation detection
 • Deskew + resize
 • Threshold + noise removal
        │
        ▼
OCR Extraction (docTR)
 • DB text detection
 • PARSeq recognition
        │
        ▼
JSON Cleanup (welllwelll3.py)
 • Noise filtering
 • Line grouping
 • Key-value detection
 • Confidence scoring
        │
        ▼
Excel Generation (jsontoxlsx3.py)
 • Header sheet
 • Line items sheet
 • Flagged/low-confidence rows
        │
        ▼
Output (runs/)
 • processed_output.xlsx
 • searchable_output.pdf
```

---

## 🛠️ Tech Stack
| Component | Technology |
|----------|------------|
| OCR Engine | docTR (DB + PARSeq) |
| Orientation Detection | Tesseract OSD |
| Image Processing | OpenCV |
| Backend | Python |
| Frontend | HTML, JavaScript |
| Data Processing | Pandas, NumPy |
| Excel Output | XlsxWriter |
| PDF Output | reportlab |

---

## 📁 Updated Project Structure
```
updatedocrproj/
│
├── main.py                     # Main backend entry point
├── static/
│   └── index.html              # Upload interface UI
│
├── pipeline/
│   ├── __init__.py
│   ├── docTR_pp2.py            # OCR + preprocessing
│   ├── welllwelll3.py          # JSON cleanup & grouping
│   ├── jsontoxlsx3.py          # Excel writer
│   └── zip2.py                 # ZIP export helper
│
├── runs/                       # Output files (auto-generated)
└── .venv/                      # Local virtual environment (ignored)
```

---

## ▶️ How to Use
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the application
```bash
python main.py
```

### 3. Open the interface
```
http://localhost:5000
```
Upload your scanned PDF → receive Excel + searchable PDF in the `runs/` folder.

---

## 🎯 Business Impact
- Reduces manual data-entry time from **days to minutes**
- Handles extremely noisy and low-quality scans with high accuracy
- Produces structured, reliable Excel output for reporting, billing, validation
- Improves consistency and eliminates human error in data entry

---

## 🧭 Future Enhancements
- Batch PDF processing
- Transformer-based handwriting OCR
- Automatic table boundary detection
- REST API for enterprise integrations
- Interactive corrections interface

---

This version delivers a **faster, cleaner, smarter** OCR pipeline for any organization handling high volumes of scanned documents.
