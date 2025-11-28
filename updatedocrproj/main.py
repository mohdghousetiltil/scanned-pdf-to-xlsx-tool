import uuid
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ------- IMPORT YOUR PIPELINE CODE -------
from pipeline.docTR_pp2 import ocr_pdf
from pipeline.welllwelll3 import preprocess_pods_json
from pipeline.jsontoxlsx3 import json_to_xlsx
from pipeline.zip2 import split_pdf_using_json


# ------- PATHS & APP SETUP -------
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
RUNS_DIR = BASE_DIR / "runs"

RUNS_DIR.mkdir(exist_ok=True)

app = FastAPI()

# Serve static files (if you add CSS/JS/assets later)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve generated output files (PDF/XLSX/ZIP)
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")


# ------- FRONT PAGE --------
@app.get("/")
async def index():
    """Serve the main HTML UI."""
    return FileResponse(STATIC_DIR / "index.html")


# ------- PIPELINE ENDPOINT --------
@app.post("/api/process")
async def process_pdf(file: UploadFile = File(...)):
    """
    A (PDF) -> docTR_pp2 -> B (searchable PDF) + C (raw JSON)
    C -> welllwelll3 -> D (processed JSON)
    D -> jsontoxlsx3 -> E (XLSX)
    B + D -> zip2 -> F (ZIP)
    Returns URLs for B, E, F.
    """

    # 1) Make a unique folder for this run
    run_id = uuid.uuid4().hex
    run_folder = RUNS_DIR / run_id
    run_folder.mkdir()

    # 2) Save uploaded PDF (A)
    input_pdf_path = run_folder / file.filename
    with input_pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 3) A -> B + C (OCR)
    searchable_pdf = run_folder / "searchable.pdf"   # B
    raw_json = run_folder / "ocr_raw.json"           # C

    ocr_pdf(
        input_pdf=str(input_pdf_path),
        json_output=str(raw_json),
        pdf_output=str(searchable_pdf),
        scale=3,
        dpi_for_pdf=300,
        use_preprocessing=False,   # or True if you prefer
    )

    # 4) C -> D (clean JSON)
    processed_json = run_folder / "processed.json"   # D
    preprocess_pods_json(str(raw_json), str(processed_json))

    # 5) D -> E (XLSX)
    xlsx_path = run_folder / "invoices.xlsx"         # E
    json_to_xlsx(str(processed_json), str(xlsx_path))

    # 6) B + D -> F (ZIP)
    zip_path_str = split_pdf_using_json(str(searchable_pdf), str(processed_json))
    zip_path = Path(zip_path_str)                    # F

    # 7) Return URLs for frontend
    return JSONResponse({
        "runId": run_id,
        "pdf": f"/runs/{run_id}/{searchable_pdf.name}",
        "xlsx": f"/runs/{run_id}/{xlsx_path.name}",
        "zip": f"/runs/{run_id}/{zip_path.name}",
    })


# ------- ENTRY POINT --------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
