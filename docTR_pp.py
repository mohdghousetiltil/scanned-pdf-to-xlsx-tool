from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np
import torch
from PIL import Image

from doctr.io import read_pdf
from doctr.models import ocr_predictor
from ocrmypdf.hocrtransform import HocrTransform
from PyPDF2 import PdfMerger

import cv2
import pytesseract


# ------------------------------
#  Helper: JSON serialization
# ------------------------------

def to_serializable(obj):
    """Recursively convert numpy types in docTR export to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


# ------------------------------
#  Orientation detection
# ------------------------------

def auto_orient_page(img_rgb: np.ndarray) -> np.ndarray:
    """
    Use Tesseract OSD to detect page rotation (0, 90, 180, 270)
    and return an upright RGB image.
    img_rgb: HxWx3 uint8 RGB
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    try:
        osd = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
        rotation = int(osd.get("rotate", 0))  # 0, 90, 180, 270
        print(f"Detected rotation: {rotation}°")
    except Exception as e:
        print(f"OSD failed, leaving page as-is: {e}")
        return img_rgb

    # IMPORTANT: this mapping is the opposite of what you had before
    if rotation == 0:
        return img_rgb
    elif rotation == 90:
        # Tesseract says "rotate 90°" -> we rotate 90° CLOCKWISE (k=3)
        return np.rot90(img_rgb, k=3)
    elif rotation == 180:
        # Same both directions
        return np.rot90(img_rgb, k=2)
    elif rotation == 270:
        # Tesseract says "rotate 270°" -> we rotate 90° COUNTER-CLOCKWISE (k=1)
        return np.rot90(img_rgb, k=1)
    else:
        print(f"Unexpected rotation angle {rotation}, leaving as-is.")
        return img_rgb



# ------------------------------
#  (Optional) extra preprocessing
# ------------------------------

def preprocess_page(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extra cleanup (deskew-ish, thresholding) AFTER orientation is fixed.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    h, w = thr.shape
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        thr = cv2.resize(thr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR)

    thr_rgb = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
    return thr_rgb


# ------------------------------
#  Global model (load once)
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL = ocr_predictor(
    det_arch="db_resnet50",
    reco_arch="parseq",
    pretrained=True,
    det_bs=4,
    reco_bs=256,
    assume_straight_pages=True,
    export_as_straight_boxes=True,
    preserve_aspect_ratio=True,
).to(device)


# ------------------------------
#  Core OCR function
# ------------------------------

def ocr_pdf(
    input_pdf: str,
    json_output: str | None = None,
    pdf_output: str | None = None,
    scale: int = 3,
    dpi_for_pdf: int = 300,
    use_preprocessing: bool = True,
) -> None:
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    print(f"\n=== Processing: {input_path.name} ===")

    # 1) Rasterize PDF → list of numpy arrays (RGB)
    print("Rasterizing PDF (higher DPI)...")
    pages = read_pdf(str(input_path), scale=scale)  # list[np.ndarray], RGB

    print(f"Total pages: {len(pages)}")

    print("Auto-orienting pages with Tesseract OSD...")
    pages = [auto_orient_page(p) for p in pages]


    # 3) Optional extra preprocessing (threshold, upscale, etc.)
    if use_preprocessing:
        print("Preprocessing pages (threshold, upscale)...")
        pages = [preprocess_page(p) for p in pages]

    # 4) Run OCR once on all pages
    print("Running OCR with docTR...")
    result = MODEL(pages)

    # 5) JSON export
    if json_output is not None:
        print(f"Exporting OCR result to JSON: {json_output}")
        raw_data = result.export()
        data = to_serializable(raw_data)
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("✅ JSON saved")

    # 6) OCR PDF export (hOCR → searchable PDF)
    if pdf_output is not None:
        print(f"Exporting OCR result to searchable PDF: {pdf_output}")

        xml_outputs = result.export_as_xml()
        merger = PdfMerger()

        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for i, ((xml_bytes, _xml_tree), img) in enumerate(zip(xml_outputs, pages)):
                img_path = tmpdir / f"page_{i}.jpg"
                xml_path = tmpdir / f"page_{i}.xml"
                pdf_page_path = tmpdir / f"page_{i}.pdf"

                # Save oriented (+preprocessed) page image
                Image.fromarray(img).save(img_path)

                # Save hOCR XML
                xml_path.write_text(xml_bytes.decode("utf-8"), encoding="utf-8")

                # Convert hOCR + image → searchable PDF page
                hocr = HocrTransform(hocr_filename=str(xml_path), dpi=dpi_for_pdf)
                hocr.to_pdf(
                    out_filename=str(pdf_page_path),
                    image_filename=str(img_path),
                )

                # Windows-safe: open → append → close
                with open(pdf_page_path, "rb") as f:
                    merger.append(f)

        with open(pdf_output, "wb") as f_out:
            merger.write(f_out)

        print("✅ Searchable OCR PDF saved")

    print("=== Done ===\n")


# ------------------------------
#  Main: example usage
# ------------------------------

if __name__ == "__main__":
    ocr_pdf(
        input_pdf="pdfs/input.pdf",
        json_output="jsons/input.json",
        pdf_output="pdfs/input_ocr.pdf",
        scale=3,
        dpi_for_pdf=300,
        use_preprocessing=True,
    )
