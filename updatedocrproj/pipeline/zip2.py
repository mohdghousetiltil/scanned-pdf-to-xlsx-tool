import json
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

# ---------------------------------------------------------
# SET THESE TO YOUR FILE NAMES
# ---------------------------------------------------------
INPUT_PDF = "PODS.pdf"         # your big PDF
JSON_PATH = "PODS_output_1-3.json"        # your JSON
# ---------------------------------------------------------

# Regex to capture something like: "Invoice No 4040166"
INVOICE_RE = re.compile(
    r"Invoice\s+No\.?\s*([0-9]+)",
    re.IGNORECASE,
)


def build_invoice_map_from_json(json_path: str) -> dict[int, str]:
    """
    Read the JSON and build a map:
        page_index -> invoice_number (string)
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    invoice_map: dict[int, str] = {}

    pages = data.get("pages", [])
    for page in pages:
        page_index = page.get("page_index")
        lines = page.get("lines", [])

        # Join all line texts into one big string
        all_text = "\n".join(line.get("text", "") for line in lines)

        match = INVOICE_RE.search(all_text)
        if match:
            invoice_no = match.group(1)
            invoice_map[page_index] = invoice_no
            print(f"Page {page_index + 1}: found invoice {invoice_no}")
        else:
            print(f"Page {page_index + 1}: NO invoice number found in JSON")

    return invoice_map


def split_pdf_using_json(pdf_path: str, json_path: str) -> str:
    """
    Split each page of pdf_path into a single-page PDF.
    Name each file based on the invoice number from JSON.

    If the same invoice number appears on multiple pages:
        1st page: 4040268.pdf
        2nd page: 4040268(1).pdf
        3rd page: 4040268(2).pdf
        ...

    If no invoice found for a page:
        page_0001.pdf, page_0002.pdf, ...

    Returns the created ZIP file path (string).
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    # Map: page_index -> invoice_number
    invoice_map = build_invoice_map_from_json(json_path)

    # Output directory based on input name
    output_dir = Path(f"{pdf_path.stem}_invoices")
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[Path] = []

    # Track how many times we've used each base name
    # base_name -> count
    name_counts: dict[str, int] = {}

    for i, page in enumerate(reader.pages):
        # Get base name from JSON, or fallback to page number
        base_name = invoice_map.get(i)
        if not base_name:
            base_name = f"page_{i + 1:04d}"

        # How many times have we seen this base name?
        count = name_counts.get(base_name, 0)
        name_counts[base_name] = count + 1

        # First occurrence: "4040268.pdf"
        # Next: "4040268(1).pdf", then "4040268(2).pdf", ...
        if count == 0:
            filename = f"{base_name}.pdf"
        else:
            filename = f"{base_name}({count}).pdf"

        out_pdf_path = output_dir / filename

        writer = PdfWriter()
        writer.add_page(page)
        with open(out_pdf_path, "wb") as f_out:
            writer.write(f_out)

        print(f"Saved page {i + 1} as {filename}")
        generated_files.append(out_pdf_path)

    # Zip them up
    zip_path = pdf_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pdf_file in generated_files:
            zf.write(pdf_file, arcname=pdf_file.name)

    print(f"\nZIP created: {zip_path}")
    return str(zip_path)


def main():
    split_pdf_using_json(INPUT_PDF, JSON_PATH)


if __name__ == "__main__":
    main()
