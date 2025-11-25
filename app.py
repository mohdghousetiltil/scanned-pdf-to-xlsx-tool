from flask import Flask, request, jsonify
from pathlib import Path
from tempfile import TemporaryDirectory
import base64
import threading
import webbrowser

# Import your real pipeline functions
from docTR_pp import ocr_pdf
from welllwelll2 import preprocess_pods_json
from jsontoxlsx import json_to_xlsx

app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    # Serve static/index.html and disable caching
    resp = app.send_static_file("index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/process", methods=["POST"])
def process_pdf():
    """
    A (PDF upload) ->
      docTR_pp.ocr_pdf -> B (raw JSON + searchable PDF)
      welllwelll2.preprocess_pods_json -> C (clean JSON)
      jsontoxlsx.json_to_xlsx -> D (XLSX)
    Returns base64 for B (pdf) and D (xlsx).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # === A: Save input PDF ===
            input_pdf_path = tmpdir / "input.pdf"
            file.save(input_pdf_path)

            # === B: docTR_pp -> raw JSON + searchable PDF ===
            raw_json_path = tmpdir / "docTR_output.json"
            searchable_pdf_path = tmpdir / "searchable_output.pdf"

            ocr_pdf(
                input_pdf=str(input_pdf_path),
                json_output=str(raw_json_path),
                pdf_output=str(searchable_pdf_path),
                scale=3,
                dpi_for_pdf=300,
                use_preprocessing=True,
            )

            # === C: welllwelll2 -> cleaned JSON ===
            processed_json_path = tmpdir / "processed_output.json"
            preprocess_pods_json(
                input_path=str(raw_json_path),
                output_path=str(processed_json_path),
            )

            # === D: jsontoxlsx -> XLSX ===
            xlsx_path = tmpdir / "processed_output.xlsx"
            json_to_xlsx(
                input_json=str(processed_json_path),
                output_xlsx=str(xlsx_path),
            )

            # Convert outputs to base64 strings for frontend
            with open(xlsx_path, "rb") as f:
                xlsx_b64 = base64.b64encode(f.read()).decode("utf-8")
            with open(searchable_pdf_path, "rb") as f:
                pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

            return jsonify({
                "xlsx_base64": xlsx_b64,  # D
                "pdf_base64": pdf_b64,    # B
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def open_browser():
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
