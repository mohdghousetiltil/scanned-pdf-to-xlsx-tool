import json
import math
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

# ----------------------------------------------------------------------
# 1) Special pattern: remove DeliveryMethod(anything)MTO words
# ----------------------------------------------------------------------

deliverymethod_pattern = re.compile(r"^DeliveryMethod.*?MTO[^\w]*$", re.IGNORECASE)

def is_deliverymethod_mto(word: str) -> bool:
    """Return True if the word matches DeliveryMethod*MTO pattern."""
    return bool(deliverymethod_pattern.match(word))

# ----------------------------------------------------------------------
# EXTRA pattern: words starting with De* and containing Method or MTO
# ----------------------------------------------------------------------
de_method_mto_pattern = re.compile(
    r"^De.*?(Method|MTO).*?$",
    re.IGNORECASE
)

def is_de_method_or_mto(word: str) -> bool:
    """Return True if word starts with De* and contains Method or MTO."""
    return bool(de_method_mto_pattern.match(word))


# ----------------------------------------------------------------------
# 2) Text cleaner: remove redundant symbols like * - _ / ; :
# ----------------------------------------------------------------------

# 2) Text cleaner: remove redundant symbols like * - _ / ; :
#    and also quotes, brackets, carets, angle brackets, etc.
REDUNDANT_SYMBOLS_PATTERN = re.compile(r'[*_\-/;:"\[\]^<>]+')

def clean_word_text(text: str) -> str:
    """
    Remove redundant symbols like * - _ / ; : and also " [ ] ^ < > from the text.
    Extend the pattern if you want more characters removed.
    """
    text = REDUNDANT_SYMBOLS_PATTERN.sub("", text)
    text = collapse_repeated_punctuation(text)
    return text.strip()

# 3) Collapse repeated dots or commas ("..", "...", ",,", "..,,...") into a single occurrence
REPEATED_PUNCT_PATTERN = re.compile(r'[.,]{2,}')

def collapse_repeated_punctuation(text: str) -> str:
    """
    Replace sequences of 2+ dots or commas with a single dot or comma.
    Example:
    '...' -> '.'
    '..,,...' -> '.'
    ',,,' -> ','
    """
    return REPEATED_PUNCT_PATTERN.sub(lambda m: m.group(0)[0], text)

# ----------------------------------------------------------------------
# 3) Extra blacklist words & line-normalisation helpers
# ----------------------------------------------------------------------

BLACKLIST_WORDS = {
    "deiverymethodmto.",   # misspelt DeliveryMethodMTO
    "price",
    "amount",
    "gestaticied",
    "kara",
}

def is_blacklisted_word(text: str) -> bool:
    return text.lower() in BLACKLIST_WORDS


def normalize_for_match(s: str) -> str:
    """
    Normalize line text for fuzzy word matching:
      - collapse repeated dots: "..", ". ." -> "."
      - lower-case
      - remove non-word chars -> space
      - collapse multiple spaces
    """
    # collapse "..", ". .", "...", etc into single "."
    s = re.sub(r"\.\s*\.+", ".", s)
    s = re.sub(r"\.\s*\.", ".", s)

    # replace non-word chars with spaces
    s = re.sub(r"[^\w]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def word_overlap_ratio(line_text: str, canonical: str) -> float:
    """
    Return fraction of canonical words that appear in line_text.
    Both strings are normalized via normalize_for_match.
    """
    norm_line = normalize_for_match(line_text)
    norm_canon = normalize_for_match(canonical)

    line_words = set(norm_line.split())
    canon_words = norm_canon.split()

    if not canon_words:
        return 0.0

    matched = sum(1 for w in canon_words if w in line_words)
    return matched / len(canon_words)


# ----------------------------------------------------------------------
# 4) Geometry / regression utilities
# ----------------------------------------------------------------------

def word_centroid(geometry: List[List[float]]) -> Tuple[float, float]:
    """
    Compute centroid (cx, cy) from geometry [[x0, y0], [x1, y1]].
    Coordinates are assumed normalized [0,1].
    """
    (x0, y0), (x1, y1) = geometry
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return cx, cy


def line_regression(words: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """
    Simple linear regression y = m*x + b on word centroids.
    Returns (m, b, angle_degrees).
    If not enough points, slope m = 0 (horizontal).
    """
    if len(words) < 2:
        cy = words[0]["centroid"][1]
        return 0.0, cy, 0.0

    xs = [w["centroid"][0] for w in words]
    ys = [w["centroid"][1] for w in words]

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs) or 1e-9

    m = num / den
    b = y_mean - m * x_mean
    angle_deg = math.degrees(math.atan(m))
    return m, b, angle_deg


# ----------------------------------------------------------------------
# 5) Trend-based splitting within each preliminary line
# ----------------------------------------------------------------------

def split_line_by_trend(line_words: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Given words that already belong to roughly one vertical row,
    split them into physical sub-lines based on how centroid y behaves
    as x increases (increasing / decreasing / flat).

    Returns a list of sub-lists of words (each sub-list = one physical line).
    """
    if not line_words:
        return []
    if len(line_words) == 1:
        return [line_words]

    # sort left-to-right
    words = sorted(line_words, key=lambda w: w["centroid"][0])

    xs = [w["centroid"][0] for w in words]
    ys = [w["centroid"][1] for w in words]
    heights = [w["height"] for w in words]

    # basic height scale for thresholds
    h_med = statistics.median(heights) if heights else 0.01

    # thresholds relative to median height
    eps_flat = 0.10 * h_med      # near-flat
    eps_same = 0.30 * h_med      # small wobble allowed in opposite direction
    eps_break = 0.70 * h_med     # bigger jump = new line

    # detect trend from the first few deltas
    dys = [ys[i+1] - ys[i] for i in range(len(ys) - 1)]
    head_len = min(5, len(dys))
    med_dy = statistics.median(dys[:head_len]) if head_len > 0 else 0.0

    if abs(med_dy) < eps_flat:
        trend = "flat"
    elif med_dy < 0:
        trend = "decreasing"
    else:
        trend = "increasing"

    sub_lines: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [words[0]]
    last_y = ys[0]

    for w, y in zip(words[1:], ys[1:]):
        dy = y - last_y
        ok = False

        if trend == "flat":
            # allow small up/down wobble
            ok = abs(dy) <= eps_break

        elif trend == "decreasing":
            # main direction: y going down (dy <= 0)
            # accept small jitter including a tiny upward bump
            if (dy <= 0 and abs(dy) <= eps_break) or (0 < dy <= eps_same):
                ok = True

        elif trend == "increasing":
            # main direction: y going up (dy >= 0)
            if (dy >= 0 and abs(dy) <= eps_break) or (-eps_same <= dy < 0):
                ok = True

        if ok:
            current.append(w)
            last_y = y
        else:
            # start new physical line
            sub_lines.append(current)
            current = [w]
            last_y = y

    sub_lines.append(current)
    return sub_lines


# ----------------------------------------------------------------------
# 6) Main line grouping function
# ----------------------------------------------------------------------

def group_words_into_lines(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group words into lines based on vertical proximity (centroid y),
    then refine each preliminary line using per-line y-trend and regression
    so that stacked rows and far-right totals (prices) are correctly split.
    """
    if not words:
        return []

    # attach height to each word for local thresholds
    for w in words:
        (x0, y0), (x1, y1) = w["bbox"]
        w["height"] = y1 - y0

    heights = [w["height"] for w in words]
    avg_height = sum(heights) / len(heights)

    # --- A) FIRST PASS: vertical clustering by centroid y ---

    words_sorted = sorted(words, key=lambda w: w["centroid"][1])
    lines_internal: List[Dict[str, Any]] = []

    for w in words_sorted:
        cy = w["centroid"][1]
        h_w = w["height"]
        placed = False

        for line in lines_internal:
            h_line = line["avg_height"]

            # local threshold using both this word and existing line
            local_thr = 0.6 * min(h_w, h_line)
            # global floor so we don't shrink too much
            global_thr = 0.4 * avg_height
            y_threshold = max(local_thr, global_thr)

            if abs(cy - line["avg_y"]) <= y_threshold:
                n = len(line["words"])
                line["avg_y"] = (line["avg_y"] * n + cy) / (n + 1)
                line["avg_height"] = (line["avg_height"] * n + h_w) / (n + 1)
                line["words"].append(w)
                placed = True
                break

        if not placed:
            lines_internal.append(
                {
                    "avg_y": cy,
                    "avg_height": h_w,
                    "words": [w],
                }
            )

    # --- B) SECOND PASS: split each prelim line by its own y-trend ---

    new_lines_internal: List[Dict[str, Any]] = []
    for line in lines_internal:
        sub_lines = split_line_by_trend(line["words"])
        for sl in sub_lines:
            if not sl:
                continue
            ys_line = [w["centroid"][1] for w in sl]
            hs_line = [w["height"] for w in sl]
            new_lines_internal.append(
                {
                    "avg_y": sum(ys_line) / len(ys_line),
                    "avg_height": sum(hs_line) / len(hs_line),
                    "words": sl,
                }
            )

    lines_internal = new_lines_internal

    # --- C) THIRD PASS: move outliers / single-price lines to nearest line ---

    # First: single-word lines (typical case: lone "$21.45", "$39.10", "$58.80")
    for i, line in enumerate(lines_internal):
        line_words = line["words"]
        if len(line_words) != 1:
            continue

        w = line_words[0]
        cy = w["centroid"][1]
        h_w = w["height"]

        # find nearest line (by avg_y) that has more than one word
        best_j = None
        best_d = None
        for j, other in enumerate(lines_internal):
            if j == i or not other["words"] or len(other["words"]) == 1:
                continue
            d = abs(cy - other["avg_y"])
            if best_d is None or d < best_d:
                best_d = d
                best_j = j

        if best_j is not None:
            other = lines_internal[best_j]
            n = len(other["words"])
            other["avg_y"] = (other["avg_y"] * n + cy) / (n + 1)
            other["avg_height"] = (other["avg_height"] * n + h_w) / (n + 1)
            other["words"].append(w)
            line["words"] = []  # clear this line (it will be skipped later)

    # Then regression-based outlier removal for longer lines (far-right totals)
    for i, line in enumerate(lines_internal):
        line_words = line["words"]
        if len(line_words) < 4:
            continue

        # regression on this line
        m, b, _ = line_regression(line_words)

        xs = [w["centroid"][0] for w in line_words]
        ys = [w["centroid"][1] for w in line_words]
        x_min, x_max = min(xs), max(xs)

        residuals = []
        for w in line_words:
            cx, cy = w["centroid"]
            y_pred = m * cx + b
            residuals.append(cy - y_pred)

        abs_res = [abs(r) for r in residuals]
        median_abs = statistics.median(abs_res) if abs_res else 0.0

        # base threshold proportional to line height; avoid too tiny
        base_thr = 0.5 * line["avg_height"]
        dyn_thr = 3.0 * median_abs
        outlier_thr = max(base_thr, dyn_thr)

        outlier_indices = []
        for idx, (w, r_abs) in enumerate(zip(line_words, abs_res)):
            if r_abs <= outlier_thr:
                continue
            cx = w["centroid"][0]
            # only consider rightmost 25% as candidate totals
            if cx >= x_min + 0.75 * (x_max - x_min):
                outlier_indices.append(idx)

        if not outlier_indices:
            continue

        to_move = [line_words[idx] for idx in outlier_indices]

        # remove outliers from this line
        for idx in sorted(outlier_indices, reverse=True):
            del line_words[idx]

        # recompute avg_y / avg_height
        if line_words:
            ys_line = [w["centroid"][1] for w in line_words]
            hs_line = [w["height"] for w in line_words]
            line["avg_y"] = sum(ys_line) / len(ys_line)
            line["avg_height"] = sum(hs_line) / len(hs_line)
        else:
            line["avg_y"] = 0.0
            line["avg_height"] = avg_height

        # move outliers to nearest other line
        for w in to_move:
            cy = w["centroid"][1]
            h_w = w["height"]
            best_j = None
            best_d = None
            for j, other in enumerate(lines_internal):
                if j == i or not other["words"]:
                    continue
                d = abs(cy - other["avg_y"])
                if best_d is None or d < best_d:
                    best_d = d
                    best_j = j

            if best_j is None:
                # create new line if nothing suitable
                lines_internal.append(
                    {
                        "avg_y": cy,
                        "avg_height": h_w,
                        "words": [w],
                    }
                )
            else:
                other = lines_internal[best_j]
                n = len(other["words"])
                other["avg_y"] = (other["avg_y"] * n + cy) / (n + 1)
                other["avg_height"] = (other["avg_height"] * n + h_w) / (n + 1)
                other["words"].append(w)

    # --- D) BUILD FINAL OUTPUT STRUCTURE ---

    processed_lines: List[Dict[str, Any]] = []

    for line_idx, line in enumerate(lines_internal):
        if not line["words"]:
            continue

        line_words = sorted(line["words"], key=lambda w: w["centroid"][0])

        m, b, angle_deg = line_regression(line_words)

        xs = [w["centroid"][0] for w in line_words]
        ys = [w["centroid"][1] for w in line_words]
        x_mean = sum(xs) / len(xs)

        deskewed_ys = [y - m * (x - x_mean) for x, y in zip(xs, ys)]
        baseline_y = sum(deskewed_ys) / len(deskewed_ys)

        # line confidence = average of word confidences
        conf_values = [
            w["confidence"] for w in line_words
            if w.get("confidence") is not None
        ]
        line_conf = sum(conf_values) / len(conf_values) if conf_values else None

        words_out = []
        for w, y_deskew in zip(line_words, deskewed_ys):
            (x0, y0), (x1, y1) = w["bbox"]
            cx, cy = w["centroid"]

            words_out.append(
                {
                    "text": w["text"],
                    "bbox": {
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                    },
                    "centroid_original": {"x": cx, "y": cy},
                    "centroid_aligned": {"x": cx, "y": baseline_y},
                    "confidence": w["confidence"],
                }
            )

        line_text = " ".join(w["text"] for w in line_words)

        processed_lines.append(
            {
                "line_index": line_idx,
                "line_angle_degrees": angle_deg,
                "baseline_y": baseline_y,
                "text": line_text,
                "confidence": line_conf,
                "words": words_out,
            }
        )

    processed_lines.sort(key=lambda ln: ln["baseline_y"])
    return processed_lines


# ----------------------------------------------------------------------
# 7) Main pre-processing function
# ----------------------------------------------------------------------

def preprocess_pods_json(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Main entry point.
    - Reads PODS-style JSON from input_path
    - Processes each page independently
    - Writes simplified / aligned JSON to output_path
    - Returns the processed JSON object
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output = {"pages": []}

    for page_idx, page in enumerate(data.get("pages", [])):
        page_words = []

        # Flatten all words from blocks/lines
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for w in line.get("words", []):
                    raw_text = w.get("value", "")
                    
                    # OCR confidence (default to 1 if missing)
                    confidence = w.get("confidence", 1.0)

                    # 0) remove words with low OCR confidence
                    if confidence < 0.75:
                        continue

                    # 1) basic strip
                    text = raw_text.strip()

                    # 2) clean redundant symbols
                    text = clean_word_text(text)

                    # if nothing left after cleaning, skip
                    if not text:
                        continue

                    # 3) remove DeliveryMethod*MTO patterns (on cleaned text)
                    if is_deliverymethod_mto(text):
                        continue
                    
                    # 3b) remove any De* word containing Method or MTO
                    if is_de_method_or_mto(text):
                        continue


                    # 4) drop specific blacklisted words
                    if is_blacklisted_word(text):
                        continue

                    # 5) remove standalone '.' or ',' tokens
                    if text in {".", ","}:
                        continue

                    # 6) remove single letters standing alone (a, h, i, q, ...)
                    if len(text) == 1 and text.isalpha():
                        continue

                    geom = w.get("geometry")
                    if not geom or len(geom) != 2:
                        continue

                    cx, cy = word_centroid(geom)
                    page_words.append(
                        {
                            "text": text,
                            "bbox": geom,  # [[x0,y0],[x1,y1]]
                            "centroid": (cx, cy),
                            "confidence": w.get("confidence", None),
                        }
                    )

        # Group into lines, handle slant, etc.
        processed_lines = group_words_into_lines(page_words)

        # --- fix specific header and order-info lines by text pattern ---

        HEADER_CANONICAL = (
            "Item Code Item Description Pre GST Price Ord Qty Del Qty "
            "UOM % Disc GST Amount GST Inclusive"
        )

        ORDER_CANONICAL = "Order No. Account No. Terms Reference No."

        for ln in processed_lines:
            # first, collapse weird dots in the visible text
            txt = ln["text"]
            txt = re.sub(r"\.\s*\.+", ".", txt)
            txt = re.sub(r"\.\s*\.", ".", txt)
            ln["text"] = txt  # update line text with normalized dots

            # 1) Header lines: if line contains >= 40% of canonical header words
            header_ratio = word_overlap_ratio(ln["text"], HEADER_CANONICAL)
            if header_ratio >= 0.40:
                ln["text"] = HEADER_CANONICAL
                continue

            # 2) Order/Account/Terms lines: if line contains >= 50% of canonical words
            order_ratio = word_overlap_ratio(ln["text"], ORDER_CANONICAL)
            if order_ratio >= 0.60:
                ln["text"] = ORDER_CANONICAL
                continue

        output["pages"].append(
            {
                "page_index": page_idx,
                "page_id": f"page_{page_idx}",
                "dimensions": page.get("dimensions"),
                "lines": processed_lines,
            }
        )


    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


# ----------------------------------------------------------------------
# 8) CLI entry
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Adjust paths as needed
    input_json = "ocr1-3.json"
    output_json = "PODS_output_1-3.json"

    processed = preprocess_pods_json(input_json, output_json)
    print(json.dumps(processed, indent=2)[:4000])  # print first part only
