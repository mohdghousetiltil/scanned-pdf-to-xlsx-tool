import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

INPUT_JSON = "PODS_output_processed.json"   # change if needed
OUTPUT_XLSX = "invoices_output.xlsx"

# Canonical header (from your processed JSON)
ITEM_HEADER = (
    "Item Code Item Description Pre GST Price Ord Qty Del Qty "
    "UOM % Disc GST Amount GST Inclusive"
)

ORDER_HEADER = "Order No. Account No. Terms Reference No."

# Regex for invoice number lines
INVOICE_RE = re.compile(r"Invoice\s+No\.?\s*[-:]?\s*(\d+)", re.IGNORECASE)

# Known UOM codes (extend if needed)
UOM_SET = {
    "CTN", "CTNS",
    "TRAY", "TRAYS",
    "BAG", "BAGS",
    "BOX", "BOXES",
    "BTL", "BTLS", "BOTTLE", "BOTTLES",
    "EACH", "EA",
    "PK", "PKT", "PACK", "CASE",
    "TUB", "DRUM", "PCE", "PCS",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def extract_numeric_value(token: str) -> Optional[str]:
    """
    Try to pull a numeric value out of a messy token.

    Handles:
    - '$63.EJ'    -> '63'
    - '$53,20'    -> '53.20'
    - '$79,10'    -> '79.10'
    - '$17.ic'    -> '17'
    - 'S6.53'     -> '6.53'
    - '0.00'      -> '0.00'
    - '63.'       -> '63'
    """
    m = re.search(r"(\d+[.,]?\d*)", token)
    if not m:
        return None

    val = m.group(1).replace(",", ".")
    # strip trailing dot if like "63."
    if val.endswith("."):
        val = val[:-1]
    return val if val else None


def extract_invoice_no(line_objs: List[Dict]) -> Optional[str]:
    """
    Use x-coordinates to find 'Invoice No' and grab the number to the right.

    Strategy:
    - For each line, sort its words by centroid x.
    - Look for 'Invoice' followed by 'No' or 'No.'.
    - Take the first word to the right that contains digits -> that's the invoice no.
    - If that fails, fall back to the regex on full line text.
    """

    # 1) Geometry-based search
    for ln in line_objs:
        words = ln.get("words") or []
        if not words:
            continue

        # sort words left-to-right by centroid x
        words_sorted = sorted(
            words,
            key=lambda w: w.get("centroid_original", {}).get("x", 0.0)
        )
        n = len(words_sorted)

        for i, w in enumerate(words_sorted):
            text = (w.get("text") or "").strip()
            t_norm = text.lower()

            # Case A: 'Invoice' then separate 'No' / 'No.'
            if t_norm == "invoice" and i + 1 < n:
                next_text = (words_sorted[i + 1].get("text") or "").strip().lower()
                if next_text in ("no", "no."):
                    base_x = words_sorted[i + 1].get(
                        "centroid_original", {}
                    ).get("x", 0.0)

                    # first numeric-ish word to the right
                    for j in range(i + 2, n):
                        wj = words_sorted[j]
                        xj = wj.get("centroid_original", {}).get("x", 0.0)
                        if xj <= base_x:
                            continue

                        cand = wj.get("text") or ""
                        m = re.search(r"(\d+)", cand)
                        if m:
                            return m.group(1)

            # Case B: 'InvoiceNo' / 'Invoice-No.' stuck together
            compact = re.sub(r"[^\w]", "", text).lower()
            if compact.startswith("invoiceno"):
                base_x = w.get("centroid_original", {}).get("x", 0.0)
                for j in range(i + 1, n):
                    wj = words_sorted[j]
                    xj = wj.get("centroid_original", {}).get("x", 0.0)
                    if xj <= base_x:
                        continue

                    cand = wj.get("text") or ""
                    m = re.search(r"(\d+)", cand)
                    if m:
                        return m.group(1)

    # 2) Fallback: old regex on whole line text
    for ln in line_objs:
        text = ln.get("text", "") or ""
        m = INVOICE_RE.search(text)
        if m:
            return m.group(1)

    return None



def extract_reference_no(lines: List[str]) -> Optional[str]:
    """
    Find line 'Order No. Account No. Terms Reference No.'
    and take the last reference-like token on the next line as Reference No.

    Now supports mixed alphanumeric references such as '1234567A'.
    """
    for idx, text in enumerate(lines):
        if text.strip() == ORDER_HEADER and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            # grab tokens that are letters/digits; pick the last one that has at least one digit
            candidates = re.findall(r"([A-Za-z0-9]+)", next_line)
            for cand in reversed(candidates):
                if any(ch.isdigit() for ch in cand):
                    return cand
            if candidates:
                return candidates[-1]
    return None



def looks_like_item_candidate(text: str) -> bool:
    """
    Heuristic: does this line look like an item, even if we can't parse it?
    Used for flagging.
    """
    s = text.strip()
    if not s or s == ITEM_HEADER:
        return False

    tokens = s.split()
    if len(tokens) < 4:
        return False

    # first token looks like a code: alnum, no spaces
    if not re.match(r"^[A-Za-z0-9]+$", tokens[0]):
        return False

    upper = s.upper()
    # must have at least one UOM word or "OUT OF STOCK", or multiple numeric-ish bits
    if any(u in upper for u in UOM_SET) or "OUT OF STOCK" in upper:
        return True

    # else, count numeric matches
    nums = re.findall(r"\d+[.,]?\d*", s)
    return len(nums) >= 3


def parse_item_line(
    line_obj: Dict,
    code_col_right_x: Optional[float],
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Parse an item line into fields, using the header layout:
    Item Code | Item Description | Pre GST Price | Ord Qty | Del Qty | UOM | % Disc | GST Amount | GST Inclusive

    We now optionally use x-coordinates to determine the *Item Code* value:
    - If `code_col_right_x` is provided (taken from the x1 of the 'Code' word
      in the header line), then all words in this line whose right edge is
      <= that x-coordinate are treated as part of the Item Code cell.
    - Everything to the right is parsed with the old token-based logic.

    If `code_col_right_x` is None, we fall back to the old behaviour where
    the first token is the Item Code.

    Returns (row_dict, error_reason).
      - If row_dict is not None, the line parsed successfully.
      - If row_dict is None but error_reason is not None, it *looked* like an item
        but we couldn't parse it; this will be flagged.
      - If both are None, it's not an item line at all.
    """
    text = (line_obj.get("text") or "").strip()
    if not text:
        return None, None

    # Skip the header line itself
    if text == ITEM_HEADER:
        return None, None

    words = line_obj.get("words") or []

    # Base tokenisation: left-to-right by centroid x
    if words:
        words_sorted = sorted(
            words,
            key=lambda w: w.get("centroid_original", {}).get("x", 0.0),
        )
        base_tokens = []
        for w in words_sorted:
            t = (w.get("text") or "").strip()
            if t:
                base_tokens.append(t)
    else:
        words_sorted = []
        base_tokens = text.split()

    if len(base_tokens) < 5:
        return None, None

    # Geometry-aware Item Code (only if we know the header's Code x1)
    work_tokens = base_tokens[:]
    code = None

    if code_col_right_x is not None and words_sorted:
        code_words = []
        other_words = []

        for w in words_sorted:
            t = (w.get("text") or "").strip()
            if not t:
                continue

            bbox = w.get("bbox") or {}
            x0 = bbox.get("x0")
            x1 = bbox.get("x1")

            # Fallbacks if bbox missing
            if x0 is None or x1 is None:
                cx = w.get("centroid_original", {}).get("x", 0.0)
                if x0 is None:
                    x0 = cx
                if x1 is None:
                    x1 = cx

            # STRICT rule:
            # - If the whole word is left of or touching Code.x1 -> Item Code
            # - If the word starts strictly to the right of Code.x1 -> next column (Item Desc / later)
            # - If it crosses the boundary (x0 <= Code.x1 < x1) we treat it as NOT Item Code
            if x1 <= code_col_right_x:
                # entire word is within the Item Code column
                code_words.append(t)
            else:
                # either x0 > Code.x1 OR it crosses over the boundary
                # in both cases it belongs to the next column(s)
                other_words.append(t)

        if code_words:
            code = " ".join(code_words)
            work_tokens = [code] + other_words
        else:
            # No words under Code column on this line -> fallback to old rule
            code = base_tokens[0]
            work_tokens = base_tokens[:]
    else:
        # Old rule: first token is Item Code
        code = base_tokens[0]
        work_tokens = base_tokens[:]


    # Detect OUT OF STOCK (keep phrase for GST Inclusive)
    joined_upper = " ".join(work_tokens).upper()
    out_of_stock_flag = "OUT OF STOCK" in joined_upper
    gst_inclusive = "OUT OF STOCK" if out_of_stock_flag else None

    # For parsing numeric fields, ignore trailing OUT / OF / STOCK tokens
    end_idx = len(work_tokens) - 1
    if out_of_stock_flag:
        while end_idx >= 0 and work_tokens[end_idx].upper() in ("OUT", "OF", "STOCK"):
            end_idx -= 1

    # If not OUT OF STOCK, GST Inclusive is usually the last amount-ish token
    if not out_of_stock_flag:
        found_amount = False
        for i in range(end_idx, 0, -1):
            val = extract_numeric_value(work_tokens[i])
            if val is not None:
                gst_inclusive = work_tokens[i]  # keep raw text (including $ etc.)
                end_idx = i - 1
                found_amount = True
                break
        if not found_amount:
            # no amount found => might still be an item candidate, but unparseable
            if looks_like_item_candidate(text):
                return None, "could_not_find_gst_inclusive_amount"
            return None, None

    i = end_idx

    # GST Amount
    gst_amount_val = None
    while i >= 1:
        val = extract_numeric_value(work_tokens[i])
        if val is not None:
            gst_amount_val = val
            i -= 1
            break
        i -= 1

    # % Disc
    pct_disc_val = None
    while i >= 1:
        # try % like "0.00%" first
        m = re.match(r"^(\d+[.,]?\d*)%$", work_tokens[i])
        if m:
            pct_disc_val = m.group(1).replace(",", ".")
            i -= 1
            break

        val = extract_numeric_value(work_tokens[i])
        if val is not None:
            pct_disc_val = val
            i -= 1
            break
        i -= 1

    # UOM
    uom = None
    while i >= 1:
        tok = work_tokens[i]
        if tok.upper() in UOM_SET:
            uom = tok
            i -= 1
            break
        i -= 1

    # Del Qty
    del_qty_val = None
    while i >= 1:
        val = extract_numeric_value(work_tokens[i])
        if val is not None:
            del_qty_val = val
            i -= 1
            break
        i -= 1

    # Ord Qty
    ord_qty_val = None
    while i >= 1:
        val = extract_numeric_value(work_tokens[i])
        if val is not None:
            ord_qty_val = val
            i -= 1
            break
        i -= 1

    # Pre GST Price
    pre_gst_price_val = None
    idx_pre = None
    while i >= 1:
        val = extract_numeric_value(work_tokens[i])
        if val is not None:
            pre_gst_price_val = val
            idx_pre = i
            i -= 1
            break
        i -= 1

    # Require key numeric fields + UOM
    missing_fields = []
    if pre_gst_price_val is None:
        missing_fields.append("Pre GST Price")
    if ord_qty_val is None:
        missing_fields.append("Ord Qty")
    if del_qty_val is None:
        missing_fields.append("Del Qty")
    if uom is None:
        missing_fields.append("UOM")

    if missing_fields:
        if looks_like_item_candidate(text):
            return None, "missing_fields: " + ", ".join(missing_fields)
        return None, None

    # Everything between code and Pre GST Price is description
    desc_tokens = work_tokens[1:idx_pre]
    item_desc = " ".join(desc_tokens).strip()

    row = {
        "Item Code": code,
        "Item Description": item_desc,
        "Pre GST Price": pre_gst_price_val,
        "Ord Qty": ord_qty_val,
        "Del Qty": del_qty_val,
        "UOM": uom,
        "% Disc": pct_disc_val if pct_disc_val is not None else "",
        "GST Amount": gst_amount_val if gst_amount_val is not None else "",
        "GST Inclusive": gst_inclusive if gst_inclusive is not None else "",
    }
    return row, None



def extract_items_from_page(
    line_objs: List[Dict],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Find the item header line, then parse all following item lines.

    We also locate the x1 coordinate of the word "Code" in the item header
    and use that as the right boundary of the Item Code column.
    """
    items: List[Dict[str, str]] = []
    flagged: List[Dict[str, str]] = []

    lines_text = [ln.get("text", "") for ln in line_objs]

    # locate header
    header_idx = None
    for idx, text_line in enumerate(lines_text):
        if text_line.strip() == ITEM_HEADER:
            header_idx = idx
            break

    if header_idx is None:
        return items, flagged

    # Determine Item Code column right boundary from the header,
    # using the x1 coordinate of the word "Code" (if present).
    code_col_right_x: Optional[float] = None
    header_line = line_objs[header_idx]
    for w in header_line.get("words") or []:
        t = (w.get("text") or "").strip().lower()
        if t == "code":
            bbox = w.get("bbox") or {}
            x1 = bbox.get("x1")
            if x1 is None:
                x1 = w.get("centroid_original", {}).get("x", 0.0)
            code_col_right_x = x1
            break

    # parse lines after header
    for ln in line_objs[header_idx + 1:]:
        text_line = (ln.get("text") or "").strip()
        upper = text_line.upper()

        # stop at totals
        if upper.startswith(("SUBTOTAL", "SUB TOTAL", "TOTAL", "INVOICE TOTAL")):
            break

        row, err = parse_item_line(ln, code_col_right_x)
        if row is not None:
            items.append(row)
        elif err is not None:
            flagged.append({"Raw Line": ln.get("text", ""), "Reason": err})

    return items, flagged



# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def json_to_xlsx(input_json: str, output_xlsx: str) -> None:
    input_path = Path(input_json)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])

    all_rows: List[Dict[str, str]] = []         # all parsed rows
    flagged_rows: List[Dict[str, str]] = []     # all flagged lines

    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        for page in pages:
            page_index = page.get("page_index")
            line_objs = page.get("lines", [])
            lines_text = [ln.get("text", "") for ln in line_objs]

            invoice_no = extract_invoice_no(line_objs)
            reference_no = extract_reference_no(lines_text)
            items, flagged = extract_items_from_page(line_objs)


            # enrich flagged rows with invoice/meta for troubleshooting
            for fl in flagged:
                fl["Invoice No"] = invoice_no or ""
                fl["Reference No"] = reference_no or ""
                fl["Page Index"] = page_index
                flagged_rows.append(fl)

            if not items:
                continue

            rows = []
            for itm in items:
                row = {
                    "Invoice No": invoice_no or "",
                    "Reference No": reference_no or "",
                }
                row.update(itm)
                rows.append(row)
                all_rows.append(row)  # add to global list

            df = pd.DataFrame(rows)

            # Sheet name: invoice number if present, else page index
            if invoice_no:
                sheet_name = str(invoice_no)
            else:
                sheet_name = f"page_{page_index}"

            # Excel sheet names max 31 chars
            sheet_name = sheet_name[:31]

            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # --- write the combined "All Invoices" sheet ---
        if all_rows:
            df_all = pd.DataFrame(all_rows)
            df_all.to_excel(writer, sheet_name="All_Invoices", index=False)

        # --- write flagged rows sheet ---
        if flagged_rows:
            df_flagged = pd.DataFrame(flagged_rows)[
                ["Invoice No", "Reference No", "Page Index", "Raw Line", "Reason"]
            ]
            df_flagged.to_excel(writer, sheet_name="Flagged_Rows", index=False)

    print(f"Written Excel file: {output_xlsx}")


if __name__ == "__main__":
    json_to_xlsx(INPUT_JSON, OUTPUT_XLSX)
