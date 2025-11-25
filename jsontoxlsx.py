import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

INPUT_JSON = "PODS2_output_processed.json"   # change if needed
OUTPUT_XLSX = "invoices2_output.xlsx"

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


def extract_invoice_no(lines: List[str]) -> Optional[str]:
    """Find 'Invoice No 4047480' etc anywhere on the page."""
    for text in lines:
        m = INVOICE_RE.search(text)
        if m:
            return m.group(1)
    return None


def extract_reference_no(lines: List[str]) -> Optional[str]:
    """
    Find line 'Order No. Account No. Terms Reference No.'
    and take the last number on the next line as Reference No.
    (e.g. 'CHILL ALLOCATION ... 4386595' -> 4386595)
    """
    for idx, text in enumerate(lines):
        if text.strip() == ORDER_HEADER and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            # last group of digits in the line
            m = re.findall(r"(\d+)", next_line)
            if m:
                return m[-1]
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


def parse_item_line(text: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Parse an item line into fields, using the header layout:
    Item Code | Item Description | Pre GST Price | Ord Qty | Del Qty | UOM | % Disc | GST Amount | GST Inclusive

    Returns (row_dict, error_reason).
      - If row_dict is not None, the line parsed successfully.
      - If row_dict is None but error_reason is not None, it *looked* like an item
        but we couldn't parse it; this will be flagged.
      - If both are None, it's not an item line at all.
    """
    s = text.strip()
    if not s:
        return None, None

    # Skip the header line itself
    if s == ITEM_HEADER:
        return None, None

    tokens = s.split()
    if len(tokens) < 5:
        return None, None

    work_tokens = tokens[:]  # for numeric parsing
    code = work_tokens[0]

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


def extract_items_from_page(lines: List[str]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Find the item header line, then parse all following item lines.

    Returns (items, flagged_rows) where:
      - items is a list of parsed rows
      - flagged_rows is a list of dicts with raw line + reason
    """
    items: List[Dict[str, str]] = []
    flagged: List[Dict[str, str]] = []

    # locate header
    header_idx = None
    for idx, text in enumerate(lines):
        if text.strip() == ITEM_HEADER:
            header_idx = idx
            break

    if header_idx is None:
        return items, flagged

    # parse lines after header
    for text in lines[header_idx + 1:]:
        upper = text.upper().strip()

        # stop at totals
        if upper.startswith(("SUBTOTAL", "SUB TOTAL", "TOTAL", "INVOICE TOTAL")):
            break

        row, err = parse_item_line(text)
        if row is not None:
            items.append(row)
        elif err is not None:
            flagged.append({"Raw Line": text, "Reason": err})

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

            invoice_no = extract_invoice_no(lines_text)
            reference_no = extract_reference_no(lines_text)
            items, flagged = extract_items_from_page(lines_text)

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
