"""Draw the entity-relationship diagram of `data/warehouse/eicu.db`.

Reads `data/schema.json` (produced by `build_database.py`) and writes
`docs/schema_er.svg`. No dependency: the SVG is written by hand.

The visual choice is plain. No icons, no shadows, no gradients. A thin rule, one
colour per domain, and a one-line description under each table name. Placement is
explicit — the database is a star, with 28 tables pointing at `patient` — which
avoids the plate of spaghetti an automatic layout produces.

Usage:
    python scripts/draw_schema.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "data" / "schema.json"
OUTPUT = ROOT / "docs" / "schema_er.svg"

# --- What each table holds --------------------------------------------------
DESCRIPTIONS = {
    "patient": "ICU stay: demographics, admission, discharge",
    "hospital": "Facility: region, size, teaching status",
    "medication": "Drugs prescribed during the stay",
    "admissiondrug": "Treatments taken before admission",
    "infusiondrug": "Continuous infusions and rates",
    "allergy": "Declared allergies",
    "treatment": "Procedures and treatments performed",
    "lab": "Routine laboratory results",
    "customlab": "Non-standard laboratory results",
    "microlab": "Microbiology cultures and organisms",
    "diagnosis": "Diagnoses made, coded in ICD-9",
    "admissiondx": "Reason for ICU admission",
    "pasthistory": "Medical history",
    "apacheapsvar": "Physiological variables of the APACHE score",
    "apachepatientresult": "APACHE score and predicted mortality",
    "apachepredvar": "Predictive variables of the APACHE score",
    "vitalperiodic": "Vitals recorded automatically (every 5 min)",
    "vitalaperiodic": "Vitals recorded occasionally",
    "nursecharting": "Timestamped nursing observations",
    "nurseassessment": "Structured nursing assessments",
    "nursecare": "Nursing care delivered",
    "physicalexam": "Physical examinations",
    "intakeoutput": "Fluid intake and output",
    "careplangeneral": "Care plan: general items",
    "careplangoal": "Care plan goals",
    "careplaneol": "End-of-life discussions",
    "careplancareprovider": "Care providers and specialties",
    "careplaninfectiousdisease": "Infectious-disease part of the care plan",
    "respiratorycare": "Mechanical ventilation and airways",
    "respiratorycharting": "Recorded respiratory parameters",
    "note": "Structured clinical notes",
}

# --- Business grouping -----------------------------------------------------------
DOMAINS = [
    ("DRUGS", "#b45309", [
        "medication", "admissiondrug", "infusiondrug", "allergy", "treatment"]),
    ("LABS", "#0369a1", ["lab", "customlab", "microlab"]),
    ("DIAGNOSIS AND SEVERITY", "#6d28d9", [
        "diagnosis", "admissiondx", "pasthistory",
        "apacheapsvar", "apachepatientresult", "apachepredvar"]),
    ("MONITORING", "#047857", [
        "vitalperiodic", "vitalaperiodic", "nursecharting",
        "nurseassessment", "nursecare", "physicalexam", "intakeoutput"]),
    ("CARE PLAN", "#a16207", [
        "careplangeneral", "careplangoal", "careplaneol",
        "careplancareprovider", "careplaninfectiousdisease"]),
    ("RESPIRATORY", "#be123c", ["respiratorycare", "respiratorycharting"]),
    ("NOTES", "#475569", ["note"]),
]

# Columns worth showing, beyond the primary and foreign keys.
FEATURED = {
    "medication": ["drugname", "dosage", "routeadmin"],
    "admissiondrug": ["drugname", "drugdosage"],
    "infusiondrug": ["drugname", "drugrate"],
    "allergy": ["allergyname", "allergytype"],
    "treatment": ["treatmentstring"],
    "lab": ["labname", "labresult"],
    "customlab": ["labothername", "labotherresult"],
    "microlab": ["culturesite", "organism"],
    "diagnosis": ["diagnosisstring", "icd9code"],
    "admissiondx": ["admitdxname", "admitdxpath"],
    "pasthistory": ["pasthistoryvalue", "pasthistorypath"],
    "apacheapsvar": ["heartrate", "creatinine", "wbc"],
    "apachepatientresult": ["apachescore", "predictedicumortality"],
    "apachepredvar": ["age", "admitdiagnosis"],
    "vitalperiodic": ["heartrate", "sao2", "observationoffset"],
    "vitalaperiodic": ["noninvasivesystolic", "noninvasivediastolic"],
    "nursecharting": ["nursingchartcelltypevalname", "nursingchartvalue"],
    "nurseassessment": ["celllabel", "cellattributevalue"],
    "nursecare": ["celllabel", "cellattributevalue"],
    "physicalexam": ["physicalexampath", "physicalexamvalue"],
    "intakeoutput": ["celllabel", "cellvaluenumeric"],
    "careplangeneral": ["cplgroup", "cplitemvalue"],
    "careplangoal": ["cplgoalcategory", "cplgoalvalue"],
    "careplaneol": ["cpleoldiscussionoffset"],
    "careplancareprovider": ["specialty", "interventioncategory"],
    "careplaninfectiousdisease": ["infectdiseasesite", "responsetotherapy"],
    "respiratorycare": ["airwaytype", "ventstartoffset"],
    "respiratorycharting": ["respcharttypecat", "respchartvalue"],
    "note": ["notetype", "notepath"],
    "patient": ["gender", "age", "ethnicity", "unittype", "hospitaldischargestatus"],
    "hospital": ["region", "numbedscategory", "teachingstatus"],
}

MAX_ROWS = 5            # columns shown per table, keys included

# --- Geometry ---------------------------------------------------------------------
BOX_W = 250
CENTRE_W = 300          # patient and hospital, slightly wider
TITLE_H = 22            # the table name
DESC_H = 15             # the description
ROW_H = 15              # one column
PAD = 9
GAP_V = 14
GAP_DOMAIN = 26
MARGIN = 36
GUTTER = 60
COL_W = max(BOX_W, CENTRE_W) + GUTTER          # horizontal step, never < the widest box

LAYOUT = [
    ["DRUGS", "LABS"],
    ["DIAGNOSIS AND SEVERITY"],
    ["__CENTRE__"],
    ["MONITORING"],
    ["CARE PLAN", "RESPIRATORY", "NOTES"],
]

BORDER = "#d8dee8"
MUTED = "#a3adbb"
INK = "#111827"
TEXT = "#4b5563"


def esc(t) -> str:
    return html.escape(str(t), quote=True)


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def short_type(t: str) -> str:
    t = (t or "").upper()
    for prefix, short in (("VARCHAR", "text"), ("TEXT", "text"), ("CHAR", "text"),
                           ("BIGINT", "int"), ("SMALLINT", "int"), ("INTEGER", "int"),
                           ("INT", "int"), ("NUMERIC", "num"), ("DOUBLE", "num"),
                           ("REAL", "num"), ("FLOAT", "num"), ("BOOL", "bool")):
        if t.startswith(prefix):
            return short
    return (t.split("(")[0][:4] or "?").lower()


def rows_of(table: str, info: dict) -> list[tuple[str, str, str]]:
    """(name, type, marker): primary key, then foreign keys, then featured columns."""
    by_name = {c["name"].lower(): c for c in info["cols"]}
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for c in info["cols"]:
        if c["pk"]:
            out.append((c["name"], c["type"], "PK"))
            seen.add(c["name"].lower())
    for f in info["fks"]:
        if f["col"].lower() not in seen:
            out.append((f["col"], "INT", "FK"))
            seen.add(f["col"].lower())
    for name in FEATURED.get(table, []):
        if len(out) >= MAX_ROWS:
            break
        c = by_name.get(name.lower())
        if c and c["name"].lower() not in seen:
            out.append((c["name"], c["type"], ""))
            seen.add(c["name"].lower())
    return out[:MAX_ROWS]


def height(table: str, info: dict) -> int:
    return TITLE_H + DESC_H + 6 + len(rows_of(table, info)) * ROW_H + PAD


def box(x: int, y: int, table: str, info: dict, accent: str, w: int) -> str:
    rows = rows_of(table, info)
    h = height(table, info)
    remaining = len(info["cols"]) - len(rows)
    s = [
        f'<g>',
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#ffffff" '
        f'stroke="{BORDER}" stroke-width="1"/>',
        f'<path d="M{x+4} {y+0.5} h{w-8}" stroke="{accent}" stroke-width="2.5"/>',
        f'<text x="{x+PAD}" y="{y+17}" font-size="12.5" font-weight="600" '
        f'fill="{INK}">{esc(table)}</text>',
        f'<text x="{x+w-PAD}" y="{y+17}" font-size="10" text-anchor="end" '
        f'fill="{MUTED}">{fmt(info["n"])}</text>',
        f'<text x="{x+PAD}" y="{y+TITLE_H+9}" font-size="9.8" fill="{TEXT}">'
        f'{esc(DESCRIPTIONS.get(table, ""))}</text>',
        f'<path d="M{x} {y+TITLE_H+DESC_H} h{w}" stroke="{BORDER}" stroke-width="1"/>',
    ]
    for i, (name, sql_type, marker) in enumerate(rows):
        yy = y + TITLE_H + DESC_H + 15 + i * ROW_H
        if marker:
            colour = "#92400e" if marker == "PK" else "#1e40af"
            s.append(
                f'<text x="{x+PAD}" y="{yy}" font-size="7.8" font-weight="700" '
                f'letter-spacing=".4" fill="{colour}">{marker}</text>'
            )
        s.append(
            f'<text x="{x+PAD+21}" y="{yy}" font-size="10.4" '
            f'font-weight="{"600" if marker else "400"}" fill="{INK if marker else TEXT}">'
            f'{esc(name)}</text>'
        )
        s.append(
            f'<text x="{x+w-PAD}" y="{yy}" font-size="9.4" text-anchor="end" '
            f'fill="{MUTED}">{esc(short_type(sql_type))}</text>'
        )
    if remaining > 0:
        s.append(
            f'<text x="{x+w-PAD}" y="{y+h-3}" font-size="8.6" text-anchor="end" '
            f'fill="{MUTED}">+{remaining} more</text>'
        )
    s.append("</g>")
    return "\n".join(s)


def main() -> int:
    schema: dict = json.loads(SCHEMA.read_text(encoding="utf-8"))
    by_domain = {n: (a, [t for t in ts if t in schema]) for n, a, ts in DOMAINS}

    def column_height(names: list[str]) -> int:
        if names == ["__CENTRE__"]:
            return height("hospital", schema["hospital"]) + 88 + height("patient", schema["patient"])
        total = 0
        for i, d in enumerate(names):
            total += 24 + sum(height(t, schema[t]) + GAP_V for t in by_domain[d][1]) - GAP_V
            if i < len(names) - 1:
                total += GAP_DOMAIN
        return total

    BODY_H = max(column_height(c) for c in LAYOUT)
    Y0 = MARGIN + 114
    TOTAL_H = Y0 + BODY_H + MARGIN
    TOTAL_W = MARGIN * 2 + COL_W * 4 + max(BOX_W, CENTRE_W)

    positions: dict[str, tuple[int, int, int]] = {}
    body: list[str] = []

    for ic, names in enumerate(LAYOUT):
        # each column is centred in its step, so nothing ever overflows
        step_x = MARGIN + ic * COL_W
        if names == ["__CENTRE__"]:
            x = step_x + (max(BOX_W, CENTRE_W) - CENTRE_W) // 2
            hh = height("hospital", schema["hospital"])
            hp = height("patient", schema["patient"])
            y_h = Y0 + (BODY_H - (hh + 88 + hp)) // 2
            body.append(box(x, y_h, "hospital", schema["hospital"], INK, CENTRE_W))
            positions["hospital"] = (x, y_h, CENTRE_W)
            y_p = y_h + hh + 88
            body.append(box(x, y_p, "patient", schema["patient"], INK, CENTRE_W))
            positions["patient"] = (x, y_p, CENTRE_W)
            continue

        x = step_x + (max(BOX_W, CENTRE_W) - BOX_W) // 2
        y = Y0 + (BODY_H - column_height(names)) // 2
        for d in names:
            accent, tables = by_domain[d]
            body.append(
                f'<text x="{x}" y="{y+12}" font-size="10" font-weight="700" '
                f'letter-spacing="1.3" fill="{accent}">{esc(d)}</text>'
            )
            y += 24
            for t in tables:
                body.append(box(x, y, t, schema[t], accent, BOX_W))
                positions[t] = (x, y, BOX_W)
                y += height(t, schema[t]) + GAP_V
            y += GAP_DOMAIN - GAP_V

    # --- Links ---
    # Routing is deliberately simple: one elbow, one shared corridor on each side of
    # `patient`, and a single arrival point in the middle of its edge. A fanned-out
    # version — one corridor per link — was tried: with 28 links it produces a
    # tangle of curves around `patient` and reads far worse.
    # Lines are drawn before the boxes, which are opaque and hide their ends.
    px, py, pw = positions["patient"]
    ph = height("patient", schema["patient"])
    links: list[str] = []
    for _name, (accent, tables) in by_domain.items():
        for t in tables:
            if not schema[t]["fks"]:
                continue
            tx, ty, tw = positions[t]
            left = tx < px
            x1 = tx + tw if left else tx
            y1 = ty + height(t, schema[t]) / 2
            x2 = px if left else px + pw
            y2 = py + ph / 2
            pivot = x2 + (-26 if left else 26)
            links.append(
                f'<path d="M{x1} {y1:.0f} H{pivot:.0f} V{y2:.0f} H{x2:.0f}" fill="none" '
                f'stroke="{accent}" stroke-width="1" opacity=".38"/>'
            )

    hx, hy, hw = positions["hospital"]
    hh = height("hospital", schema["hospital"])
    links.append(
        f'<path d="M{hx+hw//2} {hy+hh} V{py}" fill="none" stroke="{INK}" stroke-width="1.4"/>'
        f'<rect x="{hx+hw//2-34}" y="{(hy+hh+py)//2-9}" width="68" height="18" rx="3" '
        f'fill="#ffffff"/>'
        f'<text x="{hx+hw//2}" y="{(hy+hh+py)//2+4}" font-size="9.6" text-anchor="middle" '
        f'fill="{TEXT}">hospitalid</text>'
    )

    n_tables = len(schema)
    n_rows = sum(v["n"] for v in schema.values())
    n_columns = sum(len(v["cols"]) for v in schema.values())
    n_fk = sum(len(v["fks"]) for v in schema.values())

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{TOTAL_W}" height="{TOTAL_H}" viewBox="0 0 {TOTAL_W} {TOTAL_H}" font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">
<rect width="{TOTAL_W}" height="{TOTAL_H}" fill="#ffffff"/>
<text x="{MARGIN}" y="{MARGIN+22}" font-size="22" font-weight="700" fill="{INK}">eICU-CRD v2.0.1</text>
<text x="{MARGIN}" y="{MARGIN+43}" font-size="12" fill="{TEXT}">Public release of the eICU Collaborative Research Database: {fmt(schema["patient"]["n"])} real intensive-care stays, {fmt(schema["hospital"]["n"])} US hospitals, 2014-2015.</text>
<text x="{MARGIN}" y="{MARGIN+61}" font-size="12" fill="{TEXT}">Each stay is linked to its drugs, laboratory results, vital signs, diagnoses and care.</text>
<text x="{MARGIN}" y="{MARGIN+81}" font-size="11.5" fill="{MUTED}">{n_tables} tables &#183; {n_columns} columns &#183; {fmt(n_rows)} rows &#183; {n_tables} primary keys &#183; {n_fk} foreign keys &#183; 428 MB</text>
<path d="M{MARGIN} {MARGIN+94} H{TOTAL_W-MARGIN}" stroke="{BORDER}" stroke-width="1"/>
<text x="{TOTAL_W-MARGIN}" y="{MARGIN+22}" font-size="10.5" text-anchor="end" fill="{TEXT}">PK primary key &#183; FK foreign key</text>
<text x="{TOTAL_W-MARGIN}" y="{MARGIN+40}" font-size="10.5" text-anchor="end" fill="{TEXT}">Every line: patientunitstayid to patient</text>
{chr(10).join(links)}
{chr(10).join(body)}
</svg>'''

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(svg, encoding="utf-8")
    print(f"SVG  {OUTPUT}  ({TOTAL_W}x{TOTAL_H}, {OUTPUT.stat().st_size / 1024:.0f} KB)")
    export(OUTPUT)
    return 0


def export(svg_path: Path) -> None:
    """Render the SVG to PDF, then to PNG.

    The PNG goes through the PDF (pypdfium2) rather than renderPM: the latter needs
    a Cairo backend, which is absent on Windows. pypdfium2 is a self-contained wheel.
    """
    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
    except ImportError:
        print("  (PDF/PNG skipped: pip install svglib reportlab pypdfium2)")
        return

    drawing = svg2rlg(str(svg_path))
    if drawing is None:
        print("  (conversion failed)")
        return

    pdf = svg_path.with_suffix(".pdf")
    renderPDF.drawToFile(drawing, str(pdf))
    print(f"PDF  {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")

    try:
        import pypdfium2
    except ImportError:
        print("  (PNG skipped: pip install pypdfium2)")
        return

    png = svg_path.with_suffix(".png")
    page = pypdfium2.PdfDocument(str(pdf))[0]
    page.render(scale=2).to_pil().save(png)          # 2x: sharp in print
    print(f"PNG  {png}  ({png.stat().st_size / 1024:.0f} KB, 2x)")


if __name__ == "__main__":
    raise SystemExit(main())
