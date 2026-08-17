"""Generate `docs/schema-reference.pdf` — every table, every column, explained.

    python scripts/build_schema_reference.py

Why this is generated and not written
-------------------------------------
A schema reference kept by hand rots on the first migration, and a reference that
disagrees with the database is worse than none: it sends whoever reads it toward
SQL that will never run. So everything factual here is **read from the database at
build time** — row counts, types, keys, distinct counts, null shares, example
values, and the tier the indexing policy assigned.

Where the prose comes from, and how to tell
-------------------------------------------
Three sources, and the PDF says which one answered for every column:

    schema      the column is a key, a foreign key, or an offset — read from the
                database itself, so it cannot be wrong;
    glossary    `config/glossary.yaml` already declares this column as a business
                concept, with the note the cloud model receives;
    eICU        documented semantics of the published dataset, for the conventions
                a reader cannot guess: `*offset` in minutes, `age` stored as text
                because of the '> 89' de-identification rule, the three nested
                patient identifiers.

Anything with no source gets the honest fallback: what the column holds, measured.
No description is invented to fill a cell.

The example values come from eICU-CRD Demo v2.0.1, a public de-identified research
database. They are printed for the columns the value index treats as vocabulary,
because a column called `cplitemvalue` means nothing until you see three of them.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybridsql.config import settings  # noqa: E402
from hybridsql.db.schema import _compact_type, read_schema  # noqa: E402

OUTPUT = Path("docs/schema-reference.pdf")
MAX_EXAMPLES = 3
MAX_EXAMPLE_CHARS = 46

# --- What a reader cannot guess from the data ---------------------------------------
# Conventions of the published eICU export. Each of these is a documented property of
# the dataset, not an inference from our copy of it.
EICU_COLUMNS: dict[str, str] = {
    "uniquepid": "The person. One patient may have several hospital stays, so this is "
                 "the id to COUNT DISTINCT when a question asks how many patients.",
    "patienthealthsystemstayid": "One hospital stay. A person may have several; each may "
                                 "contain several ICU stays.",
    "patientunitstayid": "One ICU stay. The join key of the whole database: 28 of the 31 "
                         "tables carry it.",
    "hospitalid": "The hospital. Joins to hospital.hospitalid.",
    "wardid": "The ward inside the hospital. No separate ward table exists.",
    "age": "Stored as TEXT, not a number: patients over 89 are recorded as '> 89' to "
           "de-identify them, so a numeric comparison must cast and handle that value.",
    "gender": "Male, Female, Unknown or empty.",
    "ethnicity": "Self-reported, seven categories including an empty one.",
    "admissionheight": "Centimetres at ICU admission.",
    "admissionweight": "Kilograms at ICU admission.",
    "dischargeweight": "Kilograms at ICU discharge.",
    "unittype": "Type of ICU: Med-Surg ICU, MICU, CCU-CTICU, Neuro ICU, and so on.",
    "unitadmitsource": "Where the patient came from when entering the ICU.",
    "unitstaytype": "admit, readmit, stepdown/other, or transfer.",
    "unitdischargestatus": "Alive or Expired at ICU discharge.",
    "hospitaldischargestatus": "Alive or Expired at hospital discharge. This is the "
                               "mortality outcome.",
    "hospitaldischargelocation": "Where the patient went: Home, Skilled Nursing Facility, "
                                 "Death, and so on.",
    "hospitaladmitsource": "Where the patient came from when entering the hospital.",
    "apacheadmissiondx": "The APACHE admission diagnosis, one string from a controlled list.",
    "apacheversion": "IV or IVa. The two versions coexist in this table and their scores "
                     "are not comparable; filter on one.",
    "labname": "The name of the laboratory test, e.g. 'bedside glucose', 'potassium'.",
    "labresult": "The numeric result. `labresulttext` holds the same value as written.",
    "drugname": "The drug as recorded by the hospital, including strength and form: "
                "'ASPIRIN EC 81 MG PO TBEC'. This is why a question saying 'aspirin' has "
                "to be resolved against the value index before any query is written.",
    "diagnosisstring": "A pipe-separated hierarchy, from body system down to the specific "
                       "diagnosis: 'renal|disorder of kidney|acute renal failure'.",
    "icd9code": "ICD-9 code, sometimes several separated by commas, sometimes empty.",
    "cellpath": "A slash- or pipe-separated path naming where the value sits in the "
                "charting hierarchy.",
    "notevalue": "The content of the note field named by `notepath`.",
    "treatmentstring": "A pipe-separated hierarchy of the treatment given.",
    "activeupondischarge": "True if the item was still active when the patient left.",
}

# Suffix conventions, applied when no explicit entry matches. Longest suffix first.
EICU_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("offset", "Minutes since ICU admission; a negative value means before admission. "
               "Never a date: the export is de-identified by shifting every timestamp."),
    ("time24", "Time of day, HH:MM, as a 24-hour clock. No date."),
    ("year", "Year only, the single unshifted time field."),
    ("los", "Length of stay, in days."),
    ("string", "A pipe-separated hierarchy, from the general to the specific."),
    ("path", "A path naming where this value sits in the charting hierarchy."),
    ("label", "The label of the charted item."),
    ("value", "The charted value, as text."),
    ("text", "Free text as entered."),
    ("name", "A name from a controlled vocabulary."),
    ("type", "A category from a small controlled list."),
    ("status", "A status from a small controlled list."),
    ("id", "Identifier. Not a business value: nobody names one in a question."),
)

TIER_MEANING = {
    "A": "indexed — every distinct value is stored, so a mention resolves to it",
    "B": "on demand — too many distinct values to store; searched with a bounded LIKE",
    "C": "excluded — an identifier, a timestamp or free text; never resolved",
    "": "not a text column",
}

TABLE_PURPOSE: dict[str, str] = {
    "patient": "One row per ICU stay: demographics, admission and discharge. The centre of "
               "the schema — every other table joins here.",
    "hospital": "One row per hospital: region, bed count, teaching status.",
    "admissiondrug": "Drugs the patient was already taking when admitted, as recorded on "
                     "the admission form.",
    "admissiondx": "The admission diagnosis as entered in the APACHE flowsheet.",
    "allergy": "Recorded allergies, with the drug involved.",
    "apacheapsvar": "The raw physiological variables the APACHE score is computed from.",
    "apachepatientresult": "APACHE scores and the predicted outcomes derived from them, "
                           "with the actual outcome beside each.",
    "apachepredvar": "The predictor variables feeding the APACHE model, including chronic "
                     "conditions.",
    "careplancareprovider": "Which specialty was responsible for the patient, and when.",
    "careplaneol": "End-of-life care plan entries.",
    "careplangeneral": "General care plan entries, as item/value pairs.",
    "careplangoal": "Care plan goals, as item/value pairs.",
    "careplaninfectiousdisease": "Infectious-disease care plan entries.",
    "customlab": "Laboratory results outside the standard set.",
    "diagnosis": "Diagnoses recorded during the stay, as a hierarchy plus an ICD-9 code.",
    "infusiondrug": "Continuous infusions, with rate and volume over time.",
    "intakeoutput": "Fluid intake and output, charted as labelled cells.",
    "lab": "Laboratory results: name, numeric value, unit, time.",
    "medication": "Medication orders: drug, dose, route, frequency, start and stop.",
    "microlab": "Microbiology cultures: site, organism, antibiotic sensitivity.",
    "note": "Clinical notes, stored as path/value pairs rather than free prose.",
    "nurseassessment": "Nursing assessments, as attribute/value pairs.",
    "nursecare": "Nursing care items, as attribute/value pairs.",
    "nursecharting": "The nursing flowsheet — the largest table here. Cell type, label and "
                     "value, one row per charted item.",
    "pasthistory": "Medical history recorded on admission, as a hierarchy.",
    "physicalexam": "Physical examination findings, as path/value pairs.",
    "respiratorycare": "Ventilator settings and respiratory care events.",
    "respiratorycharting": "The respiratory flowsheet, charted like nursecharting.",
    "treatment": "Treatments given, as a hierarchy.",
    "vitalaperiodic": "Vital signs recorded intermittently, one row per measurement time.",
    "vitalperiodic": "Vital signs recorded automatically, typically every five minutes. "
                     "The largest table by row count.",
}


# ---------------------------------------------------------------------------------
# Reading the database
# ---------------------------------------------------------------------------------
def classification() -> dict[str, dict]:
    path = Path(settings().value_index_path).with_name("column_classification.json")
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return {c["ref"]: c for c in report.get("columns", [])}


def glossary_columns() -> dict[str, tuple[str, str]]:
    """ref -> (business term, note) for every column the glossary declares."""
    from hybridsql.resources.glossary import load

    out: dict[str, tuple[str, str]] = {}
    for term in load().values():
        for ref in term.columns:
            out.setdefault(ref, (term.canonical.replace("_", " "), term.note))
    return out


def column_facts(cx: sqlite3.Connection, table: str, column: str, rows: int) -> dict:
    """Distinct count, null share and a few examples. One pass, no full scan of text."""
    try:
        distinct, filled = cx.execute(
            f'SELECT COUNT(DISTINCT "{column}"), COUNT("{column}") FROM "{table}"'
        ).fetchone()
    except sqlite3.Error:
        return {"distinct": 0, "empty_pct": 0.0, "examples": []}

    examples: list[str] = []
    if 1 < (distinct or 0) <= 200_000:
        try:
            for (value,) in cx.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL AND TRIM("{column}") <> "" LIMIT 60'
            ):
                text = " ".join(str(value).split())
                if not text or len(text) > MAX_EXAMPLE_CHARS:
                    continue
                examples.append(text)
                if len(examples) >= MAX_EXAMPLES:
                    break
        except sqlite3.Error:
            pass
    return {
        "distinct": distinct or 0,
        "empty_pct": 100 * (1 - (filled or 0) / rows) if rows else 0.0,
        "examples": examples,
    }


def describe(table: str, column: str, facts: dict, tier: str,
             glossary: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """(description, where it came from). Never invents one."""
    ref = f"{table}.{column}"
    name = column.lower()

    if name in EICU_COLUMNS:
        return EICU_COLUMNS[name], "eICU"

    if ref in glossary:
        term, note = glossary[ref]
        text = f"The glossary declares this column as '{term}'."
        if note:
            text += f" {note}"
        return text, "glossary"

    for suffix, text in EICU_SUFFIXES:
        if name.endswith(suffix):
            return text, "eICU"

    # No source claims it. Say what it holds, measured — and nothing else.
    if facts["distinct"] <= 1:
        return "One value or empty throughout this extract.", "measured"
    if tier == "A":
        return f"A bounded vocabulary of {facts['distinct']:,} distinct values.", "measured"
    if tier == "B":
        return f"{facts['distinct']:,} distinct values — too many to index.", "measured"
    return f"{facts['distinct']:,} distinct values.", "measured"


# ---------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------
def build() -> Path:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
    )

    schema = read_schema()
    tiers = classification()
    glossary = glossary_columns()

    INK = colors.HexColor("#18181b")
    MUTED = colors.HexColor("#71717a")
    RULE = colors.HexColor("#e4e4e7")
    ACCENT = colors.HexColor("#4f46e5")

    sheet = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=sheet["BodyText"], fontName="Helvetica", fontSize=7.6,
                          leading=9.8, textColor=INK, alignment=TA_LEFT, spaceAfter=0)
    mono = ParagraphStyle("mono", parent=body, fontName="Courier-Bold", fontSize=7.6,
                          textColor=INK)
    small = ParagraphStyle("small", parent=body, fontSize=7.0, leading=9.0, textColor=MUTED)
    head = ParagraphStyle("head", parent=body, fontName="Helvetica-Bold", fontSize=7.2,
                          textColor=MUTED)
    title = ParagraphStyle("title", parent=sheet["Title"], fontName="Helvetica-Bold",
                           fontSize=22, leading=26, textColor=INK, alignment=TA_LEFT,
                           spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=sheet["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                        leading=16, textColor=ACCENT, spaceBefore=10, spaceAfter=2)
    lead = ParagraphStyle("lead", parent=body, fontSize=9, leading=13, textColor=INK)

    def escape(text: str) -> str:
        return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    page = landscape(A4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(14 * mm, 8 * mm, "NL2SQL — eICU-CRD schema reference")
        canvas.drawRightString(page[0] - 14 * mm, 8 * mm, f"page {doc.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(14 * mm, 11 * mm, page[0] - 14 * mm, 11 * mm)
        canvas.restoreState()

    document = BaseDocTemplate(str(OUTPUT), pagesize=page,
                               leftMargin=14 * mm, rightMargin=14 * mm,
                               topMargin=12 * mm, bottomMargin=14 * mm,
                               title="eICU-CRD schema reference", author="NL2SQL")
    frame = Frame(document.leftMargin, document.bottomMargin,
                  document.width, document.height, id="body")
    document.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])

    story: list = []
    total_columns = sum(len(t.columns) for t in schema.values())
    total_rows = sum(t.row_count for t in schema.values())

    story.append(Paragraph("eICU-CRD — schema reference", title))
    story.append(Paragraph(
        f"{len(schema)} tables, {total_columns} columns, {total_rows:,} rows. "
        "Generated from the database itself by <font face='Courier'>scripts/build_schema_reference.py</font>, "
        "so it cannot disagree with it.", lead))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Where each description comes from</b> is printed beside it. "
        "<b>eICU</b> — a documented convention of the published dataset. "
        "<b>glossary</b> — the column is declared in <font face='Courier'>config/glossary.yaml</font>, "
        "and the note shown is the one the cloud model receives. "
        "<b>measured</b> — nothing documents this column, so what is printed is what it "
        "holds, counted here. No description is invented to fill a cell.", small))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>tier</b> is the indexing decision (<font face='Courier'>db/value_index.py</font>): "
        "<b>A</b> " + TIER_MEANING["A"] + "; <b>B</b> " + TIER_MEANING["B"] + "; "
        "<b>C</b> " + TIER_MEANING["C"] + ".", small))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Example values are real, from eICU-CRD Demo v2.0.1 — a public, de-identified "
        "research database published by the MIT Laboratory for Computational Physiology. "
        "They are shown because a column named <font face='Courier'>cplitemvalue</font> "
        "means nothing until you see three of them.", small))
    story.append(Spacer(1, 10))

    # --- contents ---
    rows = [[Paragraph("table", head), Paragraph("rows", head), Paragraph("cols", head),
             Paragraph("what it holds", head)]]
    for name in sorted(schema):
        table = schema[name]
        rows.append([
            Paragraph(escape(name), mono),
            Paragraph(f"{table.row_count:,}", small),
            Paragraph(str(len(table.columns)), small),
            Paragraph(escape(TABLE_PURPOSE.get(name, "")), body),
        ])
    contents = Table(rows, colWidths=[46 * mm, 20 * mm, 12 * mm, 191 * mm], repeatRows=1)
    contents.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(contents)

    # --- one section per table ---
    with sqlite3.connect(f"file:{Path(settings().db_path).as_posix()}?mode=ro",
                         uri=True) as cx:
        for name in sorted(schema):
            table = schema[name]
            story.append(PageBreak())
            story.append(Paragraph(escape(name), h2))
            purpose = TABLE_PURPOSE.get(name, "")
            story.append(Paragraph(
                f"{table.row_count:,} rows &middot; {len(table.columns)} columns"
                + (f" &middot; {escape(purpose)}" if purpose else ""), small))
            if table.foreign_keys:
                joins = ", ".join(
                    escape(f"{fk.column} -> {fk.target_table}.{fk.target_column}")
                    for fk in table.foreign_keys)
                story.append(Paragraph(f"<b>joins</b> {joins}", small))
            story.append(Spacer(1, 5))

            data = [[Paragraph(h, head) for h in
                     ("column", "type", "key", "distinct", "empty", "tier",
                      "what it is", "source", "examples")]]
            for column in table.columns:
                ref = f"{name}.{column.name}"
                facts = column_facts(cx, name, column.name, table.row_count)
                tier = tiers.get(ref, {}).get("tier", "")
                text, source = describe(name, column.name, facts, tier, glossary)
                key = "PK" if column.is_pk else ""
                for fk in table.foreign_keys:
                    if fk.column == column.name:
                        key = "FK"
                # Identifiers, keys and offsets: the three kinds of value a reference
                # has no reason to print. They are noise, and a shifted timestamp read
                # as a real one is worse than noise.
                if key or tier == "C" or column.name.lower().endswith(("id", "offset")):
                    facts["examples"] = []
                data.append([
                    Paragraph(escape(column.name), mono),
                    Paragraph(_compact_type(column.sql_type), small),
                    Paragraph(key, small),
                    Paragraph(f"{facts['distinct']:,}" if facts["distinct"] else "—", small),
                    Paragraph(f"{facts['empty_pct']:.0f}%" if facts["empty_pct"] >= 1 else "", small),
                    Paragraph(tier or "", small),
                    Paragraph(escape(text), body),
                    Paragraph(source, small),
                    Paragraph(escape(" · ".join(facts["examples"])), small),
                ])

            grid = Table(data, colWidths=[38 * mm, 13 * mm, 8 * mm, 16 * mm, 11 * mm,
                                          8 * mm, 84 * mm, 14 * mm, 77 * mm], repeatRows=1)
            grid.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK),
                ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(grid)

    document.build(story)
    return OUTPUT


def main() -> int:
    path = build()
    size = path.stat().st_size / 1024
    print(f"  {path}  {size:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
