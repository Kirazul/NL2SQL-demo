"""Stage 1 of the pipeline — Understand. Entirely local.

What this stage does
--------------------
It turns a natural-language question into an object holding everything the later
stages need, **while nothing that must stay secret leaves it**. Three operations:

1. **Extract** — GLiNER2 spots the business mentions (`providers/extractor.py`).
2. **Scope** — the glossary says which tables and columns the question is about
   (`resources/glossary.py`).
3. **Resolve** — the value index turns each mention into a real value
   (`db/value_index.py`).

Why this order
--------------
Scoping happens *before* resolution, not after. That is what lets us search
"aspirin" in the three drug columns rather than in all 106 indexed ones: faster,
and above all more accurate — without scoping, "female" returns
`diagnosis.diagnosisstring = 'oncology|…|breast CA|female'` before
`patient.gender = 'Female'`.

The trust boundary
------------------
Nothing here touches the network. On output, `Understanding` holds two very
different kinds of thing:

- **real values** (`Resolution.value`) — which will never cross the boundary;
  stage 2 replaces them with `:v1`, `:v2` symbols;
- **vocabulary and column names** — which do go into the prompt.

That separation is carried by the type: `for_the_cloud()` returns only the second
kind. Everything heading to the cloud goes through that method.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from hybridsql.db.value_index import FoundValue, search
from hybridsql.providers import extractor
from hybridsql.resources import glossary

# Below this, we do not guess: we ask the analyst. A badly resolved value produces
# SQL that runs without error and returns a wrong answer — the worst case, far
# worse than an outright failure.
CONFIDENCE_THRESHOLD = 0.75

# When the scoped search returns nothing, we widen to every column. That widening
# rescues an incomplete glossary, but it also resolves anything anywhere:
# "hemoglobin", absent from `lab.labname` in the demo database, landed in
# `diagnosis.diagnosisstring` with a score of 1.00 — above the threshold, so
# nobody flagged it.
#
# The penalty pushes such resolutions below the confidence threshold. They are
# still offered, but as suggestions for the analyst to confirm, which is their
# real status. A value found outside its expected domain is a hint, not an answer.
OUT_OF_SCOPE_PENALTY = 0.7

# Extractor labels to glossary terms. This bridge gives each entity type its
# search scope.
TYPE_TO_TERMS: dict[str, tuple[str, ...]] = {
    "drug": ("drug",),
    "diagnosis": ("diagnosis", "medical_history"),
    "lab_test": ("lab_test",),
    "organism": ("microbiology",),
    "procedure": ("procedure", "ventilation"),
    "facility": ("hospital", "care_unit", "teaching_status"),
    "demographic": ("sex", "ethnicity", "age"),
    "region": ("region",),
}

Kind = Literal["value", "concept", "quantity", "person"]


@dataclass(frozen=True)
class Resolution:
    """A mention from the question, tied to a real value of the database."""

    mention: str                    # what the analyst wrote
    type: str                       # extractor label
    value: str | None               # the exact stored value — NEVER LEAVES
    column: str | None              # `table.column` — may leave
    score: float
    tier: str = ""                  # A or B: which path the index answered on
    alternatives: tuple[str, ...] = ()
    kind: Kind = "value"
    out_of_scope: bool = False      # found outside the columns expected for its type

    @property
    def resolved(self) -> bool:
        return self.value is not None

    @property
    def confident(self) -> bool:
        return self.resolved and self.score >= CONFIDENCE_THRESHOLD

    @property
    def to_mask(self) -> bool:
        """Values and numbers are replaced by a symbol. Concepts are not.

        A concept ("mortality rate") names a column: masking it would deprive the
        cloud model of what it needs to write the SQL, and would protect nothing,
        since the column name goes into the DDL anyway.

        Numbers are masked. They were not, once, on the argument that "over 65" is
        a threshold the analyst chose rather than data — but the same rule left
        `WHERE hospitalid = 56` going out in clear, and 56 identifies one specific
        hospital. The model does not need a number's value to write a comparison,
        so masking all of them costs nothing and closes the case where a number is
        an identifier. See `_numbers`.
        """
        if self.kind == "quantity":
            # Only the deterministic numeric pass, never the classifier's own
            # "quantity" bucket. That bucket means "no significant word left after
            # removing function words", which catches `over 65` — and also catches
            # the verb `admitted`, since it is a function word. Masking those
            # produced `:v1` bound to an empty string, a query comparing a column
            # to nothing, and a confident answer of zero.
            return self.type == "number"
        return self.kind == "value" and self.confident


@dataclass
class Understanding:
    """Stage 1 result. Contains secrets: do not serialise as-is."""

    question: str
    entities: list[extractor.Entity] = field(default_factory=list)
    resolutions: list[Resolution] = field(default_factory=list)
    tables: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    ms: float = 0.0
    active_extractor: str = ""

    @property
    def needs_confirmation(self) -> list[Resolution]:
        """Mentions too uncertain to be used without asking."""
        return [r for r in self.resolutions if r.resolved and not r.confident]

    @property
    def unresolved(self) -> list[Resolution]:
        """Values no column recognises. The question may be out of scope.

        Concepts and quantities are excluded: their lack of resolution is normal,
        not a failure.
        """
        return [r for r in self.resolutions if r.kind == "value" and not r.resolved]

    @property
    def values(self) -> list[Resolution]:
        """Mentions that will actually be replaced by a symbol."""
        return [r for r in self.resolutions if r.to_mask]

    @property
    def persons(self) -> list[Resolution]:
        """Person names spotted in the question.

        Their presence is a strong signal: the database is de-identified, so the
        question cannot be satisfied as written, and the name must certainly not
        go to the cloud. The API must say so instead of attempting an answer.
        """
        return [r for r in self.resolutions if r.kind == "person"]

    def for_the_cloud(self) -> dict[str, object]:
        """The only view allowed to cross the trust boundary.

        Holds neither `value` nor `mention`: only column names, types and business
        notes. It is the counterpart of `Resolution`, kept deliberately separate so
        that a forgotten field is visible right here.
        """
        return {
            "tables": sorted(self.tables),
            "target_columns": sorted({r.column for r in self.resolutions if r.column}),
            "notes": list(self.notes),
            "masked_value_count": len(self.values),
        }


TITLE_RE = re.compile(r"^(mr|mrs|ms|miss|dr|prof|sir|m|mme|mlle)\b\.?", re.IGNORECASE)


def _looks_like_a_name(mention: str) -> bool:
    """Does this mention have the shape of a proper noun?

    The extractor's `person` label is generous, and refusing a request is the
    heaviest thing this pipeline does — so the label alone is not enough evidence.
    Measured on casual phrasing, GLiNER2 returned `person` for: `me`, `i`, `peeps`,
    and `mny patiens`. Every one of those questions was refused with "the question
    names one or more people", which is both wrong and impossible for the analyst
    to act on.

    The test is syntactic and needs no list of names: a proper noun carries a
    title, or an initial capital on a word that is not ordinary vocabulary.
    `Mr. Bensalah` passes on both counts; `me` and `peeps` on neither.

    A name typed in all lower case slips through this — and then fails as an
    unresolvable value instead, because eICU is de-identified and holds no names.
    The request still stops; only the wording of the refusal is less precise.
    """
    from hybridsql.security.egress_gate import generic_vocabulary

    text = mention.strip()
    if not text:
        return False
    if TITLE_RE.match(text):
        return True

    generic = generic_vocabulary()
    return any(
        word[:1].isupper() and word.lower() not in generic
        for word in re.findall(r"[A-Za-zÀ-ÿ']+", text)
    )


def _classify(mention: str, entity_type: str = "") -> Kind:
    """Tell a value to look up from a concept or a quantity.

    Why this step exists
    --------------------
    Without it, the index resolves whatever it is given, and it always finds
    something. Measured on the trial questions:

        "mortality rate"  ->  careplangeneral.cplitemvalue = 'Low mortality risk'  (0.79)
        "over 65"         ->  admissiondx.admitdxtext      = '65'                  (1.00)
        "infection"       ->  admissiondx.admitdxname      = 'Renal infection...'  (1.00)

    All three are wrong, and two clear the confidence threshold — so nobody flags
    them. That is the project's worst failure mode: SQL that runs without error
    and returns a wrong answer.

    The rules
    ---------
    Classification runs on **significant words**: three letters or more, not part
    of the generic vocabulary (function words, SQL keywords). Without that
    reduction two cases still escaped:

    - "over 65" counted "over" as a real word and went to resolution;
    - "mortality rate" was not fully covered by the glossary, because of "rate".

    Once reduced:

    - **quantity** — no significant word left ("over 65", "12.5 %"). The number
      comes from the analyst, not from the database.
    - **concept** — the glossary covers every significant word. The mention names
      a column, not content: "length of stay", "mortality rate".
    - **value** — everything else, the only case we look up in the database.
    """
    from hybridsql.security.egress_gate import generic_vocabulary

    # A person name is never looked up. eICU is de-identified: the search would
    # return nothing useful, but fuzzy matching could tie the name to an arbitrary
    # value and produce an absurd resolution. The name is reported to the analyst,
    # and the egress gate stops it since it is not in the allowlist.
    if entity_type == "person":
        return "person" if _looks_like_a_name(mention) else "concept"

    generic = generic_vocabulary()
    words = glossary.WORD_RE.findall(glossary._normalize(mention))
    significant = [w for w in words if len(w) >= 3 and w.isalpha() and w not in generic]
    if not significant:
        return "quantity"

    # An exact match against a database value wins over the glossary, but **only
    # for multi-word mentions**.
    #
    # The intent: "Other Hospital" and "Neuro ICU" are covered by the synonyms
    # "hospital" and "icu", so they were classified as concepts — while being
    # stored verbatim in `unitadmitsource` and `unittype`.
    #
    # The restriction: applied to every word, the rule backfired. "icu", "culture",
    # "drug", "intubated" also exist as whole values somewhere, and everything
    # tipped to "value" — classification accuracy fell from 88% to 29% on the hard
    # set. A single word found in the glossary names the concept; it is the
    # two-word assembly that makes the specific value.
    from hybridsql.db.value_index import is_exact_value

    if len(words) >= 2 and is_exact_value(mention):
        return "value"

    covered = {w for m in glossary.recognize(mention) for w in m.trigger.split()}
    if all(w in covered for w in significant):
        return "concept"
    return "value"


def _concept_column(mention: str) -> str | None:
    """The main column a concept designates, for prompt scoping."""
    columns = glossary.columns_for(mention)
    return columns[0] if columns else None


def _scope(entity: extractor.Entity, default: list[str]) -> list[str]:
    """Columns to search this mention in, most likely first.

    The two sources are **combined**, not put in competition:

    - the **entity type** returned by GLiNER2, which is usually right;
    - the columns the **glossary derives from the whole question**.

    The first version kept only the type when it yielded anything. But GLiNER2
    labels "discharged home" as a *procedure*: the search went to
    `treatment.treatmentstring`, found nothing there, and fell back to an
    unscoped search that landed in `careplangeneral`. The discharge columns
    (`patient.hospitaldischargelocation`), although derived from "discharged" by
    the glossary, were never consulted.

    Combining keeps the type's ranking priority without letting it hide the rest.
    """
    terms = glossary.load()
    columns: list[str] = []
    for name in TYPE_TO_TERMS.get(entity.type, ()):
        term = terms.get(name)
        if term:
            columns.extend(c for c in term.columns if c not in columns)
    columns.extend(c for c in default if c not in columns)
    return columns


MAX_SPAN_RETRIES = 6


def _shorter_spans(mention: str) -> list[str]:
    """Sub-spans of a mention, longest first, for when the whole thing resolves to nothing.

    An entity extractor marks *where it thinks the interesting thing is*, and it
    routinely takes a word too many: "hemoglobin lab test" where only `hemoglobin`
    is stored, "aspirin count" where only `aspirin` is. The whole span then matches
    nothing, the value is reported as unknown, and a perfectly ordinary question is
    refused.

    Rather than trusting the span or trimming it with a list of stop words, the
    candidates are enumerated and the index decides. Longest first, so the most
    specific match still wins — `Skilled Nursing Facility` must not be reduced to
    `Nursing` when the full phrase exists.

    Spans made entirely of function words are dropped: they cannot be a value, and
    querying them wastes the retries on noise.
    """
    from hybridsql.security.egress_gate import generic_vocabulary

    words = mention.split()
    if len(words) < 2:
        return []

    generic = generic_vocabulary()
    spans: list[str] = []
    for size in range(len(words) - 1, 0, -1):
        for start in range(len(words) - size + 1):
            span = " ".join(words[start : start + size])
            if span in spans:
                continue
            if all(w.lower().strip(".,;:()") in generic for w in span.split()):
                continue
            spans.append(span)
            if len(spans) >= MAX_SPAN_RETRIES:
                return spans
    return spans


def _resolve(entity: extractor.Entity, scope: list[str]) -> Resolution:
    """Turn a mention into a real value, widening the search if needed.

    Two attempts: first within the scoped columns, then unconstrained. The
    widening rescues an incomplete glossary — but it comes second, so that scoping
    keeps priority when it works.
    """
    kind = _classify(entity.text, entity.type)
    if kind != "value":
        # We do not look up in the database what is not in it. The column is still
        # filled in: it serves to scope the schema sent to the cloud.
        return Resolution(
            mention=entity.text,
            type=entity.type,
            value=None,
            column=_concept_column(entity.text) if kind == "concept" else None,
            score=0.0,
            kind=kind,
        )

    found: list[FoundValue] = []
    out_of_scope = False
    mention = entity.text
    if scope:
        found = search(mention, columns=scope, limit=4)
    if not found:
        found = search(mention, limit=4)
        out_of_scope = bool(scope)

    # The extractor's span is a hypothesis, not a fact. It routinely returns more
    # than the value — "hemoglobin lab test", "aspirin count" — and the index,
    # asked for the whole thing, does not fail cleanly: it returns the closest
    # thing it has. Measured: `hemoglobin lab test` -> `testes` at 0.56,
    # `aspirin count` -> `reticulocyte count` at 0.63. Both are nonsense, and both
    # look like answers.
    #
    # So the retry is triggered by a *weak* match, not only by no match, and the
    # best result across all the spans wins. Asking the index is cheap; believing
    # a 0.56 is not.
    if not found or found[0].score < CONFIDENCE_THRESHOLD:
        best_so_far = found[0].score if found else 0.0
        for shorter in _shorter_spans(entity.text):
            candidates = search(shorter, columns=scope, limit=4) if scope else []
            widened = False
            if not candidates:
                candidates = search(shorter, limit=4)
                widened = bool(scope)
            if candidates and candidates[0].score > best_so_far:
                found, mention, best_so_far = candidates, shorter, candidates[0].score
                out_of_scope = widened
                if best_so_far >= 0.99:
                    break

    if not found:
        return Resolution(entity.text, entity.type, None, None, 0.0)

    best, *others = found
    # The out-of-scope penalty exists because a *fuzzy* hit in an unexpected
    # column is usually wrong. An exact string match is not: finding `aspirin`
    # spelled exactly that way in `allergy.allergyname` rather than in the drug
    # column the entity type predicted says the type guess was off, not that the
    # value is wrong. Penalising it dropped a perfect match to 0.70 and refused
    # the question.
    penalised = out_of_scope and best.score < 1.0
    score = best.score * (OUT_OF_SCOPE_PENALTY if penalised else 1.0)

    return Resolution(
        mention=entity.text,
        type=entity.type,
        value=best.value,
        column=best.ref,
        score=round(score, 4),
        tier=best.tier,
        alternatives=tuple(f"{o.ref} = {o.value}" for o in others),
        out_of_scope=penalised,
    )


NUMBER_IN_QUESTION = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w.])")


def _numbers(question: str, already_masked: list[tuple[int, int]]) -> list[Resolution]:
    """Every number in the question, so that none of them leaves in clear text.

    Why this is a separate pass, and not the extractor's job
    -------------------------------------------------------
    GLiNER2 is asked for *semantic* types — a drug, a diagnosis, a lab test. A
    bare identifier belongs to none of them, so "how many patients in hospital id
    56" produced no entity at all: nothing was found, nothing was masked, and `56`
    went to the provider inside `WHERE hospitalid = 56`. Depending on a model to
    notice a number is the wrong mechanism; a number is a syntactic class and can
    be found exactly, every time, with no model and no vocabulary.

    Why numbers are masked at all
    -----------------------------
    They used to be classified as `quantity` and deliberately left alone, on the
    argument that "over 65" is a threshold the analyst chose rather than data read
    from the database. That argument is right about `65` and wrong about `56`:
    a hospital id identifies one specific row of the database, and disclosing it
    tells the provider which hospital is being investigated.

    Telling the two apart reliably is not possible from the question alone — and
    it does not need to be, because the model never needs a number's value.
    `WHERE hospitalid = :v1` and `WHERE age > :v1` are both perfectly writable
    from the column and the operator alone. So all of them are masked: the strictly
    safer option costs nothing.

    A number is skipped only when the entity containing it is **itself going to be
    masked** — "AMOXICILLIN 500 MG" is replaced once, as a whole, and must not be
    replaced twice. Skipping every number that merely sits inside some entity was
    the first attempt and it leaked twice over: GLiNER2 returns "hospital id 56" as
    one `facility`, which the classifier calls a concept and leaves in clear, and
    "over 65" as one `demographic`, likewise. Both numbers were covered by an
    entity that protected nothing.
    """

    def inside_masked(start: int, end: int) -> bool:
        return any(s <= start and end <= e for s, e in already_masked)

    out: list[Resolution] = []
    for match in NUMBER_IN_QUESTION.finditer(question):
        if inside_masked(match.start(), match.end()):
            continue
        out.append(
            Resolution(
                mention=match.group(),
                type="number",
                # A quantity resolves to itself: there is nothing to look up, the
                # analyst supplied the value. Full confidence, so it is masked
                # rather than queued for confirmation.
                value=match.group(),
                column=None,
                score=1.0,
                kind="quantity",
            )
        )
    return out


def _missed_values(
    question: str, existing: list[Resolution], question_columns: list[str]
) -> list[Resolution]:
    """Stored values the extractor did not spot — masked here rather than blocked later.

    The two layers were contradicting each other. GLiNER2 looks for *semantic*
    entities and missed `alive` in "how many female patients were discharged
    alive"; the egress gate, which knows every stored value exactly, then refused
    the question because `Alive` is the literal content of
    `patient.hospitaldischargestatus`. One layer said "nothing to mask", the other
    said "this must not leave", and the analyst got a refusal for an ordinary
    question.

    Both are right about the facts and the disagreement is about whose job it is.
    It belongs here: the gate is a backstop that fails a request, while masking
    fixes it. So the same exhaustive inventory the gate uses to *detect* is used
    here to *mask*, before anything is sent.

    The candidates come from `find_known_values`, which is the gate's own detector
    and already subtracts everything that cannot be data: closed-class grammar,
    schema identifiers, and the concepts the glossary declares. What survives that
    subtraction is, by construction, a stored value with no other explanation —
    so the rule is simply **whatever the gate would refuse, mask instead**.

    Exact matches only. This pass runs with no extractor evidence that the word is
    a mention at all, so it is allowed to recognise but never to guess.
    """
    from hybridsql.security.egress_gate import find_known_values

    already = " ".join(r.mention.lower() for r in existing if r.to_mask)
    scope = question_columns or None
    out: list[Resolution] = []

    for candidate in find_known_values(question):
        if candidate in already or any(candidate == r.mention.lower() for r in existing):
            continue
        # Prefer a column the question is already about; fall back to anywhere,
        # since an exact match on a complete stored value is strong on its own.
        found = search(candidate, columns=scope, limit=2) if scope else []
        if not found or found[0].score < 1.0:
            found = search(candidate, limit=2)
        if not found or found[0].score < 1.0:
            continue
        best = found[0]
        out.append(
            Resolution(
                mention=candidate,
                type="stored-value",
                value=best.value,
                column=best.ref,
                score=best.score,
                tier=best.tier,
                kind="value",
            )
        )
    return out


def understand(question: str) -> Understanding:
    """Run stage 1 end to end. No network call."""
    start = time.perf_counter()

    entities = extractor.extract(question)
    question_columns = glossary.columns_for(question)
    tables = glossary.tables_for(question)
    notes = glossary.notes_for(question)

    resolutions = [_resolve(e, _scope(e, question_columns)) for e in entities]
    masked_spans = [
        (e.start, e.end) for e, r in zip(entities, resolutions) if r.to_mask
    ]
    resolutions += _numbers(question, masked_spans)
    resolutions += _missed_values(question, resolutions, question_columns)

    # Tables from resolved values count as much as those from the glossary: a
    # question saying only "aspirin" triggers no business term, but the resolution
    # does designate `medication`.
    for r in resolutions:
        if r.column:
            tables.add(r.column.split(".", 1)[0])

    return Understanding(
        question=question,
        entities=entities,
        resolutions=resolutions,
        tables=tables,
        notes=notes,
        ms=round((time.perf_counter() - start) * 1000, 1),
        active_extractor="gliner2" if extractor.available() else "glossary",
    )
