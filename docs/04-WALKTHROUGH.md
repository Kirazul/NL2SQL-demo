# One question, stage by stage

A real execution trace, not a reconstruction.

```bash
python scripts/trace_question.py "How many patients received aspirin for sepsis?" --write
```

Two sensitive things hide in that question: the **drug** and the **condition**.
Neither may reach the provider.

---

## Stage 1 — Understand *(local, no network)*

**GLiNER2 reads the sentence.** It is given the types to look for in plain
English — `"drug or medication name"`, `"medical condition or diagnosis"`, … — so
changing domain means changing that list, not retraining.

| Mention | Type | Confidence |
|---|---|---|
| `aspirin` | drug | 1.00 |
| `sepsis` | diagnosis | 1.00 |

"patients" and "how many" are not entities — nothing to mask there.

**Each mention is classified before anything is looked up.**

| Nature | Recognised by | Example | Masked? |
|---|---|---|---|
| quantity | no significant word left | "over 65" | no — it comes from the analyst |
| concept | the glossary covers it | "mortality rate" | no — it is a column name |
| person | entity type `person` | "Mr. Bensalah" | request stopped |
| value | everything else | "aspirin" | **yes → `:v1`** |

> **Why this step exists.** The index *always* finds something. Without
> classification, `over 65` resolved to `admitdxtext = '65'` with score 1.00 and
> `mortality rate` to `'Low mortality risk'` with 0.79. Both wrong, both above the
> threshold, so nothing flags them — the SQL runs and answers a different
> question. That is the worst failure mode there is: a clean error is repairable,
> a silently wrong answer is not.

**Resolution.**

```
'aspirin' → medication.drugname = 'aspirin'                       1.00
'sepsis'  → diagnosis.diagnosisstring
            = 'cardiovascular|shock / hypotension|sepsis'         1.00
```

Prefix search is what makes this work — the database holds
`ASPIRIN EC 81 MG PO TBEC`. And `sepsis` becomes a hierarchical string, which the
analyst never has to know about.

```
tables : {medication, diagnosis, patient}     latency : 235 ms
```

---

## Stage 2a — Mask *(local)*

```
before   How many patients received aspirin for sepsis?
after    How many patients received :v1 for :v2?

kept here, never written anywhere else:
  :v1 = 'aspirin'
  :v2 = 'cardiovascular|shock / hypotension|sepsis'
```

Longest mentions are replaced first — otherwise "aspirin" before "aspirin 81 mg"
would leave `:v1 81 mg`. Numbering restarts at 1 every question, so the provider
cannot track a value across requests.

---

## The gate

The prompt is split by origin, and each part checked by the rule that fits it.

| Piece | Origin | Checked by | |
|---|---|---|---|
| Instructions | us | fingerprint of a source constant | PASS |
| Schema DDL | generated | **regenerated and compared** | PASS |
| `:vN` descriptions | generated | value-vocabulary membership | PASS |
| Domain note | config | is it a declared note? | PASS |
| **Masked question** | **the user** | **word by word** | PASS |

> **Why regenerate rather than trust the label.** If we simply believed a segment
> marked "schema", slipping a value into it would carry the value out.
> Regenerating the DDL from the database and comparing means any interpolation
> breaks the match.

For the masked question, two layers: does the text contain a database value (exact
— we own the database), and is any unknown word part of the value vocabulary.
0.033 ms, no database query.

---

## Stage 2b — Generate *(cloud)*

Everything the provider receives:

```
Schema:
CREATE TABLE medication (  -- 75604 rows
  medicationid INT PRIMARY KEY,
  patientunitstayid INT,
  drugname TEXT,
  …
  FOREIGN KEY (patientunitstayid) REFERENCES patient(patientunitstayid)
);
CREATE TABLE diagnosis (  -- 24978 rows … );
CREATE TABLE patient   (  -- 2520 rows  … );

Bound parameters:
  :v1 = a value from medication.drugname
  :v2 = a value from diagnosis.diagnosisstring

Domain notes:
- A patient may have several stays; patientunitstayid identifies the ICU stay,
  not the person.

Question: How many patients received :v1 for :v2?
```

Names, types, foreign keys, row counts, a note we wrote, and a question with
holes. **No data.** Groq does not know this is about aspirin or sepsis — only that
a `drugname` value and a `diagnosisstring` value are wanted.

3 tables of 31 were sent. All 31 would cost ~6,000 tokens and the model would
drown as much as it would learn.

```
groq/openai/gpt-oss-120b    1 call, 0 repairs, 859 tokens, 1,122 ms
```

```sql
SELECT COUNT(DISTINCT m.patientunitstayid) AS patient_count
FROM medication m
JOIN diagnosis d ON m.patientunitstayid = d.patientunitstayid
WHERE m.drugname = :v1 AND d.diagnosisstring = :v2
```

Correct join, `COUNT(DISTINCT …)` so a patient with several stays is not counted
twice, and bound parameters instead of literals.

---

## Stage 3 — Validate, execute, answer *(local)*

**Validate.** One statement, no writes, real columns, and — most important for
privacy — **the parameters are actually used**. A missing `:v1` would mean the
model wrote the value in clear, so the question being asked is no longer the one
we think.

> This caught a real defect: `SELECT AVG(glucose) FROM patient`, inventing a
> column that exists in `apacheapsvar` but not in `patient`. Before, it only
> showed at execution time — too late to repair.

**Execute.**

```python
execute(sql, {"v1": "aspirin", "v2": "cardiovascular|shock / hypotension|sepsis"})
→ 1 row, ['patient_count'] = 4     4.8 ms
```

Read-only enforced by SQLite itself (`mode=ro`), time-bounded, row-capped. This is
the only place the real values meet the query — passed *beside* the SQL text,
never inside it.

**Write the answer.** Qwen3-1.7B, in this process. It is the only component that
sees real data, so it must be local — under `PRIVACY_MODE=strict` a network
backend is refused at load time, not at call time.

> **4 patients received aspirin for sepsis.**

Its prompt says: compute nothing, invent nothing. The numbers come from the
database; the model only makes sentences. That is why a model this small is
enough.

---

## The tally

| Stage | Where | Latency | Saw |
|---|---|---|---|
| Understand | local | 235 ms | the full question |
| Mask | local | < 1 ms | the full question |
| Generate | **cloud** | 1,122 ms | **the schema + `:v1`, `:v2`** |
| Validate + execute | local | 4.8 ms | the real values |
| Write | local | ~5 s | the real data |

---

## Three shorter cases

**A proper name stops everything.**

```
Did Mr. Bensalah receive his insulin?   →  STOPPED, nothing sent
```

The database is de-identified and holds no names, so the question has no answer.
The name is never looked up either — fuzzy search would match it to some arbitrary
value.

**The gate catches what stage 1 missed.**

```
What is the average potassium level?

stage 1 : GLiNER2 does not spot 'potassium'
stage 2a: nothing to mask, the question would go as is
gate    : BLOCKED — 'potassium' is in the value vocabulary
```

This is the system working, not failing. The analyst gets a clear error instead of
a silent leak.

**A value made of ordinary words.**

```
How many patients were discharged to a Skilled Nursing Facility?
```

Three harmless words — but `Skilled Nursing Facility` is stored verbatim in
`patient.hospitaldischargelocation`. The gate sweeps n-grams longest first, so the
*phrase* is caught even though each word alone is exempt.

---

## What this proves, and what it does not

**Proves:** the provider receives only schema names and symbols, readable in the
prompt above; values are bound, never concatenated; every outgoing byte is
journalled, so the claim is checkable afterwards.

**Does not prove:** that the leak rate is zero — 1.12 % of information-bearing
values would cross if unmasked, being words that are also column names. That the
demo runs on a UNIMED server — it runs on a development machine, and the guarantee
is architectural. That the understanding figures generalise — the question set was
written and annotated by the same person who fixed the code by looking at its
failures.
