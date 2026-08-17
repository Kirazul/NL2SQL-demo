# Beginner's guide

This explains the whole project from zero. No knowledge of databases or AI is
assumed. Every term is explained the first time it appears.

**Contents**

1. [What we are trying to do](#1-what-we-are-trying-to-do)
2. [Why it is hard](#2-why-it-is-hard)
3. [The idea that solves it](#3-the-idea-that-solves-it)
4. [The database](#4-the-database)
5. [Step 1 — reading the question](#5-step-1--reading-the-question)
6. [Step 2 — finding the real value](#6-step-2--finding-the-real-value)
7. [Step 3 — the glossary](#7-step-3--the-glossary)
8. [Step 4 — sorting mentions](#8-step-4--sorting-mentions)
9. [Step 5 — masking](#9-step-5--masking)
10. [Step 6 — the gate](#10-step-6--the-gate)
11. [Step 7 — asking the cloud](#11-step-7--asking-the-cloud)
12. [Step 8 — checking what came back](#12-step-8--checking-what-came-back)
13. [Step 9 — running the query](#13-step-9--running-the-query)
14. [Step 10 — writing the answer](#14-step-10--writing-the-answer)
15. [The four architectures](#15-the-four-architectures)
16. [What the numbers say](#16-what-the-numbers-say)
17. [How to run it](#17-how-to-run-it)
18. [Problems we hit](#18-problems-we-hit)

---

## 1. What we are trying to do

A **database** is a set of tables, like spreadsheets that reference each other.
Ours has 31 tables of hospital records: patients, the drugs they received, their
lab results, their diagnoses.

Someone at the company wants to know:

> "How many patients received aspirin for sepsis?"

The answer is in the database. But to get it you must write **SQL**, the language
databases understand:

```sql
SELECT COUNT(DISTINCT m.patientunitstayid)
FROM medication m
JOIN diagnosis d ON m.patientunitstayid = d.patientunitstayid
WHERE m.drugname = 'aspirin'
  AND d.diagnosisstring = '…|sepsis'
```

Reading that line by line:

- `SELECT COUNT(DISTINCT …)` — count, without counting anyone twice
- `FROM medication` — start from the drugs table
- `JOIN diagnosis ON …` — pair each drug row with the diagnosis rows of the same
  hospital stay
- `WHERE …` — keep only aspirin, and only sepsis

A business analyst cannot write that. So they ask a developer and wait, sometimes
days, for a two-minute question.

**Our goal: let them ask in English, and get the answer.** That is what
"text-to-SQL" means — turning text into SQL.

---

## 2. Why it is hard

A **large language model** (LLM) — ChatGPT, Claude, Llama — writes that SQL
easily. Give it the question and the shape of the database, and it produces the
query.

The catch: those models run on someone else's computers. Google's, OpenAI's,
Groq's. Using one means **sending your question over the internet to a stranger.**

And the question is about your data:

- "How many patients received aspirin" reveals that you track that drug.
- "Did patient Bensalah get his insulin" reveals a name and a condition.

On health data that is not acceptable — legally (GDPR) or commercially.

There are small models you can run on your own machine, which send nothing
anywhere. But they write poor SQL: they lose track of which table joins to which.

> **The dilemma:** the model that writes good SQL is exactly the one you do not
> want to tell anything.

---

## 3. The idea that solves it

Writing SQL needs two very different kinds of knowledge:

| What is needed | Example | Who can supply it |
|---|---|---|
| **Structure** | "join `medication` to `diagnosis` using `patientunitstayid`" | needs a big model |
| **Values** | "the drug is spelled `aspirin`" | just a lookup in our own database |

The insight: **the cloud model only needs the structure.** The values we can
handle ourselves, locally.

So before sending anything, we replace the values with symbols:

```
the analyst asks   How many patients received aspirin for sepsis?
we send            How many patients received :v1 for :v2?
```

The cloud writes:

```sql
… WHERE m.drugname = :v1 AND d.diagnosisstring = :v2
```

And **we**, on our own machine, put the real values back when we run it. The
cloud provider never saw the drug or the condition.

> **Why `:v1` and not something like `<VALUE_1>`?**
> `:v1` is SQL's own syntax for a **bound parameter** — a blank the database fills
> in itself, separately from the query text. So two things happen at once: the
> model gives us SQL we can run directly, and the real value is never pasted into
> the query string. That makes **SQL injection** — tricking a database by hiding
> commands inside a value — impossible by construction rather than by filtering.

Here is the whole pipeline. Only one arrow crosses to the cloud:

```
1  UNDERSTAND                                    all local
   question → find the values → look them up

2  MASK AND GENERATE
   values → :v1, :v2                             local
   THE GATE: check every byte about to leave     local
   ──────────────── trust boundary ────────────────
   the cloud model writes the SQL                cloud

3  EXECUTE AND ANSWER                            all local
   check the SQL → run it → write the sentence
```

---

## 4. The database

We wanted a **real** public database, not an invented one. Real values are far
messier to anonymise than generated ones, so the test is harder, not easier.

| Candidate | Verdict |
|---|---|
| ChEMBL | 5 GB — too heavy for the machine |
| DrugCentral | 1.3 GB, and needs converting |
| FAERS (FDA) | 7 tables — too few to join |
| Kaggle "pharma" datasets | flat files, no structure at all |
| **eICU-CRD v2.0.1** | **31 tables, real structure — selected** |

**eICU-CRD** is published by MIT on PhysioNet. It holds intensive-care records
from 186 US hospitals: 4,605,753 rows. It is **de-identified** — no names, no real
dates — and released under an open licence, so anyone can download it.

### The problem we found

The file MIT publishes has **no primary keys, no foreign keys, and no indexes**.
Three terms, explained:

- A **primary key** is the column that identifies a row uniquely. In `patient`,
  that is `patientunitstayid` — one row per hospital stay.
- A **foreign key** says that a column in one table points at another table.
  `medication.patientunitstayid` points at `patient.patientunitstayid`, meaning
  "this drug was given during this stay". It is what makes a `JOIN` possible.
- An **index** is a lookup structure that makes searching fast, the way a book's
  index beats reading every page.

Missing indexes would only make things slow. Missing **foreign keys** are fatal
here, for a reason specific to this project:

> **The schema is the only thing we ever show the cloud model.** A model told that
> `medication.patientunitstayid` references `patient.patientunitstayid` knows how
> to join those tables. A model shown the same columns with no declared
> relationship has to guess — and guesses wrong.

So we rebuilt them, taking the primary keys and indexes from MIT's own code
repository and adding the foreign keys from the documented relationships:

| | Before | After |
|---|---|---|
| Primary keys | 0 | 31 |
| Foreign keys | 0 | 30 |
| Indexes | 0 | 91 |
| Size | 282 MB | 428 MB |
| Integrity violations | — | **0** |

"Integrity violations: 0" means every foreign key we declared actually holds:
there is no drug row pointing at a stay that does not exist. We did not assume it,
we checked.

> **A lesson in passing.** Our first indexes made one query **slower** — 107 ms
> against 46 ms with no index at all. Asking the database to explain its plan
> (`EXPLAIN QUERY PLAN`) showed it was starting from an index that barely narrowed
> anything down. Replacing it with a **composite** index over two columns at once,
> `(patientunitstayid, labname)`, brought it to 42 ms. Two other indexes were
> deleted outright: 64 MB of disk for no gain. A badly chosen index costs instead
> of helping.

### What the data is actually like

Real data has traps. Three that would each produce a confident wrong answer:

- **`patient.age` is text, not a number.** Ages above 89 are stored as the string
  `"> 89"`, which is a de-identification requirement. So `WHERE age > 65` compares
  text, not numbers, and quietly returns the wrong rows.
- **Two versions of the same score coexist** in `apachepatientresult`. Without
  filtering on the version column, every stay is counted twice.
- **`medication.drugname` is free text.** Dozens of spellings for one drug:
  `ASPIRIN`, `ASPIRIN EC 81 MG PO TBEC`, `aspirin 81mg`, and so on.

We left a misspelled column name (`cplcareprovderid`) uncorrected on purpose:
fixing it would break every query written against the real eICU.

---

## 5. Step 1 — reading the question

The first thing to do is find which parts of the sentence are *values*.

We use **GLiNER2**, a small model (208 million parameters) that reads a sentence
and points at the entities in it. What makes it useful here is that you describe
the things you want in plain English, at the moment you call it:

```python
["person name", "drug or medication name", "medical condition or diagnosis",
 "laboratory test name", "microorganism or bacteria",
 "medical procedure or treatment", "hospital or unit name"]
```

That is called **zero-shot**: no training, no examples. To move this system to a
bank instead of a hospital, you change that list and nothing else.

For our question it returns:

| Mention | Type | Confidence |
|---|---|---|
| `aspirin` | drug | 1.00 |
| `sepsis` | diagnosis | 1.00 |

It ignored "patients" and "how many" — those are question structure, not values.
Nothing to hide there.

> **Why not just use an LLM to do this?** It would work — but we would have to
> send it the question, which is the exact thing we are trying to avoid. GLiNER2
> runs **inside our own program**, on the processor, with **no network call at
> all**. That is what makes "local" a fact you can check rather than a claim: it
> takes 2.8 seconds to load once, then about 160 ms per question, and it still
> works with the network unplugged.

### The type that was added afterwards

eICU is de-identified, so it contains **no person names**. But an analyst will
still write "did Mr. Bensalah get his insulin?".

Without an explicit `person name` type, that name was **not spotted at all** — so
nothing masked it, and it would have gone to the cloud. Adding the type fixed it.
Now a name is never looked up in the database, and the request is **stopped**: the
database has no names, so the question has no answer, and there is no reason to
send anything anywhere.

On our test set: **8 names out of 8 protected.**

---

## 6. Step 2 — finding the real value

The analyst writes `aspirin`. The database contains `ASPIRIN EC 81 MG PO TBEC`.

```sql
WHERE drugname = 'aspirin'    -- 0 rows
```

So a mention has to be translated into the exact stored value. We do this
**locally, before any cloud call** — which is also precisely what makes masking
possible. The cloud gets `:v1`; we keep the real string.

To do that we build a **value index**: a searchable list of the values that exist
in the database, built once.

### Why we do not index everything

On eICU we could — it is small. But the architecture has to still work on a real
company database where a single column can hold millions of values. Two problems
appear there:

1. an exhaustive index grows with the data and takes hours to rebuild;
2. noise drowns the signal — indexing thousands of numeric lab measurements helps
   nobody ask a question, and makes every search result worse.

So each column is sorted automatically into one of three tiers:

| Tier | What it is | What we do |
|---|---|---|
| **A — indexed** | 5,000 or fewer distinct text values: drug names, diagnoses, lab names | store them all in a full-text index |
| **B — on demand** | too many values to store | store nothing; query the database at question time |
| **C — excluded** | measurements, dates, free text, identifiers | ignore entirely |

**The point of this design:** the cost depends on the number of **columns**, never
on the number of **rows**. A table with two billion rows but only 300 distinct lab
names costs exactly as much to index as one with two thousand rows.

Result on eICU: 136 text columns examined, 128 indexed, 30,854 values, **2.43 MB**
— about 0.6 % of the database.

### How the search works

The index uses **FTS5**, SQLite's full-text search. A query looks for the prefix
(`aspirin*`), then results are re-ranked by how closely they resemble what was
typed.

One subtlety: for "acute renal failure" the search first requires *all* the words,
and only widens to *any* of them if that finds nothing. Widening straight away
would flood the results with everything containing "acute" and push the right
value out.

### A trap worth knowing

The first rule for excluding identifier columns was "does the name end in *id*?".

- `volumeoffluid` ends in "id" and is not an identifier.
- Worse, `patient.uniquepid` **is** a patient identifier and slipped into the
  index — 1,841 patient IDs stored where they had no business being.

The fix was a statistical rule instead of a spelling rule: that column's values
are 73 % unique, which is how identifiers behave, so it is excluded whatever it is
called.

> **An identifier is recognised by its behaviour, not by its spelling.**

---

## 7. Step 3 — the glossary

The index resolves *content*. The glossary resolves *vocabulary*.

When an analyst says "mortality", there is no value called mortality anywhere —
it is the name of a **column**. The glossary makes that link explicitly:

```yaml
mortality:
  synonyms: [mortality, death, died, deceased, survival, fatal]
  columns: [patient.hospitaldischargestatus,
            apachepatientresult.actualhospitalmortality]
  note: "Discharge status is recorded as text, not as a boolean."
```

It does two jobs:

1. **narrows the schema** we send the cloud to the tables that matter;
2. **aims the value search** at the right columns.

28 terms, checked automatically against the live schema so the file cannot drift
away from the database.

> **A leak we avoided narrowly.** The glossary listed `male, female` as synonyms
> for gender. But `Female` **is** a value stored in `patient.gender`. So the
> pipeline treated the word as a concept — meaning *do not mask it* — and it would
> have travelled to the cloud in clear.
>
> Fixed, and then locked with a test that forbids any synonym from also being a
> value of its own column. The test immediately found four more cases.

---

## 8. Step 4 — sorting mentions

This step exists because of the single most dangerous behaviour we found:
**the index always finds something.**

Give it any phrase and it returns a match, with a confidence score:

| Mention | What the index returned | Score |
|---|---|---|
| `mortality rate` | `cplitemvalue = 'Low mortality risk'` | 0.79 |
| `over 65` | `admitdxtext = '65'` | 1.00 |
| `infection` | `admitdxname = 'Renal infection/abscess'` | 1.00 |

All three are wrong. Two of them score above the confidence threshold, so nothing
flags them. The query then runs perfectly, raises no error, and answers **a
different question than the one asked**.

> That is the worst possible failure. A crash is repairable. A confident wrong
> answer is not — nobody knows to check it.

The fix is to classify every mention *before* looking anything up:

| Nature | How we recognise it | Example | Masked? |
|---|---|---|---|
| **quantity** | no meaningful word left once grammar is stripped | "over 65" | no — the number came from the analyst, not the database |
| **concept** | the glossary covers all its meaningful words | "mortality rate" | no — it is a column name |
| **person** | GLiNER2 typed it as a person | "Mr. Bensalah" | request stopped |
| **value** | everything else | "aspirin" | **yes → `:v1`** |

Only the last kind is ever looked up in the database.

A second, related fix: when a scoped search finds nothing we widen it to all
columns. But "hemoglobin", absent from the lab-name column in this database,
landed in the diagnosis column with a score of 1.00. Scores from a widened search
are now **penalised**, so the result drops below the threshold and goes back to
the analyst instead of being used.

---

## 9. Step 5 — masking

Now we replace each value with a symbol.

```
before   How many patients received aspirin for sepsis?
after    How many patients received :v1 for :v2?
```

The mapping stays on our machine and is never written anywhere else:

```python
{":v1": "aspirin",
 ":v2": "cardiovascular|shock / hypotension|sepsis"}
```

(That second value shows something useful: eICU stores diagnoses as a hierarchy,
from general to specific, separated by pipes. The analyst never has to know —
which is exactly what resolution spares them.)

Three details that matter more than they look:

**Longest mentions are replaced first.** Otherwise replacing "aspirin" before
"aspirin 81 mg" would leave `:v1 81 mg` sitting in the question.

**Numbering restarts at 1 for every question.** If `:v1` always meant aspirin, the
provider could count how often it appears across requests and work backwards to
what it is. A stable symbol is a tracking identifier. Ours are not.

**The mapping is the only secret in the system.** Everything else can be shown to
anyone.

---

## 10. Step 6 — the gate

This is the heart of the project: the single place where text can reach the cloud.

### The usual approach, and why it is not enough

Off-the-shelf privacy tools **detect** sensitive entities with a model, then
replace them. Detection is a heuristic — it has a *recall rate*, meaning some
fraction of sensitive items it never notices. And the failure is invisible:
nothing tells you what was missed.

### Why we can do better

**We own the database.** For every indexed column we have the complete list of
values that exist in it. Checking a word against that list is not an estimate —
it is exact.

### How the gate works

The outgoing text is not checked as one lump. It is split into pieces, each
labelled with **where it came from**, and each piece is checked by the rule that
actually proves *that kind* of text safe:

| Piece | Where it came from | How it is checked | Can it leak? |
|---|---|---|---|
| Instructions | written by us, in the source code | fingerprint of the constant | no |
| Schema (the `CREATE TABLE` text) | generated from the database | **regenerated and compared** | no |
| `:v1` descriptions | generated | membership in the value vocabulary | no |
| Domain notes | our glossary file | is it one of the declared notes? | no |
| **The masked question** | **the user** | **word by word** | **yes** |

> **Why regenerate instead of trusting the label?** If we simply believed a piece
> marked "schema", then slipping a value into it would carry that value straight
> out. By rebuilding the schema text from the database and comparing the two, any
> interpolation breaks the match — and the piece falls back to word-by-word
> checking.

For the one untrusted piece, the question, there are two layers:

1. **Does the text contain a database value?** Exact, because we hold the
   inventory. This catches whole phrases too: `Skilled Nursing Facility` is three
   harmless-looking words, but it is stored verbatim in a discharge column, so the
   phrase is recognised even though each word alone is fine.
2. **Is any unknown word part of the value vocabulary?** "readmitted" passes, it
   appears nowhere in the data. "MELOXICAM" is blocked, it does.

Cost: **0.033 ms per check**, against a set built once at indexing time. No
database query at request time. Past 200,000 words the set becomes a **Bloom
filter** — a compact structure that answers "have I seen this?" in 3.6 MB for 3
million words. Its error only runs one way: it can wrongly *refuse* a word, never
wrongly *allow* one. For a privacy gate, that is the correct direction.

### What this replaced

An earlier version checked the **entire** prompt word by word. It worked. It also
required allowing ~350 ordinary English words by hand — "identifies",
"hierarchical", "boolean", "readmitted" — purely so that *our own sentences* could
pass. Every new question added another word, and every addition widened the hole.

> That is not an architecture, it is endless catching-up.

| | Before | After |
|---|---|---|
| Hand-written allowed words | ~350 | **0** |
| Upkeep per new question | 1 word | none |
| Residual leak rate | 1.82 % | **1.12 %** |

### Why the leak rate is not 0 %

Measured over the 11,293 values that carry information, 1.12 % would cross the
gate if they were not masked. The cause is specific and worth stating plainly:
the table `apacheapsvar` has **columns** named `albumin`, `urine`, `bun`, `wbc` —
and those same words are also **lab names stored as values** elsewhere.

Blocking those words would forbid the model from ever writing
`SELECT albumin FROM apacheapsvar`. So the residue is exactly the overlap between
schema identifiers and stored values. It is measured, published, and not swept
under the rug.

### The gate catches what step 1 misses

```
What is the average potassium level?

step 1 : GLiNER2 does not spot 'potassium'
step 5 : nothing to mask, the question would go as it is
gate   : BLOCKED — 'potassium' is in the value vocabulary
```

**This is the system working, not failing.** The first stage missed a value and
the gate stopped it before it left. The analyst gets a clear error instead of a
silent leak.

---

## 11. Step 7 — asking the cloud

This is the only network call in the whole pipeline. Here is *everything* the
provider receives:

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

Read that carefully. It contains table and column **names**, types, foreign keys,
**row counts** (metadata, not content), a note we wrote ourselves, and a question
with holes in it.

**No data.** Groq does not know this is about aspirin. It does not know it is
about sepsis. It knows only that a `drugname` value and a `diagnosisstring` value
are wanted.

**Only 3 tables of 31 were sent.** Sending all 31 would cost about 6,000 tokens
per question, and the model would drown in them as much as it would learn from
them. The three come from step 1: the tables the glossary implies, plus the tables
of the resolved values, plus `patient`, which is the hub almost every join passes
through.

The reply:

```
groq/openai/gpt-oss-120b     1 call, 0 repairs, 859 tokens, 1,122 ms
```

```sql
SELECT COUNT(DISTINCT m.patientunitstayid) AS patient_count
FROM medication m
JOIN diagnosis d ON m.patientunitstayid = d.patientunitstayid
WHERE m.drugname = :v1 AND d.diagnosisstring = :v2
```

Exactly right: the correct join, `COUNT(DISTINCT …)` so a patient with several
stays is not counted twice, and the bound parameters instead of literal values.

> **One call, not five.** The original design asked for five candidate queries per
> question and picked the best. The free Groq tier allows 30 requests per minute,
> so five calls per question caps you at six questions a minute — an evaluation of
> 107 questions becomes impractical. Instead: one call, then a **second only if the
> query is rejected**, with the exact reason handed back. Telling the model *why*
> it was wrong corrects better than rolling the dice again. Measured over 107
> questions: **0 repairs needed.**

---

## 12. Step 8 — checking what came back

This SQL was written by a **remote model**. Treating it as trustworthy would be
absurd, so before it runs we check:

- **one statement only** — `SELECT 1; DROP TABLE patient` is refused
- **no write commands**, no dangerous functions
- **comments cannot hide a statement**
- **every column named actually exists** in the tables cited
- **the parameters we supplied are actually used**

That last one matters most for privacy. A missing `:v1` would mean the model wrote
the value in clear instead of binding it — which means the query no longer asks
the question we think it asks.

> This check caught a real defect during evaluation: the model wrote
> `SELECT AVG(glucose) FROM patient`, inventing a column that exists in
> `apacheapsvar` but not in `patient`. Before the check existed, the mistake only
> surfaced when the query ran — too late to repair it.

We also **sweep the output**: standard practice is to sanitise the input *and*
check the response. The specific risk here is the model writing
`WHERE drugname = 'aspirin'` instead of using the parameter. Nothing would have
leaked outward — but the query would no longer reflect the masked value, and
running it would silently answer something else.

---

## 13. Step 9 — running the query

```python
execute(sql, {"v1": "aspirin",
              "v2": "cardiovascular|shock / hypotension|sepsis"})

→ 1 row, ['patient_count'] = 4          4.8 ms
```

Three guarantees, applied when the connection is opened rather than left to
whoever calls it:

- **genuinely read-only** — the `mode=ro` connection is refused for writes by the
  SQLite engine itself. It is not a convention we promise to respect.
- **time-bounded** — a badly formed join is interrupted rather than running
  forever.
- **volume-bounded** — the number of rows returned is capped.

This is the only place in the whole system where the real values meet the query.
They are handed to the database **beside** the SQL text, never inside it.

---

## 14. Step 10 — writing the answer

The last component turns rows into a sentence. It is **the only part of the
pipeline that sees the real data** — the cloud model only ever saw a schema.

So it must be local, without exception. Under `PRIVACY_MODE=strict` a
network-based backend is refused **when the program loads**, not when it is
called: a component that
could leak must not even be constructed.

We use **Qwen3-1.7B**, compressed to about 1.1 GB, running on the processor. It
is deliberately small, and that is the argument: the SQL was written by a much
larger model, and the numbers come from the database. The local model only has to
turn them into a sentence. If a 1.7-billion-parameter model is enough for that,
then the only part of the work that must leave is the part that can leave blind.

Given:

```
Question: How many patients received aspirin for sepsis?
Query results: patient_count = 4
```

It returns:

> **4 patients received aspirin for sepsis.**

Its instructions carry two rules that do all the work: **compute nothing, invent
nothing.**

> **Two problems we hit.** It **looped**, repeating the same sentence until it
> burned 400 tokens — 19 seconds. The cause was that we used raw text completion
> when Qwen3 expects its chat format, so it never emitted its stop signal. Fixed:
> 4.7 to 6.5 seconds.
>
> And it **misread a single number**. Given a one-cell table whose header was
> `COUNT(DISTINCT m.patientunitstayid)`, it answered "no records found" for a
> result of 4. Two fixes: ask the cloud model to **name its output columns**
> (`AS patient_count`), and present a single value as a fact rather than as a
> table.

If the writer fails for any reason, we fall back to printing a plain table. The
numbers are already correct by that point; only the sentence is missing.

---

## 15. The four architectures

To know what the protection costs, you have to compare it against something. So
the same pipeline is built four ways:

| | Writes the SQL | Writes the answer | What leaves | Values out |
|---|---|---|---|---|
| **Full Cloud** | cloud, real question | cloud, from the rows | the question **and the results** | all of them |
| **Hybrid** | cloud, from symbols | local | schema + masked question | 0 |
| **Hybrid Opaque** | cloud, from labels | local | labels only | 0 |
| **Full Local** | local 1.7 B model | local | nothing | 0 |

**Full Cloud** is the version this project argues against. It sends everything —
the question as typed, and then every cell of every row it asks the provider to
summarise. It exists because it is the accuracy ceiling: without it, "our approach
is nearly as good" would be a claim with no measurement behind it. It runs with
the gate deliberately bypassed, the bypass is refused under `PRIVACY_MODE=strict`, and every
bypass is written to the audit log. The baseline is measured, not concealed.

**Hybrid** is the main design, described above.

**Hybrid Opaque** goes one step further. Hybrid still reveals one thing: the
schema. 31 table names and 391 column names describe a business even when no row
does — anyone reading them knows this is a hospital. So this arm relabels them
too:

```
CREATE TABLE medication (          CREATE TABLE t3 (
  medicationid INT PRIMARY KEY,      c1 INT PRIMARY KEY,
  patientunitstayid INT,      →      c2 INT,
  drugname TEXT                      c7 TEXT
);                                 );

"how many patients received :v1"  →  "how many c2 received :v1"
                                     ":v1 is a value of t3.c7"
```

The provider assembles a join path over a *shape*. It is never told the shape is a
hospital. The SQL that comes back is translated to real names on our machine, and
the label dictionary is **redrawn for every question** — so `t3` is a different
table next time, and a provider cannot accumulate one across requests.

Removing schema names normally wrecks text-to-SQL, because those names carry most
of the meaning. It works here for a specific reason: **the meaning was already
resolved locally.** We already know the value lives in a particular column, so the
prompt can state `:v1 is a value of c7` outright. What is left for the provider is
mechanical — follow the foreign keys, place the aggregate — and that survives
relabelling far better than "work out what each column means".

It is still expected to cost some accuracy. Measuring that cost is the point.

**Full Local** sends nothing at all; the small model writes the SQL itself. Expect
it to be the weakest, and read that as the finding rather than a failure. The size
of the gap between it and the others is what justifies renting a large model for
the SQL step in the first place.

The four are built as **state machines** that share every stage they have in
common, so a fix to execution is a fix in all four — and the diagrams in the
notebooks are generated from the compiled object, so no picture can show an
architecture that is not the one running.

---

## 16. What the numbers say

**107 questions, annotated by hand**, in two sets.

| Metric | Standard (79 q) | Hard (28 q) |
|---|---|---|
| Extraction recall | 93.8 % | 100 % |
| **Resolution accuracy** | **100 %** | **100 %** |
| Classification accuracy | 93.0 % | 80.0 % |
| **Person names protected** | — | **100 %** |
| **Full understanding** | **91.1 %** | **96.4 %** |
| Median latency | 167 ms | 162 ms |

**All sets together: 92.5 %.**

The hard set contains cases that are *supposed* to fail: proper names, vague
questions ("Who is doing badly?"), clinical shorthand ("vanco", "afib"), typos,
and questions about things the database does not hold.

> **Two unexpected successes.** Fuzzy matching rescued `vanco` → `vancomycin` and
> `renal falure` → `renal failure`. I had annotated both as unresolvable — my
> annotation was wrong, not the code.

### An honest warning about that number

I wrote these questions, I annotated them, and I fixed the code by looking at the
failures. That is **fitting to the test set**, not a measure of generalisation.

The number says the pipeline works on 107 known cases. It does not predict
performance on questions written by someone else. An honest evaluation needs a set
written by a third party and kept aside during development.

### What still does not work

- **GLiNER2 misses some values** — `potassium`, `glucose`, `Floor`. What is not
  spotted cannot be masked, which is the most direct leak risk there is. The gate
  catches them, but the question then fails instead of being answered.
- **Adding entity types has a cost.** A type meant to catch `Resistant` and
  `Sensitive` did not find them, *lost* `potassium` and `glucose`, and doubled the
  latency. Removed.
- **Words with two meanings** — "teaching", "antibiotic", "patients" — are
  sometimes classified as values.

---

## 17. How to run it

The simplest path is the four notebooks: notebook 1 builds everything and saves
it, and notebooks 2 to 4 read that saved output. They download nothing
themselves — attach notebook 1's output under *Add Input → Notebook Output*, and
toggle the secrets on in each notebook, since Kaggle grants access to a secret
one notebook at a time.

Locally:

```bash
python scripts/build_database.py       # download + rebuild the database (~10 min, once)
python scripts/build_value_index.py    # build the value index
bash scripts/download_models.sh        # the two local models (~1.9 GB, once)

# watch one question go through every stage
python scripts/trace_question.py "How many patients received aspirin?" --write

# the measurements
python scripts/evaluate_understanding.py   # step 1
python scripts/measure_gate.py             # the leak rate
python scripts/evaluate_pipeline.py --all  # the whole pipeline

pytest                                      # the tests, including the leak canaries
```

Each build step **skips itself if its output already exists**, so re-running is
cheap.

### Where everything lives

```
src/hybridsql/
├── config.py            settings, read from .env
├── db/                  read-only access, schema, value index
├── resources/           the glossary: business words → columns
├── providers/           GLiNER2 · cloud (Groq→OpenRouter) · Qwen3
├── security/            the gate · the SQL validator · the audit journal
├── pipeline/            understand · anonymize · generate · opaque
├── graph/               the four architectures as state machines
└── api/                 the REST service
```

The one file to read if you only read one: `security/egress_gate.py`. Everything
else is plumbing around it.

---

## 18. Problems we hit

| Problem | What we did |
|---|---|
| **Hugging Face closed Docker Spaces on the free tier** (`402 Payment Required`) | Moved the demo to Kaggle notebooks with an outbound tunnel. The service is plain FastAPI and depends on nothing specific to either host |
| **Downloads that could not resume** — 504 MB lost twice | `huggingface_hub` names its temporary file after the signed download URL, which changes on every attempt, so each retry restarted from zero. Replaced with `curl -C -`, which resumes at the right byte |
| **GLiNER2 refused to load** | A missing `encoder_config/config.json`. The error message was misleading, and worse, the pipeline fell back *silently* to a degraded path instead of failing |
| **GLiNER2's output was misread** | Results are wrapped in an `entities` key; the code was reading the type *names* as if they were the entities found |
| **Qwen3 returned 404** | The official repository only publishes the larger 1.7 GB version. The compressed one exists only in a community repository |
| **Groq returned 403** | Cloudflare was blocking the default user agent of Python's `urllib`. Neither the key nor the model was at fault |
| **A model vanished from Groq** | The catalogue had changed. Four models tested; settled on `openai/gpt-oss-120b` at 1.1 s |

> **The lesson worth keeping.** An architecture that depends on a provider's free
> tier depends on a commercial decision, not a technical constraint. That is
> exactly this project's own argument: keep the sensitive computation under your
> own control.

---

Next: [one question traced end to end](04-WALKTHROUGH.md) ·
[the architecture](01-ARCHITECTURE.md) · [the decisions](02-DECISIONS.md)
