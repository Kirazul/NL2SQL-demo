# Architecture

## The one rule

**Exactly one module may open a socket** — `providers/cloud.py` — and everything
it sends is checked first.

```
        LOCAL                                      │        CLOUD
                                                   │
 question → understand → mask → [ EGRESS GATE ] ───┼──→ writes the SQL
                                                   │
 answer ← write ← execute ← validate ← bind :v1 ←──┼─── returns the SQL
                                                   │
```

---

## The three stages

### 1. Understand — all local

```
"How many patients over 65 received aspirin?"

GLiNER2      aspirin → drug (1.00)      65 → quantity
glossary     "patients" → patient.patientunitstayid
value index  aspirin → medication.drugname = 'ASPIRIN EC 81 MG PO TBEC'
```

The index gives us two things: the exact stored spelling, **and** the column it
came from. The second is what lets the opaque arm say `:v1 is a value of c7`.

If confidence is too low we ask the analyst. We never guess — a wrongly resolved
value gives an answer that is wrong but believable.

### 2. Mask and generate — the only crossing

```
before   How many patients over 65 received aspirin?
after    How many patients over :v2 received :v1?
kept     :v1 = 'ASPIRIN EC 81 MG PO TBEC'    (never leaves)
```

Symbols are renumbered every request, so `:v1` cannot be used to track a value
across questions.

The prompt is then checked segment by segment (below), sent to Groq — OpenRouter
on failure — and comes back as parameterised SQL. If it is rejected, one repair
is attempted with the exact reason.

### 3. Validate and execute — all local

```
validator   one SELECT, no DDL/DML, real columns, :v1 actually used
execute     mode=ro, time limit, row cap, values bound by the driver
write       Qwen3-1.7B turns the rows into a sentence
```

`:v1` is bound *beside* the SQL text, never inside it — so SQL injection is
impossible by construction, not by filtering.

---

## The gate: check where text came from

The prompt is split by origin, and each part is checked by the rule that proves
*that kind* of text safe.

| Origin | Example | Checked by |
|---|---|---|
| `authored` | the instructions | fingerprint of a source constant |
| `template` | "c7 is the subject of the question" | fingerprint of the wording |
| `schema` | `CREATE TABLE medication (…)` | **regenerated from the database and compared** |
| `glossary` | "a patient may have several stays" | is it a declared note? |
| `question` | `How many patients received :v1?` | word by word — the only untrusted part |

**Why regenerate instead of trusting the label.** If we just believed a segment
marked "schema", slipping a value into it would carry the value out. Regenerating
the DDL and comparing means any interpolation breaks the match.

The word check asks "is this a database value?" against an exact set, or a Bloom
filter past 200,000 tokens. The filter's error only goes one way: it can refuse a
safe word, never permit an unsafe one. 0.033 ms per check, no database query.

An earlier version checked the *whole* prompt word by word. It needed ~350
hand-written English words just to let our own sentences pass. Now: zero.

---

## The four arms

They differ by which stages they contain, not by their code — they are LangGraph
state machines sharing every common stage (`graph/build.py`), so a fix to
`execute` is a fix in all four.

| | Full Cloud | Hybrid | Hybrid Opaque | Full Local |
|---|---|---|---|---|
| SQL by | cloud, real values | cloud, masked | cloud, relabelled schema | local 1.7 B |
| Answer by | cloud | local | local | local |
| Values out | all | 0 | 0 | 0 |
| Names out | yes | yes | **no** | no |

**Hybrid Opaque**, concretely:

```
CREATE TABLE medication (          CREATE TABLE t3 (
  medicationid INT PRIMARY KEY,      c1 INT PRIMARY KEY,
  patientunitstayid INT,      →      c2 INT,
  drugname TEXT                      c7 TEXT
);                                 );

"how many patients received :v1"  →  "how many c2 received :v1"
                                     ":v1 is a value of t3.c7"
```

The labels are redrawn every request, so `t3` is a different table next time. The
returned SQL is translated back locally.

This normally wrecks text-to-SQL, because names carry the meaning. It works here
because the meaning was already resolved locally — the provider only has to
follow foreign keys and place the aggregate.

**Full Cloud** bypasses the gate on purpose. That is what makes it the baseline.
The bypass is refused under `PRIVACY_MODE=strict` and journalled every time.

---

## Substitutes

| Interface | Default | Others | Outbound network? |
|---|---|---|---|
| Extractor | `gliner2` in-process | — | **no** |
| Local LLM | `llamacpp` in-process | `ollama`, `hf-inference` | no / no / *yes* |
| Cloud LLM | `groq` | `openrouter` | **yes** — the only one allowed |
| Database | `sqlite` read-only | — | no |

`PRIVACY_MODE=strict` refuses `hf-inference` at load time: the program will not
start if a "local" component would reach the network.

---

## Tracing

Every run is written to `traces/runs.jsonl` on local disk, and — when a key is
configured — uploaded to LangSmith in full, values included.

That is defensible here and only here: eICU is public and de-identified, and no
proprietary data exists in this project. Against real data the boundary would
have to be reimposed in this layer, and the hook for it exists — every stage
declares its **zone**, `local` or `cloud`, with no default, so a new stage has to
say which side of the line it runs on. That declaration is also what lets a reader
of a trace see at a glance which spans crossed.

---

## Deployment

Plain FastAPI, so it runs anywhere. The demo runs in a Kaggle session:
`cloudflared` opens an **outbound** tunnel and returns a public address, which
the notebook announces to a small Cloudflare Worker. The browser then talks to
the session **directly** — the Worker only ever learns *where* it is listening,
never a question or an answer.

---

## Limits

1. **eICU is not UNIMED.** The architecture does not depend on the domain.
2. **A Kaggle session is not an on-premise server.** What is proven is
   architectural.
3. **Hybrid sends table and column names.** Unavoidable in that arm — a model
   cannot join tables it has not been told exist. Hybrid Opaque closes it.
4. **1.12 % of values would cross the gate unmasked** — words that are also column
   names (`albumin`, `urine`, `wbc`). Blocking them would forbid the model from
   writing SQL over those columns. Measured, not hidden.
