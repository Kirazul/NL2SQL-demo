# Decisions

Every choice that departs from the obvious one, with the reason. Open this when a
decision is questioned.

---

## 1 — eICU-CRD, a real database

A synthetic database was the easy option. A real one is a harder test: real
values are much messier to anonymise than generated ones.

Measured, not assumed:

| Candidate | Rejected because |
|---|---|
| ChEMBL 36 | 5.2 GB → ~25 GB uncompressed, 14.3 GB free |
| DrugCentral | 1.3 GB PostgreSQL dump, conversion needed |
| FAERS 2024 | 7 tables — too few to join |
| MIMIC-IV demo | 15 MB — too small |
| Kaggle "pharma" sets | flat CSVs, no schema |
| **eICU-CRD v2.0.1** | **31 tables, real FK graph, ODbL, SQLite provided** |

---

## 2 — Bound parameters, never text replacement

The original design had the cloud return `WHERE name = 'V1'`, then we substitute
the text.

```
'V1'  →  "l'hôpital"     →  WHERE name = 'l'hôpital'   ← broken, or worse
:v1   →  bound by driver →  WHERE name = :v1           ← nothing to break
```

Beyond injection, this makes masking airtight: there is no code path where the
real value and the outgoing text meet.

---

## 3 — Fail-closed, never detection alone

The usual approach detects sensitive entities and masks what it found. A detector
that misses a term lets it through, and nothing signals the miss.

We are in a better position: **we own the database**, so for every indexed column
we have the complete list of values. Checking against it is exact, not a guess.

What is not proven safe does not leave. We will sometimes block a legitimate
query — that is the right way to be wrong here, and the block rate is measured.

---

## 4 — Verification by provenance, not by word list

*Replaces the original implementation of decision 3.*

The first gate checked the whole prompt word by word. It worked, and it did not
scale: our **own** English sentences kept being refused, so the allowlist grew to
~350 hand-written words like "identifies", "boolean", "readmitted" — purely to let
through text we had written ourselves. Each new question added one more.

Now each part is checked by the rule that fits it: fingerprints for our constants,
regeneration for the schema, word-level checking only for the question.

| | Before | After |
|---|---|---|
| Hand-written words | ~350 | **0** |
| Upkeep per new question | 1 word | none |
| Residual leak | 1.82 % | **1.12 %** |

**Why not 0 %.** `apacheapsvar` has columns named `albumin`, `urine`, `wbc` —
which are also lab names stored elsewhere. Blocking those words would forbid
`SELECT albumin FROM apacheapsvar`. The residue is exactly that overlap.

---

## 5 — Symbols redrawn every request

A stable symbol (`V1` always meaning aspirin) would be a tracking identifier: the
provider could count how often it appears and work backwards. So `:v1` is
renumbered per request — and so are the opaque arm's `t1`/`c7` labels.

---

## 6 — One call and a repair, not five candidates

The original design asked for five candidate queries per question. Free Groq caps
at **30 requests/minute** → six questions a minute.

Instead: one call, then a second **only on failure**, with the exact rejection
reason handed back. Telling the model *why* it was wrong corrects better than
sampling again. Measured over 107 questions: **0 repairs needed**.

---

## 7 — LangGraph for orchestration, nothing more

The four arms differ by which stages they contain. As four functions they would
be near-copies drifting apart; as state machines they share every common stage,
so a fix to `execute` is a fix in all four. `mermaid()` draws the graph that
actually ran, so no diagram can describe an architecture that is not running.

No LLM wrapper, no retriever, no agent. The providers stay in `providers/`, which
is what keeps "exactly one module opens a socket" checkable by reading one file.

---

## 8 — A fourth arm that hides the schema too

Hybrid still discloses 31 table names and 391 column names. Those describe the
business even when no row does.

Hybrid Opaque replaces them with `t1`, `c7`, redrawn per request, and translates
the returned SQL back locally. Stripping names normally wrecks text-to-SQL — it
works here because the meaning was already resolved locally, so the provider only
has to follow foreign keys.

It is expected to cost accuracy. **Measuring that cost is the point.**

---

## 9 — Trace everything, and declare each stage's zone

Rejecting LangSmith outright was the first position: it is a hosted dashboard, and
the local model's prompt contains real rows. But the comparison this project has
to deliver — four arms, per-stage latency and token cost — is exactly what a
tracing backend is good at, and throwing it away to protect a public,
de-identified database would be a ritual, not a control.

So everything is traced, values included. This is a demonstration on open data,
not a production deployment.

What survives is the **zone**: every stage declares `local` or `cloud` at the
point of definition, with no default. It costs nothing, it is what makes a trace
readable, and it is the hook a real deployment would redact against. A half-used
redaction path would be worse than none, so there is not one.

---

## 10 — Kaggle notebooks and a tunnel

Hugging Face closed Docker Spaces on the free tier (`402 Payment Required`, 15
August 2026). A static Space serves files only — no Python, no SQLite, no models.

So the demo runs in a Kaggle session. `cloudflared` opens an outbound tunnel and
returns a public address; the notebook announces it to a small Cloudflare Worker,
and the interface reads it back. **The browser then talks to the session
directly** — the Worker never carries a question or an answer.

Four notebooks rather than one: a single notebook downloaded 2.3 GB and rebuilt the
database on every session. Notebook 1 now builds and saves, the rest read its
output, and notebook 4 runs the whole thing in one click.

The code reaches them as an attached Kaggle dataset rather than a git clone: the
repository is private, and Kaggle's API cannot set notebook secrets, so a clone
would have meant pasting a token into every notebook by hand.

> An architecture that depends on a provider's free tier depends on a commercial
> decision, not a technical constraint. That is the project's own argument.

---

## 11 — Regenerate the API keys

**To do, before any public demo.** The Groq, OpenRouter and Hugging Face keys have
circulated in clear. `.env` is gitignored and only `.env.example` is versioned,
but the keys themselves must be reissued and kept in Kaggle Secrets.
