# The project

> Mohamed Aziz Mansour — UNIMED internship.

## The problem

A cloud model writes good SQL, but you have to send it your question — and the
question is about your data. A local model sends nothing and writes poor SQL.

**The idea:** send the cloud only the *structure* and a question with the values
removed.

```
The analyst asks   How many patients received aspirin?
The cloud sees     How many patients received :v1?
The cloud replies  ... WHERE drugname = :v1
We run it here     :v1 = 'ASPIRIN EC 81 MG PO TBEC'
```

The cloud never learns the drug. The rows never leave.

---

## The four architectures

| | Writes the SQL | Writes the answer | What leaves | Values out |
|---|---|---|---|---|
| **Full Cloud** | cloud | cloud | the question **and the results** | all |
| **Hybrid** | cloud | local | schema + masked question | 0 |
| **Hybrid Opaque** | cloud | local | `t1`, `c7` labels only | 0 |
| **Full Local** | local 1.7 B | local | nothing | 0 |

Full Cloud is the accuracy ceiling and the privacy floor. Full Local is the
opposite. The two middle arms only mean something because of that gap.

---

## The four notebooks

`python scripts/build_notebooks.py` writes them, `--push` uploads them.

| | Notebook | What it does |
|---|---|---|
| 1 | Setup | builds the database, the index and the models once, and saves them |
| 2 | Understanding | how a question is read, resolved and masked — before anything leaves |
| 3 | Architectures | Hybrid, Hybrid Opaque, Full Cloud and Full Local, compared |
| 4 | **Run All** | one click: everything above, then the live service |

Notebook 1 is the only one that downloads anything. Notebooks 2–4 read its saved
output and stop with instructions if it is not attached, rather than rebuilding
it. Secrets are attached per notebook, not once per account.

---

## What is measured

| Metric | Meaning |
|---|---|
| **Values out** | real database values in an outgoing byte. The column that decides |
| Executable | the query runs and returns rows |
| Latency | per stage |
| Cloud tokens | per arm, for cost |
| Refusals | stopped before the network — by design, not a failure |

Not declared, **journalled**: every outgoing call goes to `traces/egress.jsonl`,
and `tests/test_canary.py` tries to defeat the gate on every test run.

---

## Limits

1. **eICU is not UNIMED.** Intensive care, not pharmaceutical production. The
   architecture does not depend on the domain.
2. **A Kaggle session is not an on-premise server.** What is proven is
   architectural: the code shows nothing crosses the gate.
3. **Hybrid sends table and column names; Hybrid Opaque does not.** A limit of one
   arm, not of the project — Hybrid Opaque is what closes it.
4. **1.12 % of values would cross the gate unmasked.** Words that are also column
   names, like `albumin`. Blocking them would forbid `SELECT albumin FROM …`.
   Measured, not hidden.

---

More: [architecture](01-ARCHITECTURE.md) · [decisions](02-DECISIONS.md) ·
[indexing](03-INDEXING.md) · [a full walkthrough](04-WALKTHROUGH.md)
