# Indexing

Two different indexes, easy to confuse:

| | SQL index | Value index |
|---|---|---|
| Purpose | run queries fast | turn a mention into a real value |
| Built by | `scripts/build_database.py` | `scripts/build_value_index.py` |
| File | `eicu.db` | `value_index.db` |

---

## The SQL index

The export from PhysioNet had **no keys and no indexes at all**. We rebuilt them
from the MIT-LCP reference repository.

| | Before | After |
|---|---|---|
| Primary keys | 0 | 31 |
| Foreign keys | 0 | 30 |
| Indexes | 0 | 91 |
| Size | 282 MB | 428 MB |
| Integrity violations | — | **0** |

> **A lesson in passing.** The first indexes made a query *slower* — 107 ms
> against 46 ms with none. `EXPLAIN QUERY PLAN` showed SQLite starting from a
> poorly selective index. A composite `(patientunitstayid, labname)` fixed it:
> **42 ms**. Two indexes were removed entirely: 64 MB for no gain.

---

## The value index: the problem

```sql
-- the analyst types: aspirin
-- the database holds: ASPIRIN EC 81 MG PO TBEC
WHERE drugname = 'aspirin'    -- 0 rows
```

So the mention has to be translated **locally, before any cloud call**. That is
also what makes masking possible: the cloud gets `:v1`, we keep the value.

---

## Why not index everything

On eICU we could — it is small. But the architecture has to transfer to a real
database where one column holds millions of values, and there:

- an exhaustive index grows with the data and takes hours to rebuild;
- noise drowns signal — indexing thousands of numeric lab results helps nobody ask
  a question and degrades every ranking.

So columns are sorted into three tiers, automatically.

---

## The three tiers

| Tier | What | How |
|---|---|---|
| **A — indexed** | ≤ 5,000 distinct textual values: drugs, diagnoses, labs | FTS5, prefix search, re-ranked by similarity |
| **B — on demand** | too many to store | `LIKE` query at question time, `LIMIT 5000` |
| **C — excluded** | measurements, timestamps, free text, identifiers | nothing |

The rules that decide:

| Rule | Verdict |
|---|---|
| Identifier-shaped name (`id`, `*_id`, `*offset`) | C |
| Near-unique (> 90 %, or > 30 % if the name ends in `id`) | C |
| Mostly numeric or time-like (> 60 %) | C |
| Constant, or text over 120 characters | C |
| ≤ 5,000 distinct values | **A** |
| more | **B** |

> **Why a statistical rule and not just the name.** `volumeoffluid` ends in "id"
> without being a key. And `patient.uniquepid` *is* a patient identifier and was
> about to be indexed — 1,841 patient IDs in the index. It is 73 % unique, so it
> is excluded. **An identifier is recognised by its behaviour, not its spelling.**

**The key property:** cost depends on the number of **columns**, never on the
number of **rows**. A 2-billion-row table with 300 lab names costs the same as a
2,000-row one.

---

## Result

```
Text columns examined : 136
  Tier A (indexed)    : 128 columns — 30,854 values
  Tier B (on demand)  :   2 columns
  Tier C (excluded)   :   6 columns
Index size            : 2.43 MB   (database: 428 MB)
```

---

## Proving tier B works when eICU barely triggers it

Almost no eICU column exceeds 5,000 distinct values, so the code path carrying the
whole scaling argument would never run. The threshold is a parameter: lowering it
to 200 pushes `drugname` into tier B **exactly as a 5-million-value column would**.
`scripts/demo_scalability.py` measures it.

| Mention | Tier A | Tier B |
|---|---|---|
| `aspirin` | 15.2 ms | 188.1 ms |
| `warfarin` | 0.5 ms | 172.9 ms |

~180 ms against ~1 ms. That is the price. What it buys: zero size at rest, and a
cost that does not depend on how many values the column holds.

---

## Known limits

- **5,000 is a choice, not an optimum.** On another database it must be re-derived.
- **Tier B's `LIKE '%…%'` cannot use an SQL index.** The `LIMIT` bounds the rows
  returned, not the scan. In production the source database needs a trigram index.

---

```bash
python scripts/build_database.py      # keys + SQL indexes
python scripts/build_value_index.py   # value index
python scripts/demo_scalability.py    # tier B verification
```
