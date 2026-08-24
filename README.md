# NL2SQL

Ask a hospital database a question in English, without the data leaving the
network.

A cloud model writes the SQL having seen only the schema and a question whose
values have been replaced by symbols. The database is queried locally, and the
answer is written locally by a small model that runs on the CPU. The provider
never sees a value, and never sees a row.

```
question ──▶ understand ──▶ hide the values ──▶ [ gate ] ──▶ cloud writes the SQL
                                                                      │
answer ◀── write locally ◀── run the query locally ◀───────────────────┘
```

Built as an internship project at UNIMED on the public, de-identified
[eICU-CRD demo](https://physionet.org/content/eicu-crd-demo/2.0.1/) — 31 tables,
4.6 million rows.

## Install

```bash
pip install -e ".[local]"
cp .env.example .env          # then add your API keys
python -m nl2sql.cli database # download and build data/eicu.db
python -m nl2sql.cli index    # build the value index
python -m nl2sql.cli check    # verify everything is in place
```

## Use

```bash
python -m nl2sql.cli ask "How many patients over 65 received aspirin?"
python -m nl2sql.cli serve          # REST API on :7860
python -m nl2sql.cli bench          # compare the five variants
```

## What is in here

| Folder | |
|---|---|
| `src/nl2sql/` | the pipeline |
| `notebooks/` | five Kaggle notebooks: setup, understanding, architectures, optimization, run |
| `deploy/worker/` | the web interface, on Cloudflare |
| `docs/` | the schema reference |
| `data/` | the database, the index and the benchmark questions |

## The pipeline

```
src/nl2sql/
├── core/       the graph, the prompts, the state, the tracing
├── db/         schema, value index, column catalogue
├── nlp/        entity extraction, glossary, understanding
├── privacy/    masking, the egress gate, SQL validation
├── llm/        one module for the cloud, one for the local model
└── optimize/   five variants, and the benchmark that ranks them
```

Four architectures are compared, differing only in what leaves the machine:

| | question | schema | rows |
|---|---|---|---|
| **Full Cloud** | sent as typed | sent | sent |
| **Hybrid** | values hidden | sent | stay local |
| **Hybrid Opaque** | values hidden | renamed `t1`, `c7` | stay local |
| **Full Local** | never leaves | never leaves | stay local |

And five variants of the hybrid arm, each changing one thing: prompt size,
prompt content, model choice, sampling. `python -m nl2sql.cli bench` measures
accuracy, tokens, cost and latency, and ranks them.

## Tracing

Every step is traced — entity extraction, each value lookup, the column
arbitration, masking, every gate verdict, each model call with its token count,
validation, execution, answer writing. They go to LangSmith and to
`traces/runs.jsonl` at the same time, and the interface shows the same steps in
the same words.

`core/steps.py` names them; `core/trace.py` sends them. Nothing else imports
LangSmith.
