"""Generate the four Kaggle notebooks under `notebooks/`.

    python scripts/build_notebooks.py            write them
    python scripts/build_notebooks.py --push     write and upload to Kaggle
    python scripts/build_notebooks.py --push 3   only notebook 3

The notebooks are generated because a `.ipynb` is JSON with every line of source
escaped into a string array. Edited by hand, they rot: cells drift, stale outputs
stick to changed code, and the diffs are unreadable. Here they have one source
file and are rebuilt from it.

The chain
    1  Setup           clones the code, downloads the models, builds the database
    2  Understanding   how a question is read, before anything is sent
    3  Architectures   the four designs, run and measured
    4  Run All         the whole pipeline plus the live service

Notebook 1 is the only one that downloads anything. Notebooks 2 to 4 read its
saved output through `/kaggle/input` and stop with instructions if it is absent.

The code comes from GitHub in every notebook. The repository is private, so each
notebook needs a `GITHUB_TOKEN` secret.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

OWNER = "kirazul"
REPO = "https://github.com/Kirazul/NL2SQL-demo.git"
WORKER = "https://nl2sql.eclipse-kira.workers.dev"

QUESTION = "How many patients over 65 received aspirin?"


class Notebook:
    def __init__(self, number: int, slug: str, name: str, colour: str, subtitle: str):
        self.number = number
        self.slug = slug
        self.name = name
        self.colour = colour
        self.subtitle = subtitle
        self.cells: list[dict] = []

    @property
    def directory(self) -> Path:
        return OUT / f"{self.number}-{self.slug.split('-', 2)[-1]}"

    @property
    def kaggle_id(self) -> str:
        return f"{OWNER}/{self.slug}"

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/code/{OWNER}/{self.slug}"

    @property
    def title(self) -> str:
        return f"NL2SQL {self.number} {self.name}"


BOOK = [
    Notebook(1, "nl2sql-1-setup", "Setup", "#a1a1aa",
             "Clones the code, downloads the models, builds the database and the index."),
    Notebook(2, "nl2sql-2-understanding", "Understanding", "#34d399",
             "How an English question becomes a schema, a symbol and an exact value."),
    Notebook(3, "nl2sql-3-architectures", "Architectures", "#818cf8",
             "Four designs for the same question, run and measured."),
    Notebook(4, "nl2sql-4-run-all", "Run All", "#fbbf24",
             "The whole pipeline end to end, then the live service."),
]

BY_NUMBER = {n.number: n for n in BOOK}


# =================================================================================
#  Cells
# =================================================================================
def _fill(text: str, subs: dict[str, str]) -> str:
    for key, value in subs.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def md(nb: Notebook, text: str, **subs: str) -> None:
    nb.cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": _fill(text, subs).strip("\n").splitlines(keepends=True),
    })


def code(nb: Notebook, text: str, **subs: str) -> None:
    nb.cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _fill(text, subs).strip("\n").splitlines(keepends=True),
    })


def header(nb: Notebook) -> None:
    font = "-apple-system,BlinkMacSystemFont,Segoe UI,Inter,sans-serif"
    banner = (
        f'<div style="border-left:4px solid {nb.colour};padding:2px 0 2px 16px;'
        'margin:6px 0 18px;">'
        f'<div style="font:800 27px/1.15 {font};letter-spacing:-0.02em;">'
        f'NL2SQL <span style="font-weight:500;color:{nb.colour};">{nb.name}</span></div>'
        f'<div style="font:400 15px/1.55 {font};color:#71717a;margin-top:5px;">'
        f'{nb.subtitle}</div></div>'
    )
    nav = " &nbsp;|&nbsp; ".join(
        f"**{o.name}**" if o.number == nb.number else f"[{o.name}]({o.url})"
        for o in BOOK
    )
    md(nb, banner + "\n\n" + nav)


def next_link(nb: Notebook, closing: str = "") -> None:
    following = BY_NUMBER.get(nb.number + 1)
    if following:
        tail = f"**Next:** [{following.number}. {following.name}]({following.url})"
    else:
        tail = f"**Start again:** [1. Setup]({BY_NUMBER[1].url})"
    md(nb, "---\n\n" + (closing.strip() + "\n\n" if closing else "") + tail)


# =================================================================================
#  Setup, shared by all four notebooks
# =================================================================================
INSTALL = """
%%capture --no-stderr
!pip install -q --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \\
    "llama-cpp-python>=0.3" "gliner2>=1.3" "langgraph>=1.0" "langsmith>=0.10" \\
    "fastapi>=0.115" "uvicorn[standard]>=0.34" "pydantic-settings>=2.6" \\
    "sqlglot>=25.0" "rapidfuzz>=3.10" "pyyaml>=6.0" "httpx>=0.27" "python-dotenv>=1.0"
"""

PRELUDE = '''
import os, re, sys, json, time, shutil, subprocess
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()
WORK      = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
INPUTS    = Path("/kaggle/input")
REPO      = "{{REPO}}"

SECRETS = {
    "GITHUB_TOKEN":       "clone the code (the repository is private)",
    "GROQ_API_KEY":       "the cloud model that writes the SQL",
    "OPENROUTER_API_KEY": "fallback when Groq rate-limits",
    "LANGSMITH_API_KEY":  "tracing backend",
    "PUBLISH_TOKEN":      "announce this session to the web interface",
}
REQUIRED = {{REQUIRED}}


WHY = {}          # label -> why it could not be read, when it could not


def secret(label, default=""):
    """One secret, by label. Kaggle grants access per notebook, not per account.

    The reason a lookup failed is kept rather than swallowed: "not attached to
    this notebook" and "the backend refused" both end as an empty string, and
    without the reason the two are indistinguishable from the output.
    """
    if ON_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            value = UserSecretsClient().get_secret(label)
            if value:
                return value
            WHY[label] = "Kaggle returned an empty value"
        except Exception as error:
            WHY[label] = f"{type(error).__name__}: {str(error)[:110]}"
    return os.environ.get(label, default)


def load_secrets(project=None):
    """Read every label into the environment and print what was found.

    An empty secret is removed rather than set blank, so the package falls back to
    its own default instead of an empty string.
    """
    local = {}
    if not ON_KAGGLE and project and (project / ".env").exists():
        for line in (project / ".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                label, _, value = line.partition("=")
                local[label.strip()] = value.strip().strip("\\"'")

    for label in SECRETS:
        value = secret(label) or local.get(label, "")
        if value:
            os.environ[label] = value
        else:
            os.environ.pop(label, None)

    for label, purpose in SECRETS.items():
        if os.environ.get(label):
            state = "ok"
        elif label in REQUIRED:
            state = "REQUIRED"
        else:
            state = "-"
        print(f"  {label:<20}{state:<10}{purpose}")

    if WHY:
        print("\\n  why a secret could not be read")
        for label, reason in WHY.items():
            print(f"    {label:<20}{reason}")

    absent = [l for l in REQUIRED if not os.environ.get(l)]
    if absent:
        where = "Add-ons > Secrets, in this notebook" if ON_KAGGLE else ".env"
        print(f"\\n  Missing: {', '.join(absent)}. Set it in {where} and run this cell again.")
    return not absent


def get_code():
    """Clone the repository into a writable directory and put it on the path.

    Kaggle mounts every input read-only and notebook 1 writes a database next to
    the package, so the code never runs from where it is mounted.
    """
    if (Path.cwd() / "src/hybridsql").exists():
        return Path.cwd()

    target = WORK / "nl2sql"
    if (target / "src/hybridsql").exists():
        return target

    token = secret("GITHUB_TOKEN")
    url = REPO.replace("https://", f"https://{token}@") if token else REPO
    done = subprocess.run(["git", "clone", "--depth", "1", "--quiet", url, str(target)],
                          capture_output=True, text=True)
    if done.returncode:
        detail = done.stderr.replace(token, "***") if token else done.stderr
        raise SystemExit(
            "Could not clone the repository.\\n\\n"
            "  It is private, so this notebook needs a GITHUB_TOKEN secret:\\n"
            "  Add-ons > Secrets > attach GITHUB_TOKEN, then run this cell again.\\n\\n"
            "  Kaggle grants a secret one notebook at a time. Attaching it in\\n"
            "  another notebook does not attach it here.\\n\\n" + detail
        )
    return target
'''

# Where the built artefacts are. Kaggle has moved inputs between layouts more than
# once (`/kaggle/input/<slug>`, and now `/kaggle/input/notebooks/<owner>/<slug>`),
# and a notebook's output carries its own working directory inside that again. So
# nothing assumes a depth: one search walks outwards until it finds the file.
ARTEFACTS = '''
ARTEFACTS = {
    "database":    ("data/warehouse/eicu.db",                   None),
    "value index": ("data/warehouse/value_index.db",            None),
    "GLiNER2":     ("models/gliner2-base-v1",                   "model.safetensors"),
    "Qwen3-1.7B":  ("models/qwen3-1.7b/Qwen3-1.7B-Q4_K_M.gguf", None),
}
DEPTHS = ("", "*/", "*/*/", "*/*/*/", "*/*/*/*/", "*/*/*/*/*/")


def whole(path, probe=None):
    """Present and finished. A model directory with no weights in it is neither."""
    return (path / probe).exists() if probe else path.exists()


def find_input(relative, probe=None):
    """The first attached input carrying `relative`, at whatever depth it sits."""
    if not INPUTS.exists():
        return None
    for prefix in DEPTHS:
        for hit in sorted(INPUTS.glob(prefix + relative)):
            if whole(hit, probe):
                return hit
    return None


def attached_inputs():
    """The inputs actually attached, named by what they carry rather than by the
    directory level Kaggle happens to mount them under."""
    if not INPUTS.exists():
        return []
    markers = ("src", "data", "models", "nl2sql")
    return [p.relative_to(INPUTS).as_posix()
            for pattern in ("*", "*/*", "*/*/*")
            for p in sorted(INPUTS.glob(pattern))
            if p.is_dir() and any((p / m).exists() for m in markers)]


def locate(project):
    """Every artefact, in the working copy or in an attached input."""
    found, missing = {}, []
    for label, (relative, probe) in ARTEFACTS.items():
        if label == "value index":
            continue                       # always beside the database, see below
        local = project / relative
        path = local if whole(local, probe) else find_input(relative, probe)
        (found.__setitem__(label, path) if path else missing.append(label))

    # The package derives the index path from the database path, so the two must
    # be in the same directory. Looking for it anywhere else would resolve here
    # and fail there.
    if "database" in found:
        index = found["database"].with_name("value_index.db")
        found["value index"] = index if index.exists() else missing.append("value index")
    else:
        missing.append("value index")
    return found, [m for m in missing if m]


def configure(found):
    os.environ["DB_PATH"]             = str(found["database"])
    os.environ["GLINER_MODEL"]        = str(found["GLiNER2"])
    os.environ["LOCAL_LLM_GGUF_PATH"] = str(found["Qwen3-1.7B"])
    os.environ["LOCAL_LLM_THREADS"]   = str(max(2, os.cpu_count() or 4))
    os.environ["LOCAL_LLM_BACKEND"]   = "llamacpp"
    os.environ["PRIVACY_MODE"]        = "demo"
    os.environ["LANGSMITH_PROJECT"]   = "nl2sql"
    os.environ["LANGSMITH_TRACING"]   = "1" if os.environ.get("LANGSMITH_API_KEY") else "0"


def size_mb(path):
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
    return path.stat().st_size / 1e6 if path.exists() else 0.0


def show(found):
    for label, (relative, _) in ARTEFACTS.items():
        path = found.get(label)
        if path is None:
            print(f"  {label:<14}{'missing':>10}")
            continue
        root = path.parents[len(Path(relative).parts) - 1]
        if WORK in path.parents:
            where = "built here"
        elif INPUTS.exists() and (INPUTS in root.parents or root == INPUTS):
            where = root.relative_to(INPUTS).as_posix()
        else:
            where = str(root)
        print(f"  {label:<14}{size_mb(path):>9.0f} MB   {where}")
'''

DRIVER_BUILD = '''
print("secrets")
load_secrets()

print("\\ncode")
PROJECT = get_code()
sys.path.insert(0, str(PROJECT / "src"))
os.chdir(PROJECT)
print(f"  {PROJECT}")

configure({label: PROJECT / relative for label, (relative, _) in ARTEFACTS.items()})
'''

DRIVER_READ = '''
print("code")
PROJECT = get_code()
sys.path.insert(0, str(PROJECT / "src"))
os.chdir(PROJECT)
print(f"  {PROJECT}")

print("\\nsecrets")
load_secrets(PROJECT)

FOUND, MISSING = locate(PROJECT)
if MISSING:
    raise SystemExit(
        "Notebook 1's output is not attached, and nothing is built in this notebook.\\n"
        f"  missing:  {', '.join(MISSING)}\\n"
        f"  attached: {attached_inputs() or 'nothing'}\\n\\n"
        "  Add Input > Notebook Output > NL2SQL 1 Setup\\n"
        "  {{SETUP_URL}}"
    )

configure(FOUND)
print("\\nartefacts")
show(FOUND)
'''

# GITHUB_TOKEN is not listed: `get_code` raises a precise error if a clone is
# needed and fails, and no clone is needed when the code is already local.
NEEDS = {1: (), 2: (), 3: ("GROQ_API_KEY",), 4: ("GROQ_API_KEY",)}


def setup_cell(nb: Notebook, *, build: bool) -> str:
    body = "\n".join(part.strip("\n") for part in
                     (PRELUDE, ARTEFACTS, DRIVER_BUILD if build else DRIVER_READ))
    return _fill(body, {"REQUIRED": repr(NEEDS[nb.number]),
                        "REPO": REPO,
                        "SETUP_URL": BY_NUMBER[1].url})


def opening(nb: Notebook) -> None:
    """Header, dependencies, setup. The same three cells in notebooks 2 to 4."""
    header(nb)
    md(nb, """
## 1. Setup

The code is cloned from GitHub. The database, the index and the two models are
read from [notebook 1]({{URL1}})'s saved output, where they already are. Nothing
is downloaded or rebuilt here.

Before running: **Add Input > Notebook Output > NL2SQL 1 Setup**, add the secrets
listed below under **Add-ons > Secrets**, and enable Internet.
""", URL1=BY_NUMBER[1].url)
    code(nb, INSTALL)
    code(nb, setup_cell(nb, build=False))


# A real ping is deliberately not sent here. Every outgoing message is verified
# by the egress gate, and a synthetic "are you there" would have to bypass it.
# The chain is what actually fails: with no key it is empty, the arms raise
# NoProviderAvailable in milliseconds, and the results table fills with zeros.
PREFLIGHT = """
from hybridsql.providers import cloud

targets = cloud.chain()
for target in targets:
    print(f"  {target.name}")

if not targets:
    raise SystemExit(
        "No cloud provider is configured. Three of the four architectures call one,\\n"
        "and without a key they fail instantly with no tokens and no rows.\\n"
        "Set GROQ_API_KEY under Add-ons > Secrets, attached to this notebook,\\n"
        "then run the setup cell again."
    )
print(f"\\n  {len(targets)} target(s), tried in this order")
"""


# The files a question actually passes through, in the order it passes through
# them. Everything else in the repository builds, measures or serves this.
FILE_MAP = '''
PIPELINE = [
 ("pipeline/understand.py",   "reads the question and picks out the words that matter"),
 ("providers/extractor.py",   "the small model that finds those words"),
 ("resources/glossary.py",    "knows that 'drug' means the drugname column"),
 ("db/value_index.py",        "turns 'aspirin' into the exact value stored in the database"),
 ("pipeline/anonymize.py",    "swaps every value for a symbol, :v1 and :v2"),
 ("pipeline/opaque.py",       "hides the table and column names as well"),
 ("db/schema.py",             "describes the tables to the cloud model"),
 ("pipeline/generate.py",     "writes the prompt and asks the cloud model for SQL"),
 ("security/egress_gate.py",  "checks every word before anything is sent"),
 ("providers/cloud.py",       "the only file that opens a connection to the internet"),
 ("security/sql_validator.py","refuses anything that is not a plain SELECT"),
 ("db/connection.py",         "runs the query, read-only"),
 ("providers/local_model.py", "the small model that writes the final answer"),
 ("pipeline/answer.py",       "turns the rows into a sentence"),
 ("graph/build.py",           "wires these stages into the four architectures"),
]
'''


# =================================================================================
#  1. Setup
# =================================================================================
def build_setup(nb: Notebook) -> None:
    header(nb)
    md(nb, """
Run this notebook once, then **Save Version > Save & Run All**. The other three
attach its output and download nothing.

| Built here | Size | Used for |
|---|---|---|
| `eicu.db` | 429 MB | 31 tables, 4.6 M rows, with keys and indexes rebuilt |
| `value_index.db` | 2.4 MB | turning *aspirin* into the exact stored spelling |
| GLiNER2 | 806 MB | finding the entities in a question |
| Qwen3-1.7B Q4 | 1.1 GB | writing the answer, on the CPU |

First run takes 15 to 20 minutes, almost all of it transfer.

The data is **eICU-CRD v2.0.1**, a de-identified public research database of
intensive-care records published by the MIT Laboratory for Computational
Physiology. No private data is used anywhere in this project.
""")

    md(nb, """
---

## 1. Environment

Two CPU cores, no GPU. Everything in this project is built to run on that.
""")
    code(nb, """
import os, sys, shutil
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()
free_gb = shutil.disk_usage("/kaggle/working" if ON_KAGGLE else ".").free / 1e9

print(f"  platform    {'Kaggle' if ON_KAGGLE else 'workstation'}")
print(f"  python      {sys.version.split()[0]}")
print(f"  cores       {os.cpu_count()}")
print(f"  disk free   {free_gb:.1f} GB")

assert free_gb > 6, "6 GB of free disk is required"
""")

    md(nb, """
---

## 2. The code

`hybridsql` is cloned from GitHub rather than pasted into cells, so what runs
here is what the tests were run against.

The repository is private. Add a GitHub personal access token with `repo` scope
as a secret named `GITHUB_TOKEN`, under **Add-ons > Secrets**.

```
src/hybridsql/
  db/          schema, read-only connection, value index
  pipeline/    understand > anonymize > generate > answer
  providers/   cloud (the only outbound socket), extractor, local model
  security/    egress gate, SQL validator, audit journal
  graph/       the four architectures, as state machines
  api/         the REST service
```
""")
    code(nb, setup_cell(nb, build=True))
    code(nb, """
commit = (PROJECT / "COMMIT").read_text().strip()[:8] if (PROJECT / "COMMIT").exists() else ""
if not commit:
    commit = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
modules = sorted(p.name for p in (PROJECT / "src/hybridsql").iterdir() if p.is_dir())
print(f"  commit    {commit}")
print(f"  packages  {', '.join(m for m in modules if not m.startswith('_'))}")
""")

    md(nb, """
---

## 3. Dependencies

`llama-cpp-python` compiles from source, which takes 5 to 15 minutes. The
prebuilt CPU wheel index turns that into a 20-second download.
""")
    code(nb, INSTALL)
    code(nb, """
import importlib

for name in ["gliner2", "llama_cpp", "langgraph", "langsmith", "fastapi", "sqlglot", "rapidfuzz"]:
    module = importlib.import_module(name)
    print(f"  {name:<14}{getattr(module, '__version__', 'installed')}")
""")

    md(nb, """
---

## 4. The models

Both models run **inside this process**. Neither is an API call, which is what
makes "local" something you can check rather than something claimed.

**GLiNER2** reads the question and returns the entities in it. It is zero-shot:
the entity types are described in plain English at call time, so adding a type
costs a line of text rather than a training run.

**Qwen3-1.7B Q4** writes the final answer from the rows the query returned. It is
deliberately small. The SQL came from a much larger model and the figures come
from the database, so this model only has to turn numbers into a sentence.

They are fetched with `curl -C -`, which resumes a broken transfer at the real
byte. `huggingface_hub` cannot: its temporary filename derives from the signed
CDN URL, which changes on every attempt, so each retry restarts from zero.
""")
    code(nb, """
started = time.time()
!bash scripts/download_models.sh
print(f"\\n{time.time() - started:.0f}s")
""")

    md(nb, """
---

## 5. The database

The published eICU export has **no primary keys, no foreign keys and no
indexes**. That matters here more than it would elsewhere, because the schema is
the only thing the cloud model ever sees of this database. A model told how the
tables join writes a correct query. A model left to guess guesses.

So the build reconstructs the schema:

- **Primary keys and indexes** come from the consortium's own repository
  (`MIT-LCP/eicu-code`), which defines 17 keys and 22 indexes.
- **Foreign keys** are declared here. eICU documents the relationships but
  declares none: `patientunitstayid` links 28 tables to `patient`, `hospitalid`
  links `patient` to `hospital`.
- **Composite indexes** on "join key + filtered column" for the seven tables that
  questions actually filter. Without them the rebuilt database was slower than
  the raw export. With them SQLite uses a covering index.
""")
    code(nb, """
started = time.time()
!python scripts/build_database.py
print(f"\\n{time.time() - started:.0f}s")
""")
    code(nb, """
import sqlite3

db = sqlite3.connect(os.environ["DB_PATH"])
tables = [r[0] for r in db.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
rows       = sum(db.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0] for t in tables)
keys       = sum(1 for t in tables if any(c[5] for c in db.execute('PRAGMA table_info("%s")' % t)))
foreign    = sum(len(db.execute('PRAGMA foreign_key_list("%s")' % t).fetchall()) for t in tables)
violations = db.execute("PRAGMA foreign_key_check").fetchall()
db.close()

print(f"  tables         {len(tables)}")
print(f"  rows           {rows:,}")
print(f"  primary keys   {keys}")
print(f"  foreign keys   {foreign}")
print(f"  size           {Path(os.environ['DB_PATH']).stat().st_size / 1e6:.0f} MB")
print(f"  violations     {len(violations)}")

assert not violations, "the declared foreign keys do not hold"
""")

    md(nb, """
---

## 6. The value index

### The problem it solves

A question says *aspirin*. The database stores `ASPIRIN EC 81 MG PO TBEC`. So
`WHERE drugname = 'aspirin'` returns **nothing**, and the question fails without
anyone noticing.

The index is a small second database that maps what a person writes to what is
actually stored. It is built **here, once**, and it is what makes masking possible
at all: it is the moment the system learns which exact string must never leave.

### Why not simply index every value

Because the cost would then grow with the number of rows, and a design that stops
working on a large database is not a design. So every text column is measured once
and put in one of three tiers.

| Tier | What the column holds | What is stored | Example |
|---|---|---|---|
| **A** | a bounded vocabulary | **every distinct value** | `medication.drugname`, 1 402 names |
| **B** | thousands of distinct strings | **nothing** — searched on demand | `lab.labresulttext`, 7 669 |
| **C** | identifiers, timestamps, free text | **nothing** — never searched | `patientunitstayid` |

The decision is made from measurements, not from column names: how many distinct
values, how long they are, and how close the distinct count is to the row count. A
column with almost as many distinct values as rows is an identifier whatever it is
called — that is how `patient.uniquepid`, 73 % unique, was kept out of a file meant
for business vocabulary.

**What this buys.** The stored size is `columns x limit` — it does not depend on the
number of rows. Adding ten million rows changes nothing. If a column outgrows the
limit it moves to tier B, which stores *less*. The index below stays a few megabytes
whatever this database grows into.

Notebook 2 shows a lookup running, and what happens when the scorer gets it wrong.
""")
    code(nb, """
started = time.time()
!python scripts/build_value_index.py
print(f"\\n{time.time() - started:.0f}s")
""")

    md(nb, """
---

## 7. Check, then save

`data/raw` holds the archive and the export the database was built from. Neither
is read again once `eicu.db` exists, and leaving them in place would add half a
gigabyte to every notebook that attaches this output. They are removed first.

Then **Save Version > Save & Run All**.
""")
    code(nb, """
from hybridsql.db import value_index

raw = PROJECT / "data/raw"
if raw.exists() and Path(os.environ["DB_PATH"]).exists():
    freed = sum(f.stat().st_size for f in raw.rglob("*") if f.is_file()) / 1e6
    shutil.rmtree(raw)
    print(f"  removed the build cache, {freed:.0f} MB\\n")

FINAL = {label: PROJECT / relative for label, (relative, _) in ARTEFACTS.items()}
show(FINAL)

incomplete = [label for label, (relative, probe) in ARTEFACTS.items()
              if not whole(PROJECT / relative, probe)]
assert not incomplete, f"not built: {incomplete}"

s = value_index.stats()
print(f"\\n  {s['values_indexed']:,} values indexed over {s['tiers']['A']} columns")
print(f"  {sum(size_mb(p) for p in FINAL.values()):.0f} MB will be saved as this notebook's output")
""")

    md(nb, """
---

## 8. The files a question passes through

In order. Everything else in the repository builds these, measures them, or
serves them. The line counts are read from the files that were just cloned, so
this list cannot drift away from the code.

If you only read one, read `security/egress_gate.py`.
""")
    code(nb, FILE_MAP)
    code(nb, """
print(f"  {'file':<28}{'lines':>6}   what it does")
total = 0
for name, purpose in PIPELINE:
    path = PROJECT / "src/hybridsql" / name
    n = path.read_text(encoding="utf-8", errors="replace").count(chr(10))
    total += n
    print(f"  {name:<28}{n:>6}   {purpose}")

print(f"\\n  {total:,} lines")
""")

    next_link(nb, "Save the version now. The other three notebooks read what it contains.")


# =================================================================================
#  2. Understanding
# =================================================================================
def build_understanding(nb: Notebook) -> None:
    opening(nb)

    md(nb, """
---

## 2. The problem

Someone types, in English:

> *{{Q}}*

To answer it we need SQL. Writing good SQL over 31 unfamiliar tables is something
large cloud models do well and small local models do badly. But sending the
question to a cloud model sends what we are protecting, because the question *is
about the data*: it names a drug, an age, a ward.

So the question is taken apart **here**, on this machine, before anything is sent.
Four steps, four different jobs:

| Step | The question it answers | Tool |
|---|---|---|
| 1 | which **words** matter? | GLiNER2, a small local model |
| 2 | is a word a **stored value**? | the value index |
| 3 | is a word a **column name**? | the column catalogue |
| 4 | can what is left **safely leave**? | the egress gate |

At the end, the question has become four things and none of them is a secret: the
tables, the column names, the values (kept here, replaced by symbols), and the
sentence with holes in it.
""", Q=QUESTION)

    md(nb, """
---

## 3. The database

31 tables of intensive-care records. Almost every one hangs off a single column:
`patientunitstayid` identifies one patient's stay in one ICU and appears in 28 of
them. That is what makes *which patients on aspirin had a high creatinine*
answerable — `medication` and `lab` both carry it, so they join.
""")
    code(nb, """
from hybridsql.db import schema as sch

for key, value in sch.summary().items():
    print(f"  {key:<16}{value:,}")

print("\\n" + sch.ddl({"medication"}, with_row_counts=True)[:430])
""")

    md(nb, """
The DDL above is **the entire disclosure** of the protected architecture: table
names, column names, types, row counts, foreign keys. Not one value from any row.

---

## 4. Step 1 — find the words that matter

**GLiNER2** is a 208 M-parameter model that reads a sentence and returns the spans
that carry meaning. Two properties are why it is here instead of an LLM:

- it runs **inside this process, on the CPU**, in about 100 ms — so the question
  never leaves the machine to be understood;
- it is **zero-shot**: the entity types are written in plain English at call time.
  Changing domain means editing a list of strings, not retraining anything.

It finds *where* something interesting is. It does not know this database, and it
returns `drug`, not `medication.drugname`. The next two steps do that.

Two limits, both measured rather than assumed. **The span is a guess**: it returns
`hemoglobin lab test` where only `hemoglobin` is stored, so shorter spans are tried
against the index rather than trusting the boundaries. And **it does not see numbers
as data**: `hospital id 56` came back as one entity and `56` went out in clear text,
so numbers are found by a separate regular expression, which cannot miss one.
""")
    code(nb, """
from hybridsql.providers import extractor

print(f"  model  {extractor.state()['model']}   loaded in {extractor.state()['load_ms']} ms")
print(f"  types  {', '.join(extractor.ENTITY_TYPES[:4])}, +{len(extractor.ENTITY_TYPES) - 4} more\\n")

for question in ["{{Q}}", "Did Mr. Bensalah get his insulin?"]:
    print(f"  {question}")
    for e in extractor.extract(question):
        print(f"      {e.text:<24}{e.type:<12}{e.score:.2f}")
    print()
""", Q=QUESTION)

    md(nb, """
---

## 5. Step 2 — is the word a stored value?

The analyst writes *aspirin*. The database stores `ASPIRIN EC 81 MG PO TBEC`. A
query written `WHERE drugname = 'aspirin'` returns **nothing**. An index built once
by notebook 1 closes that gap locally — and it is the same step that tells us which
exact string must never leave.

### Why not index every value

Because then the cost grows with the number of rows, and a privacy design that
stops working on a big database is not a design. Every text column is measured once
and sorted into three tiers.

| Tier | The column looks like | Stored |
|---|---|---|
| **A** | a bounded vocabulary — 1 402 drug names, 12 ward types | every distinct value |
| **B** | thousands of distinct strings | **nothing** — searched on demand |
| **C** | identifiers, timestamps, free text | **nothing** — never searched |

**That is the whole scalability argument.** Adding ten million rows adds nothing to
the index. If a column outgrows the limit it moves to tier B and stores *less*.
""")
    code(nb, """
from hybridsql.db import value_index

s = value_index.stats()
total = sum(s["tiers"].values())
print(f"  text columns examined  {total}")
print(f"  tier A / B / C         {s['tiers']['A']} indexed, {s['tiers'].get('B', 0)} on demand, "
      f"{s['tiers'].get('C', 0)} excluded")
print(f"  values stored          {s['values_indexed']:,}")
print(f"  index size             {s['size_mb']} MB")
print(f"  ceiling, any row count {total * value_index.VOCABULARY_LIMIT * 83 / 1e6:.0f} MB\\n")

for mention in ["aspirin", "asspirin"]:
    hit = value_index.search(mention, limit=1)[0]
    print(f"  {mention:<12}-> {hit.ref:<26}{hit.value!r}  ({hit.score:.2f})")
""")

    md(nb, """
### The scorer, and the bug that lived in it

Similarity here is **not symmetric**, and treating it as if it were was the worst
defect this pipeline has had.

The legitimate case is the analyst naming *part* of a longer value: "aspirin" for
`ASPIRIN EC 81 MG PO TBEC`. The reverse — a short stored value found inside a long
question — is not a match at all. Measured, before the fix:

| The question said | What the index answered | Score |
|---|---|---|
| `10 most frequently recorded laboratory tests` | `10` | **1.00** |
| `laboratory records` | `lab` | **1.00** |
| `male patients` | `Older adult: Provide adequate time for patients…` | **0.76** |

All three cleared the confidence threshold, so all three were masked and sent to
the cloud model as filters. The queries ran and answered a different question.

Two rules fixed it: when the value is **shorter** than the mention only whole-string
similarity counts, and a compound value (`renal|acute renal failure|…`) is compared
**segment by segment** instead of as one long string that shares a word with
everything.
""")
    code(nb, """
print(f"  {'the question says':<44}{'the index now answers':<40}score")
for mention in ["10 most frequently recorded laboratory tests", "laboratory records",
                "male patients", "aspirin"]:
    hits = value_index.search(mention, limit=1)
    answer = f"{hits[0].ref} = {hits[0].value}"[:38] if hits else "nothing, which is correct"
    print(f"  {mention:<44}{answer:<40}{hits[0].score:.2f}" if hits
          else f"  {mention:<44}{answer:<40}")
""")

    md(nb, """
---

## 6. Step 3 — is the word a column name?

The index answers *"which value is this?"*. It always answers something: a fuzzy
search over 30 000 values never comes back empty. So we also need the other
question — *"which column is this?"* — or every question that names a column gets
turned into a filter on an invented value. That is exactly what happened:

> *"What are the 10 most common **diagnosis names**?"*
> → matched `pasthistory.pasthistoryvalue = 'clinical diagnosis'` at 0.75, masked it,
> asked the cloud to filter on it, and came back with **one row**.

Column names in a real database are glued-together words — `labname`, `routeadmin`,
`nursingchartcelltypecat`. Splitting them would need a dictionary of the domain we
are trying to stay independent of, so matching runs on **character trigrams**
instead: `administration routes` and `medication.routeadmin` share `rou`, `out`,
`ute`, `adm`, `dmi`. No dictionary, no model, 2 ms, and it works on any schema.
""")
    code(nb, """
from hybridsql.db import catalog

print(f"  {catalog.stats()['columns_catalogued']} columns catalogued\\n")
print(f"  {'the question says':<34}the columns it could mean")
for mention in ["diagnosis names", "medication administration routes",
                "hospital region", "ethnicity", "aspirin", "sepsis"]:
    shown = "  ".join(f"{m.ref} ({m.score:.2f})" for m in catalog.link(mention, limit=2))
    print(f"  {mention:<34}{shown or 'nothing — so it is not a column'}")
""")

    md(nb, """
`aspirin` and `sepsis` are real values, and they match **no column at all**. That
gap is what the decision below runs on.

---

## 7. Step 4 — decide

One rule:

> A word is a **value** only if it matches a value better than it matches a column.
> A word stored in the database **exactly as typed** is never overruled.

The second clause is not a detail: `creatinine`, `glucose` and `albumin` are lab
values in `lab.labname` **and** column names in `apacheapsvar`.

Numbers skip all of this and are always masked. "Over 65" is a threshold the
analyst chose, but "hospital id 56" identifies one hospital, and the sentence does
not reliably tell them apart. The model never needs a number's value to write
`WHERE hospitalid = :v1`, so masking every one costs nothing.
""")
    code(nb, """
from hybridsql.pipeline.understand import understand

for question in ["{{Q}}",
                 "What are the 10 most common diagnosis names?",
                 "How many patients in hospital id 56?"]:
    r = understand(question)
    print(f"  {question}")
    for x in r.resolutions:
        kept = f"   kept here: {x.value!r}" if x.value else ""
        print(f"      {x.mention:<22}{x.kind:<10}{str(x.column or '-'):<32}{kept}")
    print()
""", Q=QUESTION)

    md(nb, """
Three kinds come out. **value** exists in the database, so it is a secret and gets
masked. **concept** names a column — not a secret, the column name is in the DDL
anyway, and masking it would leave the model unable to write the query. **quantity**
is a number, masked.

---

## 8. Masking

Each value and each number becomes a symbol. The mapping stays in this process, and
the symbols are **renumbered on every request** — `:v1` here and `:v1` in the next
question are unrelated, so nobody watching the traffic can follow a value across two
questions.
""")
    code(nb, """
from hybridsql.pipeline.anonymize import anonymize

u = understand("{{Q}}")
a = anonymize(u)

print(f"  asked   {u.question}")
print(f"  sent    {a.masked_question}\\n")
print("  kept here, and nowhere else")
for symbol, value in a.mapping.items():
    print(f"    {symbol:<6}{value!r:<34}{a.columns.get(symbol, '')}")
""", Q=QUESTION)

    # --- The egress gate ------------------------------------------------------
    md(nb, """
---

## 9. The egress gate

Masking is the design. **The gate is what makes it checkable.**

> One rule: exactly one module may open a socket, and every piece of text it sends
> is verified first.

Not "should be" — the gate raises an exception, so an unverified send does not
happen.

### Checking by origin, not by vocabulary

The first version read the whole prompt word by word. It worked, and it forced
hundreds of ordinary English words into a hand-written list so that **our own
sentences** could pass. Every new question added one more word, and each addition
widened the hole.

The prompt is now split by where each part came from, and each part is checked by
the rule that can actually prove *that kind* of text safe. Three of the four are
reconstructions: the gate recomputes what the part should contain and compares.

| Origin | Proven safe by | Can it hide a value? |
|---|---|---|
| `authored` | SHA-256 of a literal written in the source | no, it is a constant |
| `template` | SHA-256 of the wording, labels blanked out | no, only `t3`, `c7`, numbers vary |
| `schema` | regenerating the DDL from the database and comparing | no, the DDL holds no row |
| `glossary` | membership among the declared notes | no, a modified note is not a note |
| `question` | **word by word, fail-closed** | yes — the only untrusted part |
""")
    code(nb, """
from hybridsql.pipeline import generate as gen
from hybridsql.security import egress_gate

print(f"  {'verdict':<8}{'origin':<11}{'proven by':<30}text")
for segment in gen.build_segments(u, a):
    verdict = egress_gate.check_segment(segment, "notebook")
    print(f"  [{'pass' if verdict.allowed else 'BLOCK':<4}] {segment.origin:<11}"
          f"{verdict.verified_by:<30}{' '.join(segment.text.split())[:32]}...")
""")

    md(nb, """
### The question: two layers, and what they are made of

**Layer 1, the exhaustive denylist.** We *own the database*, so we hold the complete
inventory of values for every indexed column. Matching against it is not an estimate,
it is exact — a value present in the database cannot slip past. This is the
difference from PII detection, which is a heuristic with a recall (MaskSQL measures
61.4 % on this task, so four values in ten leave in clear text).

**Layer 2, the allowlist, fail-closed.** For what the denylist cannot know: values of
non-indexed columns, misspellings, words nobody declared. A word that survives layer
1 is refused if it is a word of a short stored value.

Three sources, and only the first is written by hand. That proportion is the point:
the vocabulary a question may contain is mostly **read from the database itself**, so
it cannot drift away from it.
""")
    code(nb, """
import random

print(f"  {'text':<54}verdict")
for text in ["How many patients over :v2 received :v1?",
             "How many patients received aspirin?",
             "How many patients came from Skilled Nursing Facility?",
             "SELECT COUNT(*) FROM medication WHERE drugname = :v1"]:
    v = egress_gate.check(text, "notebook")
    print(f"  {text[:52]:<54}"
          f"{'passed' if v.allowed else 'REFUSED on ' + str(list(v.refused_tokens))}")

grammar = egress_gate.generic_vocabulary()
derived = egress_gate._schema_identifiers() | egress_gate.glossary_concepts()

print(f"  {'question grammar (written by hand)':<40}{len(grammar):>7}")
print(f"  {'schema identifiers (read from the db)':<40}{len(egress_gate._schema_identifiers()):>7}")
print(f"  {'glossary concepts (read from yaml)':<40}{len(egress_gate.glossary_concepts()):>7}")
print(f"  {'total allowed':<40}{len(egress_gate.allowlist()):>7}   "
      f"{100 * len(derived) / max(len(egress_gate.allowlist()), 1):.0f} % derived")
print(f"  {'words of stored values (blocked)':<40}{len(egress_gate.value_tokens()):>7}")
print(f"  {'complete values (blocked)':<40}{len(egress_gate.known_values()):>7}")

random.seed(7)
blocked = sorted(w for w in egress_gate.value_tokens() if w.isalpha() and len(w) > 4)
print("\\n  a sample of the words a question may NOT contain")
for _ in range(3):
    print("    " + "  ".join(f"{w:<18}" for w in random.sample(blocked, 4)))
""")

    md(nb, """
### Refusing too much is a bug as well

A gate has two ways of being wrong. It can let a value through — and it can refuse a
question that carries nothing to hide, which teaches the analyst to reword until
something passes. Only the first was ever measured, and the second was happening
constantly:

> *"How many laboratory records are **associated** with each patient ICU stay?"*
> → refused on `associated`.

`associated` is not data. In this database it appears only inside long hierarchical
diagnosis strings such as `hematology|coagulation disorders|DIC syndrome|associated
with intravascular clotting`. Every database's free text contains it.

Two changes, both rules rather than patches:

1. the blocked vocabulary now takes the words of **short values only** (three words
   or fewer) — a threshold already measured and used elsewhere in the same file, but
   never applied here. 8 708 words → 7 419. `aspirin`, `female`, `alive` and `hgb`
   stay; `associated`, `unique` and `necrotizing` go. Long values are still caught
   **whole** by layer 1;
2. the closed-class grammar was extended with the vocabulary of *asking a database a
   question* — `records`, `unique`, `names`, `one`. The same words serve a hospital,
   a bank or a warehouse; none can name a drug. A test fails the build if a word
   added there is ever also a whole stored value.
""")
    code(nb, """
print("  words that used to be refused in an ordinary question")
for word in ["associated", "records", "unique", "one", "administration"]:
    v = egress_gate.check(f"how many {word} per hospital", "notebook")
    print(f"    {word:<16}{'passes now' if v.allowed else 'still refused'}")

print("\\n  and the values they might have hidden are still refused")
for value in ["Medical Records", "aspirin", "Female", "Skilled Nursing Facility"]:
    v = egress_gate.check(f"how many patients with {value}", "notebook")
    print(f"    {value:<26}{'refused, correct' if not v.allowed else 'PASSED — a leak'}")
""")

    md(nb, """
### Both rates, measured

`scripts/measure_gate.py` replays every distinct value of the database through the
gate, and every question of the three evaluation sets through the whole pipeline.
Neither number means anything without the other.
""")
    code(nb, """
measurement = Path(os.environ["DB_PATH"]).with_name("gate_measurement.json")
if measurement.exists():
    g = json.loads(measurement.read_text())
    q = g.get("questions", {})
    print(f"  {'information-bearing values tested':<44}{g['values_bearing']:>8,}")
    print(f"  {'LEAK: how many cross the gate':<44}{g['passed_bearing']:>8,}"
          f"  {g['rate_bearing_pct']:>6.2f} %")
    print(f"  {'questions replayed end to end':<44}{q.get('questions_checked', 0):>8}")
    print(f"  {'REFUSAL: ordinary questions blocked':<44}{q.get('questions_refused', 0):>8}"
          f"  {q.get('refusal_rate_pct', 0):>6.2f} %")
else:
    print("  run scripts/measure_gate.py to produce this file")
""")

    md(nb, """
The residual leak is explainable rather than mysterious: it is the words that are
**both a column name and a value**. `apacheapsvar` has columns `albumin`,
`creatinine`, `bun` and `wbc`, which are also lab names in `lab.labname`. Blocking
them would stop the model writing SQL against those columns.

### The proof: a real value, submitted on purpose

The gate is only worth something if it refuses.
""")
    code(nb, """
from hybridsql.security.egress_gate import LeakBlocked, Segment

try:
    egress_gate.require_segments(
        [Segment("How many patients received AMOXICILLIN 500 MG PO CAPS?", "question")],
        "check")
    print("  ALLOWED. This would be a failure.")
except LeakBlocked as blocked:
    print(f"  refused: {blocked}")
""")

    md(nb, """
---

## 10. What is measured, and what would leave

Three annotated question sets, 133 questions. `complete` is the strict figure: every
value, every column and every name right in the same question — the only one that
describes what the user sees.

- **standard**, 79 questions naming something stored;
- **hard**, 28 written to break it — typos, slang, person names, values absent from
  this extract;
- **analytics**, 26 that name a **column** and aggregate over it, which is what an
  analyst actually types. Added the day the system answered *"the 10 most common
  diagnosis names"* with one row.
""")
    code(nb, """
results = PROJECT / "data/evaluation/understanding_results.json"
if results.exists():
    r = json.loads(results.read_text())
    print(f"  {'set':<12}{'questions':>10}{'extraction':>12}{'resolution':>12}{'complete':>10}")
    for name in ("standard", "hard", "analytics"):
        s = r.get(name)
        if s:
            print(f"  {name:<12}{s['questions']:>10}{s['extraction_recall_pct']:>11.1f}%"
                  f"{s['resolution_accuracy_pct']:>11.1f}%{s['full_understanding_pct']:>9.1f}%")
    o = r["overall"]
    print(f"\\n  {o['correct_questions']}/{o['questions']} questions understood completely "
          f"({o['full_understanding_pct']} %)")

print(f"\\n  the question    {u.question}")
print(f"  what is sent    {a.masked_question}")
print(f"  values out      0")
print(f"  kept here       {len(a.mapping)} value(s), {len(u.tables)} table name(s)")
""")

    next_link(nb, "Four architectures do different things with this. Three send it, "
                  "one does not, and the difference is measurable.")


# =================================================================================
#  3. Architectures
# =================================================================================
ASK = '''
from hybridsql.graph import ARMS, run
from hybridsql.graph.state import public

QUESTIONS = [
    "How many patients received aspirin?",
    "What is the average age of patients admitted to the MICU?",
    "How many female patients were discharged alive?",
]

results = []


def ask(question, arm, write=False):
    """Run one question through one architecture and print every stage of it."""
    r = public(run(question, arm=arm, write=write))
    results.append(r)

    print(f"  question    {question}")
    if r["masked_question"]:
        print(f"  sent        {r['masked_question']}")
    if r["opaque"].get("question"):
        print(f"  relabelled  {r['opaque']['question']}")
    if r["sql"]:
        print(f"  sql         {' '.join(r['sql'].split())}")
    if r["success"]:
        print(f"  rows        {r['row_count']}")
    else:
        print(f"  failed      {r['failed_stage']}: {r['failure_reason']}")
    if r["answer"]:
        print(f"  answer      {' '.join(r['answer'].split())[:180]}")
    print(f"  cost        {sum(r['ms'].values()):.0f} ms, {r['cloud_tokens']} cloud tokens, "
          f"{r['egress_chars']} chars sent, {r['egress_values']} values sent\\n")
    return r
'''

DIAGRAMS = '''
from hybridsql import graph
from IPython.display import Image, Markdown, display


def diagram(arm):
    """Draw the compiled graph of one architecture.

    A notebook renders markdown, not mermaid, so the source that `graph.mermaid()`
    returns arrives as a block of text rather than as a picture. It is drawn to a
    PNG instead. If that service cannot be reached the stages are printed: the
    diagram is worth having, not worth stopping the notebook for.
    """
    drawing = graph.compiled(arm).get_graph()
    display(Markdown(f"**{arm}**"))
    try:
        display(Image(drawing.draw_mermaid_png(max_retries=3, retry_delay=2.0)))
    except Exception as error:
        stages = [name for name in drawing.nodes if not name.startswith("__")]
        print(f"  no diagram ({type(error).__name__}), the stages are")
        print("  start -> " + " -> ".join(stages) + " -> end")


for arm in ARMS:
    diagram(arm)
'''

COMPARISON = """
from collections import defaultdict

rows = defaultdict(lambda: {"n": 0, "ok": 0, "ms": 0.0, "values": 0, "chars": 0, "tokens": 0})
for r in results:
    e = rows[r["arm"]]
    e["n"] += 1
    e["ok"] += int(r["success"])
    e["ms"] += sum(r["ms"].values())
    e["values"] += r["egress_values"]
    e["chars"] += r["egress_chars"]
    e["tokens"] += r["cloud_tokens"]

head = f"{'architecture':<16}{'ran':>7}{'avg ms':>9}{'values sent':>13}{'chars sent':>12}{'tokens':>8}"
print(head)
print("-" * len(head))
for arm in ARMS:
    e = rows[arm]
    if e["n"]:
        print(f"{arm:<16}{str(e['ok']) + '/' + str(e['n']):>7}{e['ms'] / e['n']:>9.0f}"
              f"{e['values']:>13}{e['chars']:>12}{e['tokens']:>8}")

failed = [r for r in results if not r["success"]]
if failed:
    print("\\nwhat failed")
    for r in failed:
        print(f"  {r['arm']:<16}{r['failed_stage']}: {r['failure_reason'][:70]}")
"""


def build_architectures(nb: Notebook) -> None:
    opening(nb)

    md(nb, """
---

## 2. The cloud provider

Three of the four architectures call a cloud model. This checks one answers before
running twelve questions against it, so a missing key shows up here as one line
rather than as four empty rows in the results table.
""")
    code(nb, PREFLIGHT)

    md(nb, """
---

## 3. The four designs

Notebook 2 turned the question into a schema, some symbols and an exact value, with
nothing sent. The question now is **what may leave**, and there is more than one
defensible answer. Four are built, and they are compared on the same questions.

| | Who writes the SQL | Who writes the answer | What leaves | Real values sent |
|---|---|---|---|---|
| **Hybrid** | cloud, from symbols | local | schema + a sentence with holes | **0** |
| **Hybrid Opaque** | cloud, from labels | local | labels only, no business word | **0** |
| **Full Cloud** | cloud, raw question | cloud | the question and every row | all |
| **Full Local** | local 1.7 B model | local | **nothing** | 0 |

Full Cloud is the baseline everybody ships. Full Local is the other extreme. The two
in the middle are the point of the project, and the table at the end says what each
costs.

They are built as state machines that share every stage they have in common, so a
fix to execution is a fix in all four. The diagrams are generated from the compiled
graphs, which means they cannot describe an architecture that is not the one
running.
""")
    code(nb, ASK)
    code(nb, DIAGRAMS)

    # --- The symbols ---------------------------------------------------------
    md(nb, """
---

## 4. The symbols: `:v1`, `t1`, `c1`

Everything below rests on three kinds of substitution. They are simple, and it is
worth being exact about which one hides what.

### `:v1` — a value

A value is never written into the query text. It becomes `:v1`, and the real string
is bound at execution time by the SQLite driver:

```
      asked   How many patients received aspirin?
       sent   How many patients received :v1?
    written   SELECT COUNT(*) FROM medication WHERE drugname = :v1
   executed   the driver binds :v1 = 'ASPIRIN EC 81 MG PO TBEC'
```

The query text and the value **never meet in a single string**. That is also why
SQL injection is impossible here: the value is not part of the statement.

The numbering is redrawn on every request, so `:v1` in one question and `:v1` in the
next are unrelated. Nobody watching the traffic can follow a value across two
questions.

### `t1`, `c1` — a table and a column

Hybrid still sends the schema, and 31 table names with 391 column names describe a
business even when no row does. `medication` says this is a hospital;
`drugname` says which column holds the drugs.

Hybrid Opaque replaces those too. `medication` becomes `t3`, `drugname` becomes
`c7`, and — like the values — **the labels are drawn again on every request**, so
the same table is a different label next time.

```
      real   SELECT COUNT(*) FROM medication WHERE drugname = :v1
     model   SELECT COUNT(*) FROM t3        WHERE c7       = :v1
```

The model receives the second line, and the answer is translated back here.

### Why the model can still write the query

Stripping schema names normally wrecks text-to-SQL, because the names carry the
meaning. It works here because **the meaning was already resolved locally**: the
index has established which column the value belongs to, so the prompt can state
`:v1 is a value of c7` and leave the model a mechanical job — follow the foreign
keys, place the aggregate.
""")

    # --- Hybrid --------------------------------------------------------------
    md(nb, """
---

## 5. Hybrid

The cloud model gets the schema and a sentence with holes in it. It writes
`WHERE drugname = :v1`. The value is bound here, so nothing about it leaves.

**Modules:** `pipeline/understand.py`, `pipeline/anonymize.py`,
`pipeline/generate.py`, `security/egress_gate.py`, `providers/cloud.py`,
`db/connection.py`, `pipeline/answer.py`

The cell below prints **the whole prompt that leaves**, verbatim. There is nothing
else in it.
""")
    code(nb, """
from hybridsql.pipeline.understand import understand
from hybridsql.pipeline.anonymize import anonymize
from hybridsql.pipeline import generate as gen

u = understand(QUESTIONS[0])
a = anonymize(u)

for message in gen.build_messages(u, a):
    print(f"--- {message['role']} " + "-" * 56)
    print(message["content"][:850])
    print()
print(f"kept here: {a.mapping}")
""")

    md(nb, "**Running it**, on three questions.")
    code(nb, 'for q in QUESTIONS:\n    ask(q, "hybrid")')

    md(nb, """
### The answer is written here

The rows never left. Turning them into a sentence is the local model's job, and that
is what lets the results of a protected architecture stay on this machine. The first
call loads 1.1 GB of weights, so it is slow once and quick afterwards.
""")
    code(nb, """
written = public(run(QUESTIONS[0], arm="hybrid", write=True))

print(f"  question           {QUESTIONS[0]}")
print(f"  sql written by     {written['sql_author']}")
print(f"  rows               {written['row_count']}, none of which left this process")
print(f"  answer written by  {written['answer_author']}")
print(f"\\n  {written['answer']}")
""")

    # --- Hybrid Opaque -------------------------------------------------------
    md(nb, """
---

## 6. Hybrid Opaque

Same pipeline, one stage added: the schema names are replaced by labels before the
prompt is built, and the SQL that comes back is translated to the real names here.

**Modules:** Hybrid's, plus `pipeline/opaque.py`
""")
    code(nb, """
view = gen.build_opaque(u, a).view()

print(f"  asked       {QUESTIONS[0]}")
print(f"  hybrid      {a.masked_question}")
print(f"  opaque      {view['question']}")
print(f"\\n  {view['parameters'].strip()}")
print(f"\\n  schema sent: {view['tables']} tables, {view['columns']} columns, every name a label")
print(view["ddl"][:340])

print("\\n  the dictionary that stays here — read it right to left:")
for alias, real in view["labels"].items():
    print(f"    {alias:<6}{real}")
""")

    md(nb, "**Running it.** The line to watch is `relabelled`: that is what the model received.")
    code(nb, 'for q in QUESTIONS:\n    ask(q, "hybrid_opaque")')

    # --- Full Cloud ----------------------------------------------------------
    md(nb, """
---

## 7. Full Cloud

The baseline. There is no masking stage, and its absence *is* the architecture: the
question leaves as typed, the model writes the SQL, and the rows are then sent back
to the model so it can write the answer. Every cell of every row crosses the network.

It is here to be measured, not recommended. The egress gate is bypassed explicitly,
`PRIVACY_MODE=strict` refuses that outright, and the audit journal records every
bypass.

**Modules:** `pipeline/generate.py`, `providers/cloud.py`, `db/connection.py`
""")
    code(nb, """
for q in QUESTIONS:
    ask(q, "full_cloud")

from hybridsql.security import audit

print("  the journal — every bypass is written down")
for key, value in audit.leak_rate().items():
    print(f"    {key:<20}{value}")
lines = audit.read()
print(f"    {'bypassed':<20}{sum(1 for line in lines if line.get('bypassed'))}")
""")

    # --- Full Local ----------------------------------------------------------
    md(nb, """
---

## 8. Full Local

Sends nothing at all. The 1.7 B model writes the SQL itself, on two CPU cores, from
the same schema the cloud model would have received.

It is the other end of the range, and it is what makes the two middle architectures
worth building: if a 1.7 B model wrote SQL as well as a 120 B one, there would be
nothing to protect and nothing to argue about.

**Modules:** `pipeline/understand.py`, `providers/local_model.py`,
`db/connection.py`, `pipeline/answer.py`
""")
    code(nb, 'for q in QUESTIONS:\n    ask(q, "full_local")')

    # --- Comparison ----------------------------------------------------------
    md(nb, """
---

## 9. The comparison

`ran` counts queries that executed, not answers that were right — accuracy is a
separate evaluation, in `scripts/evaluate_pipeline.py`. The column that decides is
**values sent**.
""")
    code(nb, COMPARISON)

    md(nb, """
Hybrid and Hybrid Opaque send **zero** values. Full Cloud sends every one it touches.
Full Local sends nothing at all and pays for it in the `ran` column.
""")

    next_link(nb, "The same pipeline in one run, then the live service.")


# =================================================================================
#  4. Run All
# =================================================================================
def build_run_all(nb: Notebook) -> None:
    opening(nb)

    md(nb, """
---

## 2. The cloud provider
""")
    code(nb, PREFLIGHT)

    md(nb, """
---

## 3. The four architectures

Three questions through each of the four. [Notebook 2]({{URL2}}) takes the
understanding stage apart, [notebook 3]({{URL3}}) the four designs; this one just
runs them. Free Groq allows 30 requests a minute, so the loop paces itself.
""", URL2=BY_NUMBER[2].url, URL3=BY_NUMBER[3].url)
    code(nb, ASK)
    code(nb, """
for i, question in enumerate(QUESTIONS, 1):
    print(f"[{i}/{len(QUESTIONS)}] {question}\\n")
    for arm in ARMS:
        ask(question, arm)
    time.sleep(2.5)
""")

    md(nb, "**What Hybrid Opaque actually sent.** Labels only, redrawn for this request.")
    code(nb, """
opaque = next((r["opaque"] for r in results
               if r["arm"] == "hybrid_opaque" and r["opaque"].get("question")), None)
if opaque:
    print(f"  relabelled  {opaque['question']}")
    print(f"  parameters  {opaque['parameters'].strip()}")
    print(f"  schema      {opaque['tables']} tables, {opaque['columns']} columns, all labels")
    print("\\n  what the labels meant, kept here:")
    for alias, real in opaque["labels"].items():
        print(f"    {alias:<6}{real}")
""")

    md(nb, """
### The answer, written here

The weights are already loaded from the Full Local runs, so this is the writing
step on its own: rows turned into a sentence without leaving the machine.
""")
    code(nb, """
written = public(run(QUESTIONS[0], arm="hybrid", write=True))
print(f"  sql written by     {written['sql_author']}")
print(f"  answer written by  {written['answer_author']}")
print(f"\\n  {written['answer']}")
""")

    md(nb, """
---

## 5. The comparison

`ran` counts queries that executed. The column that decides is **values sent**.
""")
    code(nb, COMPARISON)

    md(nb, """
---

## 5. The live service

The same pipeline, behind an address a browser can reach.

The interface is served by a Cloudflare Worker at **{{WORKER}}**. A Kaggle session
has no stable address: `cloudflared` gives this notebook a public hostname, but a
new one every restart. So the notebook publishes its current address to the
Worker, and the page reads it back. That is the only thing the Worker does.

**It does not carry the conversation.** Your browser connects straight to this
session, and the question, the SQL and the answer never pass through Cloudflare.
Proxying would have been easier, one origin and no CORS, but this project's claim
is that the data stays inside a known boundary, and routing every answer through
a third party to tidy up a URL would have contradicted it.

| Endpoint, served here | Purpose |
|---|---|
| `GET /health` | component status |
| `GET /meta` | schema summary, available architectures |
| `POST /ask` | one question, one answer |
| `POST /ask/stream` | one event per stage |
| `POST /compare` | the same question through all four |
| `GET /egress/report` | the audit journal |
""", WORKER=WORKER)

    md(nb, "**Start the API and the tunnel.**")
    code(nb, """
import threading, uvicorn, httpx
from hybridsql.api.app import app

WORKER = "{{WORKER}}"

threading.Thread(
    target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning"),
    daemon=True,
).start()

for _ in range(60):
    try:
        if httpx.get("http://127.0.0.1:8000/health", timeout=2).status_code == 200:
            print("  api       listening on 8000")
            break
    except Exception:
        time.sleep(0.5)
else:
    raise SystemExit("the API did not start")

if not Path("cloudflared").exists():
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    !chmod +x cloudflared

log = Path("tunnel.log")
log.unlink(missing_ok=True)
tunnel = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
    stdout=log.open("w"), stderr=subprocess.STDOUT,
)

public_url = ""
for _ in range(60):
    time.sleep(1)
    if log.exists():
        match = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", log.read_text())
        if match:
            public_url = match.group(0)
            break
assert public_url, "the tunnel did not start, see tunnel.log"
print(f"  tunnel    {public_url}")
""", WORKER=WORKER)

    md(nb, """
**Publish the address, then check it.** Three things have to hold, and each fails
differently, so each is checked and named.
""")
    code(nb, """
ok = True
token = os.environ.get("PUBLISH_TOKEN", "")

if not token:
    ok = False
    print("  announce   skipped, PUBLISH_TOKEN is not set")
    print("             It must be the same value in two places: a Kaggle secret")
    print("             on this notebook, and `wrangler secret put PUBLISH_TOKEN`")
    print("             in deploy/worker.")
else:
    r = httpx.post(f"{WORKER}/api/backend",
                   headers={"Authorization": f"Bearer {token}"},
                   json={"url": public_url, "label": "kaggle run-all"}, timeout=20)
    if r.status_code == 200:
        print(f"  announce   ok, {r.json().get('url')}")
    elif r.status_code == 401:
        ok = False
        print("  announce   refused (401): this notebook's PUBLISH_TOKEN and the")
        print("             Worker's are different values")
    else:
        ok = False
        print(f"  announce   failed, HTTP {r.status_code} {r.text.strip()[:100]}")

try:
    entry = httpx.get(f"{WORKER}/api/backend", timeout=20).json()
except Exception as error:
    ok, entry = False, {}
    print(f"  rendezvous unreachable, {type(error).__name__}")

if entry.get("online") and entry.get("url", "").rstrip("/") == public_url.rstrip("/"):
    print(f"  rendezvous ok, the page will be sent to {entry['url']}")
elif entry.get("online"):
    ok = False
    print(f"  rendezvous stale, it points at {entry.get('url')}")
elif entry:
    ok = False
    print(f"  rendezvous offline, {entry.get('reason', '')}")

try:
    health = httpx.get(f"{public_url}/health", timeout=30).json()
    print(f"  round trip ok, status {health.get('status')}, "
          f"database {'present' if health.get('database') else 'MISSING'}")
except Exception as error:
    ok = False
    print(f"  round trip failed, {type(error).__name__}")

print(f"\\n  interface  {WORKER}" if ok else f"\\n  the pipeline works, the public link does not")
print(f"  api docs   {public_url}/docs")
""")

    md(nb, """
### Leave this cell running

It holds the session open. Interrupt it to stop.
""")
    code(nb, """
print("Serving. Interrupt this cell to stop.\\n")
started = time.time()
try:
    while True:
        time.sleep(60)
        alive = tunnel.poll() is None
        print(f"  {(time.time() - started) / 60:5.0f} min   tunnel {'up' if alive else 'DOWN'}",
              flush=True)
        if not alive:
            print("  the tunnel stopped, re-run the cell above")
            break
except KeyboardInterrupt:
    tunnel.terminate()
    print("stopped")
""")


BUILDERS = {1: build_setup, 2: build_understanding,
            3: build_architectures, 4: build_run_all}


# =================================================================================
#  Write and push
# =================================================================================
def write(nb: Notebook) -> Path:
    BUILDERS[nb.number](nb)

    document = {
        "cells": nb.cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
            "kaggle": {"accelerator": "none", "dataSources": [], "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb.directory.mkdir(parents=True, exist_ok=True)
    path = nb.directory / f"{nb.slug}.ipynb"
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")

    # No dataset source: the code comes from GitHub. Notebooks 2 to 4 declare
    # notebook 1 so Kaggle mounts its output without anyone clicking anything.
    (nb.directory / "kernel-metadata.json").write_text(json.dumps({
        "id": nb.kaggle_id,
        "title": nb.title,
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [] if nb.number == 1 else [BY_NUMBER[1].kaggle_id],
    }, indent=2) + "\n", encoding="utf-8")
    return path


def push(nb: Notebook) -> int:
    print(f"pushing {nb.kaggle_id}")
    return subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(nb.directory)]
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Kaggle notebooks.")
    parser.add_argument("only", nargs="*", type=int, help="notebook numbers, default all")
    parser.add_argument("--push", action="store_true", help="upload to Kaggle as well")
    arguments = parser.parse_args()

    chosen = [BY_NUMBER[n] for n in (arguments.only or sorted(BY_NUMBER))]
    for nb in chosen:
        path = write(nb)
        n_code = sum(1 for c in nb.cells if c["cell_type"] == "code")
        print(f"{path.relative_to(ROOT)}  {len(nb.cells)} cells "
              f"({len(nb.cells) - n_code} markdown, {n_code} code)")

    if arguments.push:
        print()
        failures = [nb.slug for nb in chosen if push(nb)]
        if failures:
            print(f"failed: {', '.join(failures)}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
