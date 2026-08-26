"""Generate the five Kaggle notebooks.

    python notebooks/build.py              write them
    python notebooks/build.py --push       write and upload to Kaggle
    python notebooks/build.py --push 3     only notebook 3

A `.ipynb` is JSON with every line of source escaped into a string array. Edited
by hand they rot: cells drift, stale outputs stick to changed code, and the diffs
are unreadable. Here they have one source file and are rebuilt from it.

    1  Setup           clone, install, build the database and the index
    2  Understanding   how a question is read, before anything is sent
    3  Architectures   the four designs, run and measured
    4  Optimization    five variants of the hybrid arm, benchmarked and ranked
    5  Run             the whole pipeline, then the live service

Notebook 1 is the only one that builds anything. The others install the package,
then read notebook 1's saved output through `/kaggle/input` — the database, the
index and the weights are opened where they are mounted rather than copied — and
stop with instructions if that output is missing.
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

FONT = "-apple-system,BlinkMacSystemFont,Segoe UI,Inter,sans-serif"


class Notebook:
    def __init__(self, number: int, slug: str, name: str, colour: str, subtitle: str) -> None:
        self.number, self.slug, self.name = number, slug, name
        self.colour, self.subtitle = colour, subtitle
        self.cells: list[dict] = []

    @property
    def directory(self) -> Path:
        return OUT / f"{self.number}-{self.slug.split('-', 2)[-1]}"

    @property
    def kaggle_id(self) -> str:
        return f"{OWNER}/{self.slug}"

    @property
    def title(self) -> str:
        return f"NL2SQL {self.number} {self.name}"

    # -- cells ---------------------------------------------------------------
    def md(self, text: str) -> None:
        self.cells.append({
            "cell_type": "markdown",
            "id": f"md{len(self.cells)}",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        })

    def code(self, text: str) -> None:
        # Compile before writing. A notebook whose third cell has a syntax error
        # only says so after two minutes of installs, on a machine that is not this
        # one — and the fix is a rebuild, so the cost is paid twice. A shell line
        # is not Python; blanking it would empty the block it sits in and invent a
        # syntax error the notebook does not have, so it becomes `pass` instead.
        checkable = "\n".join(
            line[: len(line) - len(line.lstrip())] + "pass"
            if line.lstrip().startswith(("!", "%"))
            else line
            for line in text.splitlines()
        )
        try:
            compile(checkable, f"{self.slug} cell {len(self.cells)}", "exec")
        except SyntaxError as e:
            raise SystemExit(f"{self.slug}: cell {len(self.cells)} does not compile - {e}") from e

        # nbformat 4.5 wants an id on every cell and warns on each run without one,
        # at the top of the log, where it reads like a fault in the notebook.
        self.cells.append({
            "cell_type": "code",
            "id": f"code{len(self.cells)}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        })

    def header(self) -> None:
        self.md(
            f'<div style="border-left:4px solid {self.colour};padding:2px 0 2px 16px;'
            'margin:6px 0 18px;">'
            f'<div style="font:800 27px/1.15 {FONT};letter-spacing:-0.02em;">'
            f'NL2SQL <span style="font-weight:500;color:{self.colour};">{self.name}</span></div>'
            f'<div style="font:400 15px/1.55 {FONT};color:#71717a;margin-top:5px;">'
            f'{self.subtitle}</div></div>'
        )

    def step(self, number: int, title: str, text: str = "") -> None:
        body = f'<div style="font:400 14px/1.6 {FONT};color:#52525b;margin-top:4px;">{text}</div>' if text else ""
        self.md(
            f'<div style="margin:22px 0 10px;">'
            f'<div style="font:600 16px/1.3 {FONT};color:#18181b;">'
            f'<span style="color:{self.colour};">{number}.</span> {title}</div>{body}</div>'
        )

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.slug}.ipynb"
        path.write_text(
            json.dumps(
                {
                    "cells": self.cells,
                    "metadata": {
                        "kernelspec": {
                            "display_name": "Python 3",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {"name": "python", "version": "3.11"},
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        (self.directory / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": self.kaggle_id,
                    "title": self.title,
                    "code_file": f"{self.slug}.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": True,
                    "enable_gpu": False,
                    "enable_internet": True,
                    "dataset_sources": [],
                    "competition_sources": [],
                    "kernel_sources": [] if self.number == 1 else [f"{OWNER}/nl2sql-1-setup"],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return path


# =================================================================================
#  Shared cells
# =================================================================================
BOOTSTRAP = f'''
# The code comes from GitHub. The repository is private, so a fresh clone needs a
# GITHUB_TOKEN secret (Add-ons -> Secrets).
#
# A version created through the API cannot read that secret - Kaggle answers 400
# no matter how the box is ticked in the editor, and the attachment cannot be
# declared in kernel-metadata.json either (Kaggle/kaggle-cli#582). Notebook 1's
# output carries the whole repository, so fall back to that copy. Only notebook 1
# has no input to fall back to, and only notebook 1 has to be saved from the
# browser rather than pushed.
import os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path("/kaggle/working/nl2sql")


def clone_from_github() -> str:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
    url = "{REPO}".replace("https://", f"https://{{token}}@")
    subprocess.run(["git", "clone", "--depth", "1", url, str(ROOT)], check=True)
    # git writes the clone URL into .git/config, token and all, and Kaggle saves
    # .git with the notebook output. Put the plain address back immediately.
    subprocess.run(["git", "-C", str(ROOT), "remote", "set-url", "origin", "{REPO}"], check=True)
    return "a fresh clone"


def find_in_mounts(relative: str, max_depth: int = 7) -> "list[Path]":
    """Every path under /kaggle/input, so nothing has to guess how deep it is.

    Kaggle has mounted a notebook's output at /kaggle/input/<slug>/ and at
    /kaggle/input/notebooks/<owner>/<slug>/, and the repository sits a further
    level inside that. Each guess at the shape reported a perfectly good output
    as a missing database, so walk for it. Bounded, and never down into the two
    directories that hold the weights and the git objects, because an attached
    output is gigabytes.
    """
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    hits = []
    for dirpath, dirnames, _ in os.walk(root):
        here = Path(dirpath)
        if len(here.relative_to(root).parts) >= max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in {{".git", "models", "wheels"}}]
        candidate = here / relative
        if candidate.exists():
            hits.append(candidate)
    return sorted(hits)


def copy_from_setup() -> str:
    # Any pyproject.toml would match, so take the one with the package beside it.
    marker = next((m for m in find_in_mounts("pyproject.toml")
                   if (m.parent / "src" / "nl2sql").is_dir()), None)
    if marker is None:
        return ""
    # Everything except data/ and models/: those are the two gigabytes that get
    # read where they are mounted and are never worth copying.
    shutil.copytree(marker.parent, ROOT,
                    ignore=shutil.ignore_patterns("data", "models", ".git"))
    # The mount is read-only and copytree keeps the modes, but `pip install -e .`
    # writes an egg-info back into the tree.
    subprocess.run(["chmod", "-R", "u+w", str(ROOT)], check=True)
    return f"the copy in {{marker.parent}}"


if ROOT.exists():
    source = "the working directory"
else:
    try:
        source = clone_from_github()
    except Exception as e:
        source = copy_from_setup()
        if not source:
            # Attached-but-empty and nothing-attached read identically from here,
            # and telling them apart is the whole difficulty, so name what is
            # mounted instead of guessing which one it is.
            mounted = sorted(p.name for p in Path("/kaggle/input").glob("*") if p.is_dir())
            where = ("the attached input(s) " + ", ".join(mounted) + " carry no "
                     "nl2sql/pyproject.toml") if mounted else "no input is attached"
            raise SystemExit(
                f"Nothing to run from: GITHUB_TOKEN could not be read "
                f"({{type(e).__name__}}: {{e}}), and {{where}}. Save this notebook from the "
                "browser, where the secret is readable - or attach a version of NL2SQL 1 "
                "Setup that ran to the end."
            ) from e
        print("GITHUB_TOKEN unreadable, falling back to notebook 1's output:", e)

sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
print("code:", ROOT, "from", source)
'''

INSTALL = '''
# pip writes to site-packages, which is not part of notebook 1's saved output, so
# every session installs again. Nearly all of it is already in the Kaggle image.
!pip install -q -e . 2>&1 | tail -2
print("dependencies ready")
'''

LOCAL_RUNTIME = '''
# Only the pages that run a model on this machine need llama-cpp-python, and
# building it costs several minutes. Notebook 1 keeps the wheel it built with its
# output, so installing from that is a copy. Runs after the cell above, so a mount
# that turned out to be unusable stops the notebook before the compile, not after.
wheelhouse = next((d for d in find_in_mounts("wheels")
                   if any(d.glob("llama_cpp_python-*.whl"))), None)
if wheelhouse:
    !pip install -q --find-links {wheelhouse} llama-cpp-python 2>&1 | tail -2
else:
    print("no wheel in notebook 1's output - building from source, a few minutes")
    !pip install -q llama-cpp-python 2>&1 | tail -2

try:
    import llama_cpp
    print("local model runtime: llama-cpp-python", llama_cpp.__version__)
except ImportError as e:
    print("llama-cpp-python is unavailable:", e)
    print("Full Local will fail, and the answer writer will show a plain table instead.")
'''

REUSE_SETUP = '''
# Notebook 1 built the database, the index and the model weights and saved them
# with its output. Kaggle mounts that output read-only under /kaggle/input, and
# every one of the three is opened read-only here too - so point the settings at
# the mount rather than copying two gigabytes into the working directory.
#
# Where inside the mount they sit is not fixed - Kaggle has used
# /kaggle/input/<slug>/ and /kaggle/input/notebooks/<owner>/<slug>/ - so this
# walks for the database rather than matching a guessed shape. find_in_mounts
# comes from the first cell.
found = find_in_mounts("data/eicu.db")
if not found:
    # Printing the tree beats asserting a cause: the last two guesses at what
    # was wrong here were both wrong, and both would have been settled by this.
    print("nothing matched. What is actually under /kaggle/input:")
    root = Path("/kaggle/input")
    listed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > 3 or listed > 40:
            dirnames[:] = []
            continue
        print("   " * depth, Path(dirpath).name + "/", " ".join(sorted(filenames)[:6]))
        listed += 1
    raise SystemExit(
        "No data/eicu.db anywhere under /kaggle/input. Notebook 1's saved output is "
        "what carries it: check Input -> Add Input -> Your Work -> NL2SQL 1 Setup, and "
        "that the version pinned there is one that ran to the end."
    )
SETUP = found[0].parents[1]
print("setup output:", SETUP)

# Name a missing piece here rather than several cells later, from inside whichever
# library opens it first.
for path in (SETUP / "data/index.db", SETUP / "models/gliner2-base-v1"):
    if not path.exists():
        print("missing from notebook 1's output:", path)

# Set before nl2sql is imported anywhere: settings() is read once and cached. A
# subprocess started later - the API server in notebook 5 - inherits these too.
os.environ["DB_PATH"] = str(SETUP / "data" / "eicu.db")
os.environ["INDEX_PATH"] = str(SETUP / "data" / "index.db")
os.environ["GLINER_MODEL"] = str(SETUP / "models" / "gliner2-base-v1")
weights = sorted(SETUP.glob("models/*/*.gguf"))
if weights:
    os.environ["LOCAL_GGUF_PATH"] = str(weights[0])

for name in ("DB_PATH", "INDEX_PATH", "GLINER_MODEL", "LOCAL_GGUF_PATH"):
    print(f"  {name:<16} {os.environ.get(name, 'missing - the steps that need it will say so')}")
'''

SECRETS = '''
# Keys live in Kaggle secrets, never in the notebook.
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
unreadable = []
for name in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "LANGSMITH_API_KEY"):
    try:
        os.environ[name] = secrets.get_secret(name)
    except Exception:
        unreadable.append(name)

# All three failing at once is one cause, not three: a version created through the
# API cannot read a notebook secret however the editor shows it, and there is no
# field for the attachment in kernel-metadata.json (Kaggle/kaggle-cli#582). Say so
# once here rather than let every provider step fail separately further down.
if unreadable:
    print("could not read:", ", ".join(unreadable))
    if len(unreadable) == 3:
        print("A version created through the API cannot read notebook secrets. Save this")
        print("notebook from the browser to run the steps that call a provider.")

os.environ["LANGSMITH_TRACING"] = "1"
os.environ["LANGSMITH_PROJECT"] = "nl2sql"
'''


# =================================================================================
#  1 — Setup
# =================================================================================
def setup(nb: Notebook) -> None:
    nb.header()

    nb.step(1, "Install", "Everything the pipeline needs, plus the local model runtime.")
    nb.code(BOOTSTRAP)
    nb.code('''
!pip install -q -e . 2>&1 | tail -2
!pip install -q huggingface-hub 2>&1 | tail -2

# Installing llama-cpp-python on this image took six and a half minutes, nearly
# all of it compiling. Build the wheel once and keep it with this notebook's
# output; notebooks 3 and 5 then install it in seconds.
!pip wheel -q --no-deps llama-cpp-python -w /kaggle/working/wheels 2>&1 | tail -2
!pip install -q --find-links /kaggle/working/wheels llama-cpp-python 2>&1 | tail -2

# `pip wheel -q` says nothing when the build fails, and the install above then
# compiles from source instead - so the wheel is absent from this notebook's
# output, notebooks 3 and 5 pay the six minutes again every session, and nothing
# on this page says why. Look for the file.
built = sorted(Path("/kaggle/working/wheels").glob("llama_cpp_python-*.whl"))
print("wheel:", built[0].name if built else
      "not built - notebooks 3 and 5 will compile from source instead")

try:
    import llama_cpp
    print("llama-cpp-python", llama_cpp.__version__)
except ImportError as e:
    print("llama-cpp-python did not install:", e)
    print("The database and the index below are unaffected; the local model is not.")
''')

    nb.step(2, "Models", "Two, both small enough to run on CPU. Downloaded once and saved with the output.")
    nb.code('''
from huggingface_hub import hf_hub_download, snapshot_download

snapshot_download("fastino/gliner2-base-v1", local_dir=ROOT / "models/gliner2-base-v1")
hf_hub_download("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf",
                local_dir=ROOT / "models/qwen3-1.7b")
print("models ready")
''')

    nb.step(3, "Database", "The published eICU-CRD demo: 31 tables, 4.6 million rows.")
    nb.code('''
!python -m nl2sql.cli database
''')

    nb.step(4, "Value index", "Which column holds which vocabulary, so a word can be traced to a real value.")
    nb.code('''
!python -m nl2sql.cli index 2>&1 | tail -12
''')

    nb.step(5, "Check", "Schema, index, glossary and gate, verified without calling any model.")
    nb.code('''
!python -m nl2sql.cli check
''')

    nb.md(
        f'<div style="font:400 14px/1.6 {FONT};color:#52525b;border-top:1px solid #e4e4e7;'
        'padding-top:12px;margin-top:26px;">Save Version &rarr; Save &amp; Run All, let '
        'it finish, then add this notebook as an input to notebooks 2 to 5. They read '
        'the database, the index and the weights out of that run&rsquo;s output, so a '
        'version that stopped early leaves them nothing to read.</div>'
    )


# =================================================================================
#  2 — Understanding
# =================================================================================
def understanding(nb: Notebook) -> None:
    nb.header()
    nb.md(
        f'<div style="font:400 15px/1.65 {FONT};color:#3f3f46;">'
        'Everything on this page happens on this machine. Nothing has left yet, and by the '
        'end the question is ready to be sent with every real value removed.</div>'
    )
    nb.code(BOOTSTRAP)
    nb.code(INSTALL)
    nb.code(REUSE_SETUP)
    nb.code(SECRETS)

    nb.step(1, "The database", "What there is to ask about.")
    nb.code('''
from nl2sql.db import schema

summary = schema.summary()
print(f"{summary['tables']} tables, {summary['columns']} columns, "
      f"{summary['rows']:,} rows, {summary['foreign_keys']} declared relationships")

print(schema.ddl({"patient", "medication"}))
''')

    nb.step(2, "The index", "Two tiers. A column is indexed when its vocabulary is bounded, "
                            "resolved on demand when it is not.")
    nb.code('''
from nl2sql.db import values

stats = values.stats()
print("tier A (indexed) :", stats["tiers"].get("A"), "columns")
print("tier B (on demand):", stats["tiers"].get("B"), "columns")
print("values indexed   :", f"{stats['values_indexed']:,}", f"({stats['size_mb']} MB)")
print()
for ref, n in stats["top"][:6]:
    print(f"  {ref:<44} {n:>6}")
''')
    nb.md(
        f'<div style="font:400 14px/1.6 {FONT};color:#52525b;">The cost of the index is bounded '
        'by the number of columns, not by the number of rows. That is what makes it transfer to '
        'a database far larger than this one.</div>'
    )

    nb.step(3, "Finding a value", "The analyst writes a word; the database holds something longer.")
    nb.code('''
for mention in ["aspirin", "asspirin", "sepsis", "female"]:
    found = values.search(mention, limit=1)
    if found:
        best = found[0]
        print(f"{mention:<12} -> {best.value[:46]:<48} {best.ref:<28} {best.score:.2f}")
    else:
        print(f"{mention:<12} -> nothing")
''')

    nb.step(4, "Naming a column", "Not every word is content. Some name a column, and the two compete.")
    nb.code('''
from nl2sql.db import catalog

for mention in ["diagnosis names", "administration routes", "aspirin"]:
    match = catalog.best(mention)
    if match:
        print(f"{mention:<24} -> {match.ref:<34} {match.score:.2f}  ({match.why})")
    else:
        print(f"{mention:<24} -> no column matches it, so it can only be a value")
''')
    nb.md(
        f'<div style="font:400 14px/1.6 {FONT};color:#52525b;">A real value scores low against '
        'every column, and a column name scores low against every value. That gap is what lets '
        'the pipeline tell them apart instead of guessing.</div>'
    )

    nb.step(5, "Reading the question", "The three previous steps, run together, with every "
                                       "decision recorded.")
    nb.code(f'''
from nl2sql.core import trace
from nl2sql.nlp.understand import understand

trace.configure()
with trace.record({QUESTION!r}) as run:
    u = understand({QUESTION!r})

for step in run.steps:
    print(f"  {{step.label:<40}} {{step.ms:>7.0f}} ms  {{step.summary}}")
''')
    nb.code('''
print("tables:", sorted(u.tables))
print()
for r in u.resolutions:
    kind = r.kind.upper()
    value = f"-> {r.value}" if r.value else ""
    print(f"  {r.mention:<16} {kind:<10} {str(r.column):<34} {value}")
''')

    nb.step(6, "Hiding the values", "Each value becomes a symbol. The mapping stays here.")
    nb.code('''
from nl2sql.privacy.mask import mask

masked = mask(u)
print("before:", u.question)
print("after :", masked.question)
print()
for symbol, value in masked.mapping.items():
    print(f"  {symbol} = {value!r}   from {masked.columns.get(symbol, 'the analyst')}")
''')

    nb.step(7, "The message that would be sent", "Assembled, then checked part by part before "
                                                 "any connection is opened.")
    nb.code('''
from nl2sql.core import prompt
from nl2sql.privacy import gate

built = prompt.hybrid(u, masked)
for verdict in gate.verdicts(built.segments):
    mark = "ok " if verdict["allowed"] else "NO "
    print(f"  {mark}{verdict['origin']:<10} {verdict['checked_by']:<30} {verdict['preview'][:52]}")
''')
    nb.code('''
print(built.messages[-1]["content"])
''')
    nb.md(
        f'<div style="font:400 14px/1.6 {FONT};color:#52525b;">Not one real value appears above. '
        'The provider is given the shape of the database and a sentence with holes in it.</div>'
    )

    nb.step(8, "What it refuses", "Two questions that cannot be answered, and are stopped here "
                                  "rather than half-answered.")
    nb.code('''
from nl2sql.privacy.mask import UnmaskableQuestion, UnresolvableValue

for question in ["Did Mr. Bensalah receive insulin?", "How many patients received asparatan?"]:
    try:
        mask(understand(question))
        print(f"{question}\\n  -> sent\\n")
    except (UnmaskableQuestion, UnresolvableValue) as e:
        print(f"{question}\\n  -> {e}\\n")
''')


# =================================================================================
#  3 — Architectures
# =================================================================================
def architectures(nb: Notebook) -> None:
    nb.header()
    nb.md(
        f'<div style="font:400 15px/1.65 {FONT};color:#3f3f46;">'
        'Four ways of answering the same question, differing only in what leaves the machine. '
        'They share every node they have in common, so the comparison is between designs and '
        'not between four separate programs.</div>'
    )
    nb.code(BOOTSTRAP)
    nb.code(INSTALL)
    nb.code(REUSE_SETUP)
    nb.code(LOCAL_RUNTIME)
    nb.code(SECRETS)

    nb.step(1, "The four designs")
    nb.md(f'''
<table style="font:400 14px/1.6 {FONT};border-collapse:collapse;width:100%;">
<tr style="border-bottom:2px solid #e4e4e7;text-align:left;">
  <th style="padding:8px 10px;">Design</th><th style="padding:8px 10px;">Question</th>
  <th style="padding:8px 10px;">Schema</th><th style="padding:8px 10px;">Rows</th>
  <th style="padding:8px 10px;">Writes the SQL</th></tr>
<tr style="border-bottom:1px solid #f4f4f5;"><td style="padding:7px 10px;"><b>Full Cloud</b></td>
  <td style="padding:7px 10px;color:#dc2626;">sent as typed</td><td style="padding:7px 10px;color:#dc2626;">sent</td>
  <td style="padding:7px 10px;color:#dc2626;">sent</td><td style="padding:7px 10px;">cloud</td></tr>
<tr style="border-bottom:1px solid #f4f4f5;"><td style="padding:7px 10px;"><b>Hybrid</b></td>
  <td style="padding:7px 10px;color:#16a34a;">values hidden</td><td style="padding:7px 10px;color:#ca8a04;">sent</td>
  <td style="padding:7px 10px;color:#16a34a;">stay here</td><td style="padding:7px 10px;">cloud</td></tr>
<tr style="border-bottom:1px solid #f4f4f5;"><td style="padding:7px 10px;"><b>Hybrid Opaque</b></td>
  <td style="padding:7px 10px;color:#16a34a;">values hidden</td><td style="padding:7px 10px;color:#16a34a;">renamed t1, c7</td>
  <td style="padding:7px 10px;color:#16a34a;">stay here</td><td style="padding:7px 10px;">cloud</td></tr>
<tr><td style="padding:7px 10px;"><b>Full Local</b></td>
  <td style="padding:7px 10px;color:#16a34a;">never leaves</td><td style="padding:7px 10px;color:#16a34a;">never leaves</td>
  <td style="padding:7px 10px;color:#16a34a;">stay here</td><td style="padding:7px 10px;">this machine</td></tr>
</table>
''')

    nb.step(2, "Drawn from the code", "Each diagram is generated from the graph that runs, so it "
                                      "cannot describe something else.")
    nb.code('''
from nl2sql.core import graph

print(graph.mermaid("hybrid"))
''')

    nb.step(3, "One question, four times")
    nb.code(f'''
import time
from nl2sql.core.state import ARMS

results = {{}}
for arm in ARMS:
    started = time.perf_counter()
    results[arm] = graph.run({QUESTION!r}, arm=arm, write=False)
    print(f"{{arm:<15}} {{(time.perf_counter() - started):>6.1f}} s")
''')

    nb.step(4, "What each one sent")
    nb.code('''
print(f"{'design':<16}{'ok':<5}{'characters out':>15}{'real values out':>17}{'tokens':>9}")
for arm, state in results.items():
    print(f"{arm:<16}{str(state.get('success')):<5}"
          f"{state.get('egress_chars', 0):>15,}{state.get('egress_values', 0):>17}"
          f"{state.get('cloud_tokens', 0):>9}")
''')
    nb.md(
        f'<div style="font:400 14px/1.6 {FONT};color:#52525b;">The column that matters is '
        '<b>real values out</b>. It is zero for the three protected designs and not zero for the '
        'baseline, and no line of code asserts that — the gate measured it.</div>'
    )

    nb.step(5, "The queries they wrote")
    nb.code('''
for arm, state in results.items():
    print(f"--- {arm}")
    print((state.get("sql") or "(none)")[:300])
    print(f"    rows: {state.get('row_count')}   {state.get('failure_reason', '')[:80]}\\n")
''')

    nb.step(6, "What the opaque design showed the provider")
    nb.code('''
opaque = results["hybrid_opaque"].get("opaque", {})
if opaque:
    print("question as sent:", opaque["question"])
    print("parameters      :", opaque["parameters"].strip())
    print()
    for label, real in opaque.get("labels", {}).items():
        print(f"  {label:<6} was {real}")
''')


# =================================================================================
#  4 — Optimization and benchmarking
# =================================================================================
def optimization(nb: Notebook) -> None:
    nb.header()
    nb.md(
        f'<div style="font:400 15px/1.65 {FONT};color:#3f3f46;">'
        'The hybrid design works. This page asks what the best version of it is — five variants, '
        'each changing one thing, measured on the same questions and ranked.</div>'
    )
    nb.code(BOOTSTRAP)
    nb.code(INSTALL)
    nb.code(REUSE_SETUP)
    nb.code(SECRETS)

    nb.step(1, "The five variants")
    nb.code('''
from nl2sql.optimize.variants import catalogue

for v in catalogue():
    print(f"  {v['name']:<11} changes {v['changes']:<15} {v['what']}")
''')

    nb.step(2, "How hard is a question?",
            "Computed from what the local stage already worked out: how many tables it touches, "
            "how many values it filters on, whether it asks for a rate or a ranking.")
    nb.code('''
from nl2sql.nlp.understand import understand
from nl2sql.optimize.variants import difficulty, starting_rung

questions = [
    "How many hospitals are there?",
    "How many patients received aspirin?",
    "What is the mortality rate per hospital region for patients over 65?",
]
for question in questions:
    score = difficulty(understand(question))
    print(f"  {score:.2f}  start at the {starting_rung(score):<7} model   {question}")
''')
    nb.md(
        f'<div style="font:400 14px/1.6 {FONT};color:#52525b;">An easy question does not need '
        'the most expensive model. This score decides where to start; perplexity decides whether '
        'to climb.</div>'
    )

    nb.step(3, "Perplexity",
            "How surprised a model was by its own answer. Low means confident, high means it was "
            "guessing — the only quality signal available without knowing the right answer.")
    nb.code(f'''
from nl2sql.core import graph

state = graph.run({QUESTION!r}, variant="cascade", write=False)
print("difficulty      :", state.get("difficulty"))
print("model that answered:", state.get("cloud_target"))
print("perplexity      :", state.get("perplexity"), "(lower is more confident)")
print("had to climb    :", state.get("escalated"))
print("calls           :", state.get("cloud_calls"))
''')

    nb.step(4, "Agreement", "The consensus variant asks three times and keeps the answer the "
                            "queries agree on, judged on what they return rather than on how "
                            "they are written.")
    nb.code(f'''
state = graph.run({QUESTION!r}, variant="consensus", write=False)
for candidate in state.get("candidates", []):
    mark = "kept" if candidate["chosen"] else "    "
    print(f"  {{mark}}  perplexity {{candidate['perplexity']}}  {{candidate['sql'][:74]}}")
''')

    nb.step(5, "The benchmark",
            "Every variant over the same questions. Accuracy is measured against a reference "
            "query where one exists, and against what the variants agree on where none does.")
    nb.code('''
from nl2sql.optimize.benchmark import compare

report = compare(limit=25)     # raise or remove for the full set
''')
    nb.code('''
if not report["table"]:
    print("no variant produced a row - check the provider keys printed above")
else:
    header = list(report["table"][0])
    print(" | ".join(f"{h:>16}" for h in header))
    print("-" * (19 * len(header)))
    for row in report["table"]:
        print(" | ".join(f"{str(row[h]):>16}" for h in header))
''')

    nb.step(6, "The ranking")
    nb.code('''
for position, (name, accuracy) in enumerate(report["ranking"], 1):
    print(f"  {position}. {name:<11} {accuracy:.0%}")

print("\\nwhere each one failed:")
for variant, failures in report["failures"].items():
    if failures:
        print(f"  {variant:<11} {failures}")
''')

    nb.step(7, "Where to put the escalation threshold",
            "Not a preference: the point that best separates the answers that turned out right "
            "from the ones that turned out wrong.")
    nb.code('''
from nl2sql.optimize.benchmark import calibrate
import json

print(json.dumps(calibrate(), indent=1))
''')


# =================================================================================
#  5 — Run and serve
# =================================================================================
def run_and_serve(nb: Notebook) -> None:
    nb.header()
    nb.md(
        f'<div style="font:400 15px/1.65 {FONT};color:#3f3f46;">'
        'The whole pipeline on one question, every step visible, then the same pipeline behind '
        'a public address so it can be used from a browser.</div>'
    )
    nb.code(BOOTSTRAP)
    nb.code(INSTALL)
    nb.code(REUSE_SETUP)
    nb.code(LOCAL_RUNTIME)
    nb.code(SECRETS)

    nb.step(1, "One question, end to end")
    nb.code(f'''
from nl2sql.core import graph

state = graph.run({QUESTION!r})
print(state["answer"])
''')

    nb.step(2, "Every step it took")
    nb.code('''
for step in state.get("trace", []):
    zone = "cloud" if step["zone"] == "cloud" else "here "
    print(f"  [{zone}] {step['label']:<42} {step['ms']:>7.0f} ms  {step['summary']}")
''')

    nb.step(3, "What left the machine")
    nb.code('''
print("characters sent  :", state.get("egress_chars"))
print("real values sent :", state.get("egress_values"))
print("query            :", state.get("sql"))
print()
from nl2sql.privacy import audit
print(audit.report())
''')

    nb.step(4, "The service", "FastAPI, started in the background.")
    nb.code('''
import subprocess, time

import httpx

# Keep the log: a server that dies on import says why here, and nowhere else.
log = open(ROOT / "uvicorn.log", "w")
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "nl2sql.api:app", "--host", "0.0.0.0", "--port", "7860"],
    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
)

# Ask until it answers rather than sleeping a guessed number of seconds: loading
# the local weights is what takes the time, and it varies.
health = None
for _ in range(60):
    if server.poll() is not None:
        raise SystemExit("the server exited:\\n" + (ROOT / "uvicorn.log").read_text()[-2000:])
    try:
        health = httpx.get("http://127.0.0.1:7860/health", timeout=5).json()
        break
    except Exception:
        time.sleep(2)

if health is None:
    raise SystemExit("the server did not answer:\\n" + (ROOT / "uvicorn.log").read_text()[-2000:])
print(health)
''')

    nb.step(5, "A public address", "A tunnel, because a Kaggle session is not reachable from "
                                   "outside on its own.")
    nb.code('''
!pip install -q pycloudflared 2>&1 | tail -1
from pycloudflared import try_cloudflare

tunnel = try_cloudflare(port=7860)
print("public address:", tunnel.tunnel)
''')

    nb.step(6, "Tell the front end where to find it",
            f"The page at {WORKER} keeps one key: the address the pipeline is currently on.")
    nb.code(f'''
import httpx

response = httpx.post(
    "{WORKER}/api/backend",
    json={{"url": tunnel.tunnel}},
    timeout=30,
)
print(response.status_code, response.text)
print("\\nopen {WORKER}")
''')

    nb.md(
        f'<div style="font:400 14px/1.6 {FONT};color:#52525b;border-top:1px solid #e4e4e7;'
        'padding-top:12px;margin-top:26px;">Leave this notebook running while the demo is in '
        'use. When the session stops, the address stops with it.</div>'
    )


BOOK = [
    (Notebook(1, "nl2sql-1-setup", "Setup", "#a1a1aa",
              "Clone the code, download the models, build the database and the index."), setup),
    (Notebook(2, "nl2sql-2-understanding", "Understanding", "#34d399",
              "How an English question becomes a schema, a symbol and an exact value."), understanding),
    (Notebook(3, "nl2sql-3-architectures", "Architectures", "#818cf8",
              "Four designs for the same question, run and measured."), architectures),
    (Notebook(4, "nl2sql-4-optimization", "Optimization", "#f472b6",
              "Five variants of the hybrid design, benchmarked and ranked."), optimization),
    (Notebook(5, "nl2sql-5-run", "Run", "#fbbf24",
              "The whole pipeline end to end, then the live service."), run_and_serve),
]


def push(notebook: Notebook) -> None:
    print(f"pushing {notebook.kaggle_id}")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(notebook.directory)],
        capture_output=True, text=True,
    )
    print((result.stdout or result.stderr).strip())
    # kaggle exits non-zero on a rejected push and the message scrolls away behind
    # the four that follow it; a failed upload must not read as a successful one.
    if result.returncode != 0:
        raise SystemExit(f"push of {notebook.kaggle_id} failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", nargs="?", const="all", help="upload to Kaggle")
    args = parser.parse_args()

    wanted = None if args.push in (None, "all") else {int(args.push)}
    for notebook, fill in BOOK:
        fill(notebook)
        path = notebook.save()
        print(f"  {path.relative_to(ROOT)}  ({len(notebook.cells)} cells)")
        if args.push and (wanted is None or notebook.number in wanted):
            push(notebook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
