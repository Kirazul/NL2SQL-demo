/**
 * NL2SQL — client.
 *
 * The page has one job that a normal chat client does not: it has to make the
 * pipeline legible. A question here does not produce an answer, it produces a
 * sequence of stages, and which side of the trust boundary each stage ran on is
 * the thing worth showing. So the transport is server-sent events and the trace
 * is rendered as it arrives, rather than assembled at the end.
 *
 * The connection goes straight to the notebook. This page is served by a
 * Cloudflare Worker, but the Worker only tells us *where* the notebook is; it
 * never carries a question or an answer.
 */

const ARMS = [
  {
    id: "hybrid",
    name: "Hybrid",
    tagline: "The architecture under test",
    desc: "The provider writes the SQL having seen the schema and :v1 — never a value. Execution and answer writing stay local.",
  },
  {
    id: "full_cloud",
    name: "Full Cloud",
    tagline: "Unprotected baseline",
    desc: "Your question and the query results are sent as they are. Fast and accurate, and the provider sees everything.",
  },
  {
    id: "hybrid_opaque",
    name: "Hybrid Opaque",
    tagline: "Nothing recognisable leaves",
    desc: "As Hybrid, but the table and column names are pseudonymised too. The provider sees t1/c7 labels, foreign keys and :v1 — not one business word.",
  },
  {
    id: "full_local",
    name: "Full Local",
    tagline: "Nothing leaves",
    desc: "A 1.7B model on the notebook's CPU writes the SQL itself. Slower, and weaker at joins — that gap is the point.",
  },
];

const SUGGESTIONS = [
  "How many patients received aspirin?",
  "What is the average age of patients admitted to the MICU?",
  "How many female patients were discharged alive?",
  "Did Mr. Bensalah receive his insulin?",
];

const STAGES = {
  understand: { label: "Understand", hint: "entities and their real values" },
  anonymize: { label: "Mask", hint: "values become symbols" },
  generate: { label: "Generate SQL", hint: "" },
  execute: { label: "Execute", hint: "read-only SQLite" },
  write: { label: "Write the answer", hint: "" },
};

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const state = {
  arm: localStorage.getItem("nl2sql-arm") || "hybrid",
  backend: null,
  online: false,
  busy: false,
  showTrace: localStorage.getItem("nl2sql-trace") !== "0",
};

/* ------------------------------------------------------------------ boot -- */
function boot() {
  renderSuggestions();
  buildArmMenu();
  setArm(state.arm);
  wireComposer();
  wireTheme();
  wireSheet();
  $("trace-toggle").classList.toggle("is-active", state.showTrace);
  refreshBackend();
  setInterval(refreshBackend, 30000);
}

function renderSuggestions() {
  const box = $("suggestions");
  SUGGESTIONS.forEach((text) => {
    const b = el("button", "suggestion", text);
    b.type = "button";
    b.addEventListener("click", () => {
      $("input").value = text;
      autoGrow();
      submit();
    });
    box.appendChild(b);
  });
}

/* ------------------------------------------------------------ arm picker -- */
function buildArmMenu() {
  const menu = $("arm-menu");
  ARMS.forEach((arm) => {
    const option = el("button", "picker-option");
    option.type = "button";
    option.role = "option";
    option.dataset.arm = arm.id;

    const dot = el("span", "picker-dot");
    dot.dataset.arm = arm.id;

    const body = el("span");
    body.appendChild(el("span", "picker-option-name", arm.name));
    body.appendChild(el("span", "picker-option-desc", arm.desc));

    const check = el("span", "picker-check");
    check.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m5 13 4 4L19 7"/></svg>';

    option.append(dot, body, check);
    option.addEventListener("click", () => {
      setArm(arm.id);
      closeMenu();
    });
    menu.appendChild(option);
  });

  $("arm-button").addEventListener("click", (e) => {
    e.stopPropagation();
    menu.hidden ? openMenu() : closeMenu();
  });
  document.addEventListener("click", (e) => {
    if (!menu.hidden && !menu.contains(e.target)) closeMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeMenu();
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "m") {
      e.preventDefault();
      menu.hidden ? openMenu() : closeMenu();
    }
  });
}

function openMenu() {
  $("arm-menu").hidden = false;
  $("arm-button").setAttribute("aria-expanded", "true");
}
function closeMenu() {
  $("arm-menu").hidden = true;
  $("arm-button").setAttribute("aria-expanded", "false");
}

function setArm(id) {
  const arm = ARMS.find((a) => a.id === id) || ARMS[0];
  state.arm = arm.id;
  localStorage.setItem("nl2sql-arm", arm.id);
  $("arm-label").textContent = arm.name;
  document.querySelector(".picker-trigger .picker-dot").dataset.arm = arm.id;

  const badge = $("hero-arm");
  if (badge) {
    badge.textContent = arm.name.toUpperCase();
    badge.dataset.arm = arm.id;
  }

  document.querySelectorAll(".picker-option").forEach((o) => {
    o.setAttribute("aria-selected", String(o.dataset.arm === arm.id));
  });
  $("composer-note").textContent =
    arm.id === "full_cloud"
      ? "Full Cloud sends your question and the results to the provider, unmasked. That is what it measures."
      : "";
}

/* --------------------------------------------------------------- backend -- */
async function refreshBackend() {
  const dot = document.querySelector(".status-dot");
  const text = document.querySelector(".status-text");
  try {
    const r = await fetch("/api/backend", { cache: "no-store" });
    const info = await r.json();
    state.backend = info.url || null;
    state.online = Boolean(info.online && info.url);
    dot.dataset.state = state.online ? "online" : "offline";
    text.textContent = state.online ? "live" : "offline";
    $("status").title = state.online
      ? `Pipeline reachable at ${info.url} (published ${Math.round(info.age_s / 60)} min ago)`
      : info.reason || "no session is running";
  } catch {
    state.online = false;
    dot.dataset.state = "offline";
    text.textContent = "offline";
  }
  updateSend();
}

/* -------------------------------------------------------------- composer -- */
function wireComposer() {
  const input = $("input");
  input.addEventListener("input", () => {
    autoGrow();
    updateSend();
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    submit();
  });
  $("trace-toggle").addEventListener("click", () => {
    state.showTrace = !state.showTrace;
    localStorage.setItem("nl2sql-trace", state.showTrace ? "1" : "0");
    $("trace-toggle").classList.toggle("is-active", state.showTrace);
    document.querySelectorAll(".trace").forEach((t) => {
      t.hidden = !state.showTrace;
    });
  });
  input.focus();
}

function autoGrow() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 176) + "px";
}

function updateSend() {
  const ready = $("input").value.trim().length > 2 && !state.busy;
  $("send").disabled = !ready;
}

function wireTheme() {
  $("theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("nl2sql-theme", next);
  });
}

function wireSheet() {
  const sheet = $("sheet");
  $("schema-button").addEventListener("click", async () => {
    sheet.hidden = false;
    await fillSheet();
  });
  sheet.addEventListener("click", (e) => {
    if (e.target.hasAttribute("data-close")) sheet.hidden = true;
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") sheet.hidden = true;
  });
}

async function fillSheet() {
  const body = $("sheet-body");
  body.textContent = "";
  if (!state.online) {
    body.appendChild(
      el("p", null, "The pipeline is not running, so its schema cannot be read.")
    );
    return;
  }
  try {
    const meta = await (await fetch(`${state.backend}/meta`)).json();
    const s = meta.schema || {};

    const intro = el("p", "sheet-intro");
    intro.textContent =
      "eICU Collaborative Research Database v2.0.1 — de-identified intensive-care " +
      "records from 186 US hospitals, published by MIT under an open licence. " +
      "No proprietary data is used anywhere in this project.";
    body.appendChild(intro);

    // Four numbers, big enough to read at a glance. The published export had no
    // keys at all; the foreign keys are the ones we reconstructed, and they are
    // the only reason the cloud model can write a join.
    const stats = el("div", "stat-row");
    [
      [(s.rows || 0).toLocaleString("en-US"), "rows"],
      [s.tables, "tables"],
      [s.columns, "columns"],
      [s.foreign_keys, "foreign keys"],
    ].forEach(([value, label]) => {
      const tile = el("div", "stat");
      tile.appendChild(el("b", null, String(value ?? "—")));
      tile.appendChild(el("span", null, label));
      stats.appendChild(tile);
    });
    body.appendChild(stats);

    body.appendChild(el("h3", null, "What leaves this machine"));
    const note = el("p", "sheet-note");
    note.innerHTML =
      "The table structure and your question with every value replaced by " +
      '<mark class="symbol">:v1</mark>. Never a value, never a row, never the answer.';
    body.appendChild(note);

    body.appendChild(el("h3", null, `The ${(meta.tables || []).length} tables`));
    const grid = el("div", "table-grid");
    (meta.tables || []).forEach((t) => grid.appendChild(el("span", "table-chip", t)));
    body.appendChild(grid);
  } catch (e) {
    body.appendChild(el("p", null, `Could not read the schema: ${e.message}`));
  }
}

/* ----------------------------------------------------------------- asking -- */
async function submit() {
  const input = $("input");
  const question = input.value.trim();
  if (question.length < 3 || state.busy) return;

  input.value = "";
  autoGrow();

  if (!state.online) {
    await refreshBackend();
    if (!state.online) {
      showOffline(question);
      updateSend();
      return;
    }
  }

  state.busy = true;
  updateSend();
  $("send").classList.add("is-busy");

  const main = $("main");
  const messages = $("messages");
  messages.hidden = false;
  main.classList.add("has-messages");

  const turn = el("div", "turn");
  turn.appendChild(el("div", "bubble", question));

  const trace = el("div", "trace");
  trace.hidden = !state.showTrace;
  turn.appendChild(trace);

  const pending = el("div", "thinking");
  pending.innerHTML = "<i></i><i></i><i></i>";
  turn.appendChild(pending);

  messages.appendChild(turn);
  scrollDown();

  try {
    await streamAnswer(question, trace, turn, pending);
  } catch (e) {
    pending.remove();
    turn.appendChild(errorBlock(e.message || String(e)));
    refreshBackend();
  } finally {
    state.busy = false;
    $("send").classList.remove("is-busy");
    updateSend();
    input.focus();
    scrollDown();
  }
}

function showOffline(question) {
  const messages = $("messages");
  messages.hidden = false;
  $("main").classList.add("has-messages");
  const turn = el("div", "turn");
  turn.appendChild(el("div", "bubble", question));
  turn.appendChild(
    errorBlock(
      "The pipeline is not running. It lives in a Kaggle notebook that has to be " +
        "started before a demo — run the last cell, wait for it to publish its " +
        "address, then ask again."
    )
  );
  messages.appendChild(turn);
  scrollDown();
}

function errorBlock(reason) {
  const box = el("div", "answer is-error");
  box.appendChild(el("span", "error-tag", "Not reachable"));
  box.appendChild(document.createTextNode(reason));
  return box;
}

/**
 * Read the SSE body by hand.
 *
 * `EventSource` cannot be used: it only issues GET requests, and the question
 * belongs in a body rather than in a URL that ends up in every access log along
 * the way. So we stream the response and split on the blank line ourselves —
 * about fifteen lines, and it keeps the question out of the logs.
 */
async function streamAnswer(question, trace, turn, pending) {
  const response = await fetch(`${state.backend}/ask/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, arm: state.arm, write: true }),
  });
  if (!response.ok || !response.body) {
    throw new Error(`the pipeline answered ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      const event = parseEvent(block);
      if (!event) continue;

      if (event.name === "stage") {
        trace.appendChild(renderStage(event.data));
        scrollDown();
      } else if (event.name === "done") {
        pending.remove();
        turn.appendChild(renderAnswer(event.data));
        turn.appendChild(renderMetrics(event.data));
      } else if (event.name === "error") {
        pending.remove();
        turn.appendChild(errorBlock(event.data.reason));
      }
    }
  }
}

function parseEvent(block) {
  let name = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) name = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  if (!data.length) return null;
  try {
    return { name, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

/* ----------------------------------------------------------- trace render -- */
function renderStage(stage) {
  const meta = STAGES[stage.stage] || { label: stage.stage, hint: "" };
  const wrap = el("div");

  const head = el("button", "stage");
  head.type = "button";

  const zone = el("span", "zone", stage.zone === "cloud" ? "Cloud" : "Local");
  zone.dataset.zone = stage.zone || "local";

  const label = el("span", "stage-label", meta.label);
  if (meta.hint) label.appendChild(el("span", "stage-hint", meta.hint));

  const time = el("span", "stage-time", stage.ms != null ? `${formatMs(stage.ms)}` : "");

  head.append(zone, label, time);

  const body = el("div", "stage-body");
  body.hidden = true;
  fillStageBody(body, stage);

  head.addEventListener("click", () => {
    body.hidden = !body.hidden;
  });

  // The masking step opens by itself. It is the one stage that proves the claim,
  // and a demo that requires the viewer to go looking for the evidence has
  // already lost them.
  if (stage.stage === "anonymize" && !stage.failed) body.hidden = false;
  if (stage.failed) body.hidden = false;

  wrap.append(head, body);
  return wrap;
}

function fillStageBody(body, stage) {
  if (stage.failed) {
    const warn = el("div", "warn");
    warn.innerHTML =
      '<svg viewBox="0 0 24 24"><path d="M12 8v5"/><path d="M12 16.5v.2"/><path d="M10.3 3.9 2.6 17.4A2 2 0 0 0 4.3 20.4h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>';
    warn.appendChild(el("span", null, stage.reason || "stage failed"));
    body.appendChild(warn);
    return;
  }

  if (stage.stage === "understand") {
    if (!stage.entities?.length) {
      body.appendChild(el("p", null, "No entity spotted — the question has no value to protect."));
    } else {
      const chips = el("div", "chips");
      stage.entities.forEach((e) => {
        const chip = el("span", "chip");
        chip.dataset.kind = e.kind;
        chip.appendChild(el("b", null, e.mention));
        const detail =
          e.kind === "value" && e.value
            ? `${e.kind} · ${e.column} = ${e.value}`
            : e.kind === "concept" && e.column
              ? `concept · ${e.column}`
              : e.kind;
        chip.appendChild(el("small", null, detail));
        chips.appendChild(chip);
      });
      body.appendChild(chips);
    }
    if (stage.tables?.length) {
      body.appendChild(spacer());
      body.appendChild(el("div", "metrics")).innerHTML =
        `<span>tables in scope <b>${stage.tables.join(", ")}</b></span>`;
    }
    return;
  }

  if (stage.stage === "anonymize") {
    const mapping = stage.mapping || {};
    const box = el("div", "masking");

    const before = el("div", "masking-line");
    before.appendChild(el("span", "masking-tag", "stays in"));
    const beforeText = el("span", "masking-text");
    beforeText.innerHTML = highlight(stage.before || "", Object.values(mapping), "secret");
    before.appendChild(beforeText);

    const after = el("div", "masking-line");
    after.appendChild(el("span", "masking-tag", "leaves"));
    const afterText = el("span", "masking-text");
    afterText.innerHTML = highlight(stage.after || "", Object.keys(mapping), "symbol");
    after.appendChild(afterText);

    box.append(before, after);
    body.appendChild(box);

    if (Object.keys(mapping).length) {
      body.appendChild(spacer());
      const kv = el("dl", "kv");
      Object.entries(mapping).forEach(([symbol, value]) => {
        kv.appendChild(el("dt", null, symbol));
        kv.appendChild(el("dd", null, `${value}   ·   ${stage.columns?.[symbol] ?? ""}`));
      });
      body.appendChild(kv);
      body.appendChild(
        el("p", "composer-note", "This table never leaves the machine. It is the only secret in the system.")
      );
    }
    return;
  }

  if (stage.stage === "generate") {
    if (stage.egress_values > 0) {
      const warn = el("div", "warn");
      warn.innerHTML =
        '<svg viewBox="0 0 24 24"><path d="M12 8v5"/><path d="M12 16.5v.2"/><path d="M10.3 3.9 2.6 17.4A2 2 0 0 0 4.3 20.4h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>';
      warn.appendChild(
        el("span", null,
          `Egress gate bypassed: ${stage.egress_values} real value(s) and ${stage.egress_chars} characters left unmasked. This is the Full Cloud baseline.`)
      );
      body.appendChild(warn);
      body.appendChild(spacer());
    }

    // Hybrid Opaque: the whole claim of this arm is that the names went too, so
    // the labels the provider was handed are shown before anything else.
    if (stage.opaque?.question) {
      body.appendChild(renderOpaque(stage.opaque));
      body.appendChild(spacer());
    }

    if (stage.gate?.length) {
      const gate = el("div", "gate");
      stage.gate.forEach((seg) => {
        const row = el("div", "gate-row");
        row.dataset.allowed = String(seg.allowed);
        row.appendChild(el("span", "gate-verdict", seg.allowed ? "PASS" : "BLOCK"));
        row.appendChild(el("span", "gate-origin", seg.origin));
        row.appendChild(el("span", "gate-how", seg.allowed ? seg.verified_by : `refused ${seg.refused.join(", ")}`));
        gate.appendChild(row);
      });
      body.appendChild(gate);
      body.appendChild(spacer());
    }

    if (stage.sql) {
      const pre = el("pre", "sql");
      pre.innerHTML = colourSql(stage.sql);
      body.appendChild(pre);
    }

    body.appendChild(spacer());
    const m = el("div", "metrics");
    m.innerHTML =
      `<span>by <b>${escapeHtml(stage.provider || "—")}</b></span>` +
      `<span>tokens <b>${stage.tokens ?? 0}</b></span>` +
      `<span>calls <b>${stage.calls ?? 0}</b></span>` +
      `<span>repairs <b>${stage.repairs ?? 0}</b></span>` +
      `<span>sent <b>${stage.egress_chars ?? 0}</b> chars</span>` +
      `<span>values sent <b>${stage.egress_values ?? 0}</b></span>`;
    body.appendChild(m);
    return;
  }

  if (stage.stage === "execute") {
    if (!stage.rows?.length) {
      body.appendChild(el("p", null, "No matching record."));
      return;
    }
    const scroll = el("div", "table-scroll");
    const table = el("table", "rows");
    const thead = el("thead");
    const hrow = el("tr");
    (stage.columns || []).forEach((c) => hrow.appendChild(el("th", null, c)));
    thead.appendChild(hrow);
    const tbody = el("tbody");
    stage.rows.forEach((r) => {
      const tr = el("tr");
      r.forEach((v) => tr.appendChild(el("td", null, v === null ? "—" : String(v))));
      tbody.appendChild(tr);
    });
    table.append(thead, tbody);
    scroll.appendChild(table);
    body.appendChild(scroll);
    if (stage.row_count > stage.rows.length) {
      body.appendChild(el("p", "composer-note", `${stage.row_count} rows in total, first ${stage.rows.length} shown.`));
    }
    return;
  }

  if (stage.stage === "write") {
    body.appendChild(el("p", null, stage.answer || "—"));
    body.appendChild(el("div", "metrics")).innerHTML =
      `<span>written by <b>${escapeHtml(stage.author || "—")}</b></span>`;
  }
}

/**
 * The Hybrid Opaque prompt, as the provider received it.
 *
 * Hybrid can be shown with two lines — the question in, the question out. Opaque
 * needs a third dimension: the *names* were replaced as well, and a viewer who
 * only sees `:v1` in the outgoing question has been shown the hybrid guarantee,
 * not this one. So the labels are listed against what they stood for, and that
 * dictionary is marked as the secret it is — it is drawn again, differently, at
 * the next question.
 */
function renderOpaque(opaque) {
  const wrap = el("div");
  const box = el("div", "masking");

  const question = el("div", "masking-line");
  question.appendChild(el("span", "masking-tag", "leaves"));
  const questionText = el("span", "masking-text");
  questionText.innerHTML = highlightLabels(opaque.question);
  question.appendChild(questionText);
  box.appendChild(question);

  if (opaque.parameters) {
    const params = el("div", "masking-line");
    params.appendChild(el("span", "masking-tag", "leaves"));
    const paramsText = el("span", "masking-text");
    paramsText.innerHTML = highlightLabels(opaque.parameters.trim());
    params.appendChild(paramsText);
    box.appendChild(params);
  }
  wrap.appendChild(box);

  if (opaque.ddl) {
    wrap.appendChild(spacer());
    const details = el("details", "opaque-schema");
    details.appendChild(
      el("summary", null, `the schema, relabelled — ${opaque.tables} tables, ${opaque.columns} columns`)
    );
    const pre = el("pre", "sql");
    pre.innerHTML = highlightLabels(opaque.ddl);
    details.appendChild(pre);
    wrap.appendChild(details);
  }

  const labels = Object.entries(opaque.labels || {});
  if (labels.length) {
    wrap.appendChild(spacer());
    const kv = el("dl", "kv");
    labels.forEach(([alias, real]) => {
      kv.appendChild(el("dt", null, alias));
      kv.appendChild(el("dd", null, real));
    });
    wrap.appendChild(kv);
    wrap.appendChild(
      el("p", "composer-note",
        "Read right to left: those names stayed here. The dictionary is redrawn for every question, so the same table is a different label next time.")
    );
  }
  return wrap;
}

function renderAnswer(result) {
  if (result.refusal) {
    const box = el("div", "answer is-refusal");
    box.appendChild(el("span", "refusal-tag", "Request stopped — nothing was sent"));
    box.appendChild(document.createTextNode(result.failure_reason));
    return box;
  }
  if (!result.success) {
    const box = el("div", "answer is-error");
    box.appendChild(el("span", "error-tag", `Failed at ${result.failed_stage || "an unknown stage"}`));
    box.appendChild(document.createTextNode(result.failure_reason || "no reason given"));
    return box;
  }
  return el("div", "answer", result.answer);
}

function renderMetrics(result) {
  const total = Object.values(result.ms || {}).reduce((a, b) => a + b, 0);
  const m = el("div", "metrics");
  m.innerHTML =
    `<span><b>${formatMs(total)}</b> total</span>` +
    `<span>values sent to the provider <b>${result.egress_values ?? 0}</b></span>` +
    (result.row_count ? `<span>rows <b>${result.row_count}</b></span>` : "") +
    (result.cloud_tokens ? `<span>tokens <b>${result.cloud_tokens}</b></span>` : "");
  return m;
}

/* --------------------------------------------------------------- helpers -- */
const SQL_KEYWORDS =
  /\b(SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP BY|ORDER BY|HAVING|LIMIT|AS|AND|OR|NOT|IN|IS|NULL|COUNT|DISTINCT|SUM|AVG|MIN|MAX|CAST|REAL|CASE|WHEN|THEN|ELSE|END|WITH|UNION|LIKE|BETWEEN|DESC|ASC)\b/gi;

function colourSql(sql) {
  return escapeHtml(sql)
    .replace(SQL_KEYWORDS, (m) => `<span class="kw">${m}</span>`)
    .replace(/(:v\d+)/g, '<span class="param">$1</span>');
}

/** Mark the opaque labels and the bound parameters — everything a Hybrid Opaque
 *  prompt is allowed to contain besides plain English and SQL. `t1` must not
 *  match inside `t10`, hence the boundaries. */
const highlightLabels = (text) =>
  escapeHtml(text).replace(
    /(?<![A-Za-z0-9_])([tc]\d+|:v\d+)(?![A-Za-z0-9_])/g,
    '<mark class="symbol">$1</mark>'
  );

/** Wrap each needle in a <mark>. Longest first, so a needle contained inside
 *  another does not break the outer one's markup. */
function highlight(text, needles, className) {
  let html = escapeHtml(text);
  [...needles]
    .filter(Boolean)
    .sort((a, b) => b.length - a.length)
    .forEach((needle) => {
      const pattern = new RegExp(escapeRegExp(escapeHtml(needle)), "gi");
      html = html.replace(pattern, (m) => `<mark class="${className}">${m}</mark>`);
    });
  return html;
}

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );

const escapeRegExp = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const formatMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

const spacer = () => {
  const s = el("div");
  s.style.height = "0.625rem";
  return s;
};

/**
 * Keep the newest content in view.
 *
 * A single scroll call is not enough on the first message: the hero is still
 * collapsing over 400 ms, so the page height it scrolls against is not the
 * height it will have when the animation ends. Hence the follow-up pass once
 * the transition has settled.
 */
let settleTimer = null;
function scrollDown() {
  const main = $("main");
  const jump = () =>
    main.scrollTo({ top: main.scrollHeight, behavior: "smooth" });
  jump();
  clearTimeout(settleTimer);
  settleTimer = setTimeout(jump, 450);
}

boot();
