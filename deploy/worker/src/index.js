/**
 * NL2SQL — the stable address.
 *
 * What this Worker does
 * ---------------------
 * Two things, and deliberately no more:
 *
 *   1. It serves the chat interface (static assets).
 *   2. It remembers where the pipeline is currently running.
 *
 * The second one exists because the pipeline runs inside a Kaggle notebook,
 * reached through a tunnel whose hostname is regenerated every time the session
 * restarts. A demo cannot be handed to anyone as a link that changes hourly, so
 * the notebook publishes its current address here on boot, and the page reads it
 * back. One KV key, holding one URL.
 *
 * What this Worker deliberately does NOT do
 * -----------------------------------------
 * It does not proxy the conversation. When you ask a question, your browser
 * opens a connection straight to the notebook; the question, the SQL and the
 * answer never pass through Cloudflare.
 *
 * Proxying would have been easier — one origin, no CORS, no rendezvous to read
 * client-side. It was rejected because this project's entire claim is that the
 * data stays inside a known boundary. Routing every answer through a third party
 * to make the URL bar tidier would have contradicted the thing being
 * demonstrated, and the report would have had to explain why it did not count.
 * The only thing Cloudflare ever learns here is that a demo exists and where it
 * is listening.
 */

const RENDEZVOUS_KEY = "backend";

// A published address is considered stale past this. A Kaggle session dies on
// its own — the tab closes, the 12-hour cap hits, the kernel is interrupted —
// and it does not get to send a goodbye. Without an expiry the page would keep
// pointing at a tunnel that stopped answering hours ago and blame the user's
// network for it.
const FRESH_FOR_S = 60 * 60 * 6;

const json = (body, status = 200, extra = {}) =>
  new Response(JSON.stringify(body, null, 2), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      "access-control-allow-origin": "*",
      ...extra,
    },
  });

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers": "content-type, authorization",
          "access-control-max-age": "86400",
        },
      });
    }

    // ---- where is the pipeline listening right now? ------------------------
    if (url.pathname === "/api/backend" && request.method === "GET") {
      const raw = await env.RENDEZVOUS.get(RENDEZVOUS_KEY);
      if (!raw) {
        return json({
          online: false,
          reason: "no session has ever published an address",
        });
      }
      const entry = JSON.parse(raw);
      const age = Math.round(Date.now() / 1000 - entry.published_at);
      return json({
        online: age < FRESH_FOR_S,
        url: entry.url,
        label: entry.label ?? "",
        published_at: entry.published_at,
        age_s: age,
        reason: age < FRESH_FOR_S ? "" : "the last published address has expired",
      });
    }

    // ---- the notebook announcing itself ------------------------------------
    if (url.pathname === "/api/backend" && request.method === "POST") {
      const token = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
      // Constant-time comparison is overkill for a demo rendezvous, but a plain
      // `!==` on a secret is the kind of thing that gets copied into something
      // that matters later.
      if (!env.PUBLISH_TOKEN || !timingSafeEqual(token, env.PUBLISH_TOKEN)) {
        return json({ error: "unauthorized" }, 401);
      }

      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: "expected a JSON body" }, 400);
      }

      // Only an https origin, and nothing else from the payload is kept. The
      // page fetches whatever lands here, so accepting an arbitrary string would
      // turn this endpoint into an open redirect for anyone holding the token.
      let backend;
      try {
        backend = new URL(body.url);
      } catch {
        return json({ error: "url is not a valid URL" }, 400);
      }
      if (backend.protocol !== "https:") {
        return json({ error: "the backend must be reachable over https" }, 400);
      }

      const entry = {
        url: backend.origin,
        label: String(body.label ?? "").slice(0, 80),
        published_at: Math.round(Date.now() / 1000),
      };
      await env.RENDEZVOUS.put(RENDEZVOUS_KEY, JSON.stringify(entry), {
        expirationTtl: FRESH_FOR_S,
      });
      return json({ ok: true, ...entry });
    }

    return env.ASSETS.fetch(request);
  },
};

function timingSafeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
