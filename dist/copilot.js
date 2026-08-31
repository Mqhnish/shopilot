/* Shopping Copilot — demo page logic.
 *
 * Talks to server.py over /api/copilot. Every panel renders a field the agent
 * actually produced: `track` and `specificity` come from src/route.py, `gains`
 * from src/clarify.py, `trace` from src/rank.py, and `decision` reconstructs
 * the list-length choice in src/agent.py::_trim. Nothing here is simulated —
 * if a field is missing the panel says so rather than inventing a value.
 *
 * INTEGRATION NOTE
 * The shared visual effects (cursor, magnetic buttons, tilt, kinetic reveals,
 * scroll progress, ticker) are kept from the original page and each sits behind
 * `claimEffect()`, which no-ops if a host site already installed it. Standalone,
 * nothing has claimed them, so they all run.
 */

(() => {
  "use strict";

  const API = window.COPILOT_API_BASE || "/api/copilot";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const fine = window.matchMedia("(pointer: fine)").matches;
  const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Returns false if the host site already owns this effect. */
  const claimed = new Set(
    $$("[data-shared-effect]")
      .filter((el) => el.dataset.claimedBy && el.dataset.claimedBy !== "copilot")
      .map((el) => el.dataset.sharedEffect)
  );
  function claimEffect(name) {
    if (claimed.has(name) || window.__siteEffects?.[name]) return false;
    claimed.add(name);
    return true;
  }

  /* ── Shared visual effects ───────────────────────────────────────── */

  function initCursor() {
    if (!fine || calm || !claimEffect("cursor")) return;
    const dot = Object.assign(document.createElement("div"), {
      className: "cp-cursor cp-cursor-dot",
    });
    const ring = Object.assign(document.createElement("div"), {
      className: "cp-cursor cp-cursor-ring",
    });
    document.body.append(dot, ring);
    document.body.classList.add("cp-has-cursor");

    let mx = innerWidth / 2, my = innerHeight / 2, rx = mx, ry = my;
    addEventListener("pointermove", (e) => {
      mx = e.clientX; my = e.clientY;
      dot.style.transform = `translate(${mx}px, ${my}px)`;
    }, { passive: true });

    (function loop() {
      // The ring lags the dot. Lerp rather than CSS transition so it keeps
      // trailing continuously instead of easing to each discrete position.
      rx += (mx - rx) * 0.16;
      ry += (my - ry) * 0.16;
      ring.style.transform = `translate(${rx}px, ${ry}px)`;
      requestAnimationFrame(loop);
    })();

    const hot = "a, button, input, select, .chip, [data-magnetic]";
    addEventListener("pointerover", (e) => {
      if (e.target.closest?.(hot)) ring.classList.add("is-hot");
    });
    addEventListener("pointerout", (e) => {
      if (e.target.closest?.(hot)) ring.classList.remove("is-hot");
    });
  }

  function initMagnetic() {
    if (!fine || calm || !claimEffect("magnetic")) return;
    $$("[data-magnetic]").forEach((el) => {
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        const dx = e.clientX - (r.left + r.width / 2);
        const dy = e.clientY - (r.top + r.height / 2);
        el.style.transform = `translate(${dx * 0.22}px, ${dy * 0.3}px)`;
      });
      el.addEventListener("pointerleave", () => { el.style.transform = ""; });
    });
  }

  function initTilt() {
    if (!fine || calm || !claimEffect("tilt")) return;
    $$("[data-tilt]").forEach((el) => {
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width - 0.5;
        const py = (e.clientY - r.top) / r.height - 0.5;
        el.style.transform =
          `perspective(1000px) rotateY(${px * 2.4}deg) rotateX(${-py * 2.4}deg)`;
      });
      el.addEventListener("pointerleave", () => { el.style.transform = ""; });
    });
  }

  function initKinetic() {
    if (!claimEffect("kinetic")) return;
    const targets = $$("[data-kinetic]");
    // With reduced motion the CSS already forces the end state; splitting
    // would only add markup for no benefit.
    if (calm) { targets.forEach((t) => t.classList.add("is-revealed")); return; }

    targets.forEach((el) => {
      const html = el.innerHTML.split("<br>").map((line) =>
        line.trim().split(/\s+/)
          .map((w, i) => `<span class="kw"><i style="--d:${i * 42}ms">${w}</i></span>`)
          .join(" ")
      ).join("<br>");
      el.innerHTML = html;
    });

    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          en.target.classList.add("is-revealed");
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.25 });
    targets.forEach((t) => io.observe(t));
  }

  function initScrollProgress() {
    const bar = $('[data-shared-effect="scroll-progress"] i');
    if (!bar || !claimEffect("scroll-progress")) return;
    const update = () => {
      const max = document.documentElement.scrollHeight - innerHeight;
      bar.style.width = `${max > 0 ? (scrollY / max) * 100 : 0}%`;
    };
    addEventListener("scroll", update, { passive: true });
    update();
  }

  function initTicker() {
    if (calm || !claimEffect("ticker")) return;
    // Duplicate the run so the -50% marquee keyframe loops seamlessly.
    const run = $(".ticker-run");
    if (run) run.innerHTML += run.innerHTML;
  }

  /* ── Elements ────────────────────────────────────────────────────── */

  const el = {
    thread: $("#thread"),
    empty: $("#emptyState"),
    chips: $("#starterChips"),
    msg: $("#msg"),
    send: $("#send"),
    typeahead: $("#typeahead"),
    scopeNote: $("#scopeNote"),
    reset: $("#reset"),
    status: $("[data-status]"),
    turns: $("#turnCounter"),
    latency: $("#latency"),
    needle: $("#needle"),
    trackLabel: $("#trackLabel"),
    trackNote: $("#trackNote"),
    weightList: $("#weightList"),
    slots: $("#slots"),
    slotCount: $("#slotCount"),
    phraseCloud: $("#phraseCloud"),
    askLabel: $("#askLabel"),
    gains: $("#gains"),
    decisionMode: $("#decisionMode"),
    decisionHead: $("#decisionHead"),
    decisionWhy: $("#decisionWhy"),
    retrievalKv: $("#retrievalKv"),
    oursResults: $("#oursResults"),
    baseResults: $("#baseResults"),
    oursTag: $("#oursTag"),
    baseTag: $("#baseTag"),
    profileChips: $("#profileChips"),
    profileSummary: $("#profileSummary"),
    profilePanel: $("#profilePanel"),
    profileWeight: $("#profileWeight"),
    modeChips: $("#modeChips"),
    modeSummary: $("#modeSummary"),
    replayPick: $("#replayPick"),
    replayRun: $("#replayRun"),
    replayRandom: $("#replayRandom"),
    replayStatus: $("#replayStatus"),
    replayOut: $("#replayOut"),
    replayTarget: $("#replayTarget"),
    replayTurns: $("#replayTurns"),
    footMeta: $("#footMeta"),
  };

  /* Profiles are the exact shape the evaluator hands the agent: safe
     aggregates only, never an identifier. */
  const PROFILES = {
    none: {
      label: "no history",
      profile: null,
      summary: "No prior signal; the agent has only what you say.",
    },
    fit: {
      label: "fit · comfort · durability",
      profile: {
        average_prior_rating: 5.0,
        preference_tags: ["fit", "comfort", "durability"],
        purchase_frequency: "3-4 prior purchases",
        rating_style: "usually positive",
        summary: "Prior purchases emphasize fit, comfort, durability; ratings are usually positive.",
      },
      summary: "Prior purchases emphasize fit, comfort, durability; ratings usually positive.",
    },
    value: {
      label: "value · quality",
      profile: {
        average_prior_rating: 3.4,
        preference_tags: ["value", "quality"],
        purchase_frequency: "10+ prior purchases",
        rating_style: "critical",
        summary: "Frequent buyer who emphasizes value and quality; ratings are critical.",
      },
      summary: "Frequent buyer, emphasizes value and quality; ratings are critical.",
    },
    style: {
      label: "style · design",
      profile: {
        average_prior_rating: 4.6,
        preference_tags: ["style", "design", "material"],
        purchase_frequency: "1-2 prior purchases",
        rating_style: "usually positive",
        summary: "Prior purchases emphasize style, design, material; ratings are usually positive.",
      },
      summary: "Prior purchases emphasize style, design and material.",
    },
  };

  /* Two configurations of the same agent over one index. "scored" is exactly
     what the evaluator runs; "shopper" turns off list truncation, which costs
     0.070 of composite score and is the better experience for a person. */
  const MODES = {
    shopper: {
      label: "show me options",
      summary: "Full ranked list every turn — the better shopping experience.",
    },
    scored: {
      label: "scored configuration",
      summary: "Withholds nine results to probe with one. Worth +0.070 of score, worse to use.",
    },
  };

  let sessionId = newSession();
  /* src.rank.W_PROFILE_TAG, replaced by the real value on the first turn. Kept
     as a variable rather than read off `window` so the card cannot quote a
     weight the ranker does not use. */
  let profileWeight = 0.18;
  let modeKey = "shopper";
  let profileKey = "fit";
  let busy = false;
  let lastSlots = {};

  function newSession() {
    return `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  }

  const money = (v) => (v == null ? "—" : `$${Number(v).toFixed(2)}`);
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, d = 3) => (v == null ? "—" : Number(v).toFixed(d));
  /* 12,481 -> 12.5k. Review counts run to six figures and are context,
     not data: at full width they out-shout the product title. */
  const compact = (n) =>
    n >= 1e6 ? `${(n / 1e6).toFixed(1)}M`
    : n >= 1e4 ? `${Math.round(n / 1e3)}k`
    : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k`
    : String(n);

  function setStatus(s) { if (el.status) el.status.dataset.status = s; }

  /* ── Networking ──────────────────────────────────────────────────── */
  /* Two modes behind one seam. Live, these are ordinary fetches against
     server.py. Static — used for the hosted build — they resolve out of a
     bundle recorded from a running agent by tools/build_static.py, because the
     agent needs a persistent process and a static host has none.

     Nothing in the bundle is fabricated: every turn is the real response the
     real agent gave. What a static build cannot do is answer a sentence nobody
     recorded, so the page offers the recorded conversations instead of a text
     box, and says why. */

  const STATIC_SRC = window.COPILOT_STATIC || null;
  let CANNED = null;              // the recorded bundle, once loaded
  let script = null;              // the conversation being played back
  let scriptAt = 0;               // how far through it we are

  async function loadCanned() {
    if (!STATIC_SRC || CANNED) return CANNED;
    const res = await fetch(STATIC_SRC);
    if (!res.ok) throw new Error(`could not load ${STATIC_SRC}`);
    CANNED = await res.json();
    return CANNED;
  }

  /* The recorded reply for exactly what was just said, or null.

     Strict on purpose. An earlier version advanced the script on any input,
     which meant typing "show me a laptop" returned the next recorded turn --
     a real reply, but to a different question. Showing someone a genuine
     response to something they did not ask is worse than telling them the
     build cannot answer it. */
  function cannedTurn(message) {
    if (!CANNED) return null;
    const said = String(message || "").trim().toLowerCase();
    const opening = CANNED.scripts.find(
      (s) => s.recorded[0]?.said.trim().toLowerCase() === said);
    if (opening) { script = opening; scriptAt = 0; }
    if (!script) return null;
    const step = script.recorded[scriptAt];
    if (!step || step.said.trim().toLowerCase() !== said) return null;
    scriptAt += 1;
    return step;
  }

  /* The next line of the conversation being played back, so the page can offer
     it rather than leaving a visitor to guess the exact wording. */
  function cannedNext() {
    return STATIC_SRC && script ? script.recorded[scriptAt]?.said || null : null;
  }

  async function get(path) {
    if (STATIC_SRC) {
      await loadCanned();
      if (path === "/benchmark") return CANNED.benchmark;
      if (path === "/suggestions") return CANNED.suggestions;
      if (path === "/sessions") return { sessions: CANNED.sessions };
      if (path === "/health") return CANNED.health;
      if (path.startsWith("/categories")) {
        const q = new URLSearchParams(path.split("?")[1] || "").get("q") || "";
        return CANNED.categories[q] || CANNED.categories[""] ||
               { categories: [], corrections: {}, small_talk: null, note: null };
      }
      if (path.startsWith("/refinements")) {
        const step = script?.recorded[Math.max(0, scriptAt - 1)];
        return step?.refinements || { ready: false, values: [], actions: [] };
      }
      if (path.startsWith("/similar")) return { similar: [] };
      throw new Error(`not recorded: ${path}`);
    }
    const res = await fetch(`${API}${path}`);
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  }

  async function post(path, body) {
    if (STATIC_SRC) {
      await loadCanned();
      if (path === "/reset") { script = null; scriptAt = 0; return { ok: true }; }
      if (path === "/chat") {
        const step = cannedTurn(body.message);
        if (!step) {
          const err = new Error("not recorded");
          err.notRecorded = true;
          throw err;
        }
        return step.reply;
      }
      if (path === "/baseline") {
        const step = script?.recorded[Math.max(0, scriptAt - 1)];
        return step?.baseline || { ready: false, results: [], note:
          "The BM25 comparison was not recorded for this turn." };
      }
      if (path === "/replay") {
        const run = CANNED.replays[body.sample_id];
        if (!run) throw new Error("not recorded");
        return run;
      }
      throw new Error(`not recorded: ${path}`);
    }
    const res = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`${res.status} ${detail.slice(0, 200)}`);
    }
    return res.json();
  }

  /* ── Conversation rendering ──────────────────────────────────────── */

  /* ── Result cards ────────────────────────────────────────────────── */

  /* Only 20.8% of the frozen catalog carries a price, and there are no images
     in it at all -- the brief restricts the task to text catalogs. So the card
     has to earn its place on what is actually there: the title, the store, the
     rating, and above all *why this row is here*. A ranked list with no reasons
     is a leap of faith, and it is the first thing anyone asks about. */

  function stars(rating) {
    // Five glyphs, half-steps rounded, so the rating reads at a glance instead
    // of being parsed as a number.
    const filled = Math.round((Number(rating) || 0) * 2) / 2;
    let out = "";
    for (let i = 1; i <= 5; i += 1) {
      out += i <= filled ? "★" : (i - 0.5 === filled ? "◐" : "☆");
    }
    return out;
  }

  function ratingCell(r) {
    if (!r.rating) return "";
    const count = r.rating_count
      ? `<i class="c">${compact(r.rating_count)}</i>` : "";
    return `<span class="m-rating" title="${num(r.rating, 2)} from ${
      (r.rating_count || 0).toLocaleString()} ratings">
        <i class="s" aria-hidden="true">${stars(r.rating)}</i>${count}
      </span>`;
  }

  /* Price leads when there is one. Only 20.8% of the catalog has a price, so
     when there is not, the rating leads and the absence drops to a quiet second
     line — a bare dash reads as broken data rather than absent data, but
     "no price" set above the stars on four cards in five shouts about the one
     thing the card cannot tell you. */
  function statsCell(r) {
    const rating = ratingCell(r);
    if (r.price != null) {
      return `<span class="m-price">${money(r.price)}</span>${rating}`;
    }
    return `${rating}<span class="m-price is-none"
      title="This product carries no price in the frozen catalog">no price</span>`;
  }

  /* The evidence row. Every chip is a constraint the product literally carries,
     established by the same two-tier lookup the phrase route scores with, so a
     chip is on the card if and only if the ranker credited it. */
  function whyRow(r, shared = EMPTY) {
    // Evidence every row on the turn carries is not evidence about *this* row.
    // It is stated once above the list and dropped from the cards, so what is
    // left on a card is exactly what separates it from the rows around it.
    const why = (r.why || []).filter((w) => !shared.has(w.text));
    const tags = r.matched_tags || [];
    if (!why.length && !tags.length && r.in_category !== false && !r.over_budget) return "";
    const LIMIT = 4;
    const shown = why.slice(0, LIMIT);
    const chips = shown.map((w) => {
      const cls = ["ev",
        w.decisive ? "is-decisive" : "",
        w.tier === "soft" ? "is-soft" : "",
        w.tier === "text" ? "is-term" : ""].filter(Boolean).join(" ");
      const how = w.tier === "exact"
        ? `carried verbatim by ${w.count.toLocaleString()} of 50,000 products`
        : w.tier === "soft"
          ? `matched with punctuation stripped — ${w.count.toLocaleString()} products carry it`
          : `a word you used, present in this product's own indexed text (${
              w.count.toLocaleString()} products use it)`;
      return `<i class="${cls}" title="${esc(w.text)} — ${how}">${esc(w.text)}</i>`;
    }).join("");
    const more = why.length > LIMIT
      ? `<i class="ev is-more" title="${esc(why.slice(LIMIT).map((w) => w.text).join(", "))}"
            >+${why.length - LIMIT}</i>` : "";
    const profile = tags.map((t) => `
      <i class="ev is-profile" title="From your profile, weighted ${profileWeight
        } against 1.00 for anything you actually say — it can tilt a tie, never overrule you.">${
        esc(t)}</i>`).join("");
    const off = r.in_category === false
      ? `<i class="ev is-off" title="Outside the category you named — the category is a bonus in the ranker, never a filter, so the text evidence outweighed it">other category</i>`
      : "";
    // The catalog writes a price as "budget around $X" — a target, not a
    // ceiling — and the ranker has no numeric comparison at all. Rather than
    // paper over that, the rows that break a stated ceiling say so.
    const over = r.over_budget
      ? `<i class="ev is-over" title="Above the ceiling you named. The catalog states a product's price as “budget around $X”, which is a target rather than a limit, and the scored ranker does no numeric comparison — so this is flagged, not filtered.">over budget</i>`
      : "";
    if (!chips && !more && !profile && !off && !over) return "";
    return `<span class="card-why">${over}${off}${chips}${more}${profile}</span>`;
  }

  /* `mark` is a class name, not raw markup: the caller used to pass a whole
     ` class="is-shared"` attribute, which silently produced two `class`
     attributes on the same element once the card grew one of its own. */
  const EMPTY = new Set();

  function productRow(r, mark = "", shared = EMPTY) {
    const meta = [r.store, r.category].filter(Boolean).join(" · ");
    const rank = r.rank ?? "";
    const classes = ["card", rank === 1 ? "is-lead" : "", mark]
      .filter(Boolean).join(" ");
    return `
      <li class="${classes}" data-asin="${esc(r.parent_asin)}" tabindex="0">
        <span class="card-rank" aria-hidden="true">${rank}</span>
        <span class="card-body">
          <span class="card-title" title="${esc(r.title)}">${esc(r.title)}</span>
          ${meta ? `<span class="card-meta">${esc(meta)}</span>` : ""}
        </span>
        <span class="card-stats">${statsCell(r)}</span>
        ${whyRow(r, shared)}
      </li>`;
  }

  /* Evidence texts carried by every one of these cards. Only worth factoring
     out when there are cards to compare: on a one-row probe turn the single
     card's reasons are the interesting thing, not a shared preamble. */
  function sharedEvidence(cards) {
    if (cards.length < 2) return EMPTY;
    let common = null;
    for (const card of cards) {
      const here = new Set((card.why || []).map((w) => w.text));
      if (common === null) { common = here; continue; }
      for (const text of common) if (!here.has(text)) common.delete(text);
      if (!common.size) return EMPTY;
    }
    return common || EMPTY;
  }

  /* ── Product detail on hover ─────────────────────────────────────── */
  /* Four near-identical similar titles was the wrong payload: in a category of
     duplicates they read as four more of the same row. This is a detail card —
     the full title, the stats, and above all the *disclosure surface*: the
     exact strings the customer simulator would reveal about this product if the
     agent asked, and which question would pull each one out. Fetched on hover
     (and on focus, so it is reachable from the keyboard) and cached, because
     the same row is hovered repeatedly while reading.

     Positioned `fixed` against the row's viewport box rather than absolutely
     inside it: the conversation scrolls under `overflow-y: auto`, which clips
     an absolutely-positioned child, so the panel used to be cut in half on
     every row below the fold. Fixed placement also lets it flip above the row
     when there is no room below. */

  const similarCache = new Map();
  let similarTimer = null;
  let similarPanel = null;
  let similarFor = null;   // the row the visible panel belongs to
  let similarSeq = 0;      // discards a fetch overtaken by a later hover

  function hideSimilar() {
    clearTimeout(similarTimer);
    similarSeq += 1;
    if (similarPanel) { similarPanel.remove(); similarPanel = null; }
    similarFor = null;
  }

  function placeSimilar(row, panel) {
    const r = row.getBoundingClientRect();
    const gap = 6;
    const w = panel.offsetWidth;
    const h = panel.offsetHeight;
    // Prefer below-right, flip above when that would leave the viewport, then
    // clamp regardless: a row can be scrolled out of view between the hover and
    // this measurement (Tab-focus scrolls the row itself), and a panel placed
    // off-screen is indistinguishable from a broken one.
    let top = r.bottom + gap;
    if (top + h > window.innerHeight - 8) top = r.top - h - gap;
    top = Math.min(Math.max(8, top), Math.max(8, window.innerHeight - h - 8));
    let left = r.right - w;
    left = Math.min(Math.max(8, left), Math.max(8, window.innerWidth - w - 8));
    panel.style.top = `${Math.round(top)}px`;
    panel.style.left = `${Math.round(left)}px`;
  }

  async function showSimilar(row) {
    const asin = row.dataset.asin;
    if (!asin || similarFor === row) return;
    const seq = ++similarSeq;
    let data = similarCache.get(asin);
    if (!data) {
      try {
        data = await get(`/similar?asin=${encodeURIComponent(asin)}` +
                         `&session_id=${encodeURIComponent(sessionId)}`);
        // Bounded: a long session hovers hundreds of rows and this is a cache,
        // not a store.
        if (similarCache.size > 120) similarCache.clear();
        similarCache.set(asin, data);
      } catch { return; }
    }
    // A later hover, a reset, or a re-render happened while this was in flight.
    if (seq !== similarSeq || !row.isConnected || !data.product) return;

    const p = data.product;
    // The disclosure surface: the exact strings the simulator would reveal
    // about this product if asked, and which question would pull each one out.
    // It is what the whole system is built around and the one place a person
    // can look at it directly.
    const discloses = (data.discloses || []).map((d) => `
      <li class="sp-dis${d.known ? " is-known" : ""}">
        <span class="sp-dis-text">${esc(d.text)}</span>
        <b class="mono">${esc(d.attribute)}</b>
      </li>`).join("");
    const similar = (data.similar || []).map((s) => `
      <li>
        <span class="t">${esc(s.title)}</span>
        <span class="p">${s.price != null ? money(s.price)
          : `<i class="s">${stars(s.rating)}</i>`}</span>
      </li>`).join("");

    if (similarPanel) similarPanel.remove();
    similarPanel = document.createElement("div");
    similarPanel.className = "similar-pop";
    similarPanel.innerHTML = `
      <p class="sp-title">${esc(p.title)}</p>
      <p class="sp-meta">${esc([p.store, p.category].filter(Boolean).join(" · "))}</p>
      <p class="sp-stats">
        ${p.price != null ? `<span class="m-price">${money(p.price)}</span>`
                          : `<span class="m-price is-none">no price</span>`}
        ${p.rating ? `<span class="m-rating"><i class="s">${stars(p.rating)}</i>
           <i class="c">${num(p.rating, 1)} · ${(p.rating_count || 0).toLocaleString()}</i>
         </span>` : ""}
      </p>
      ${discloses ? `
        <p class="sp-head mono">would disclose if asked
          <i>${data.discloses.filter((d) => d.known).length}/${data.discloses.length} heard</i></p>
        <ul class="sp-list">${discloses}</ul>` : ""}
      ${similar ? `
        <p class="sp-head mono">closest in ${esc(p.category || "this category")}
          <i>${(data.pool || 0).toLocaleString()} in it</i></p>
        <ol class="sp-list sp-similar">${similar}</ol>` : ""}`;
    document.body.append(similarPanel);
    similarFor = row;
    placeSimilar(row, similarPanel);
  }

  document.addEventListener("pointerover", (e) => {
    const row = e.target.closest?.("li.card[data-asin]");
    if (!row) return;
    clearTimeout(similarTimer);
    // A delay, or every row you sweep past fires a request.
    similarTimer = setTimeout(() => showSimilar(row), 320);
  });
  document.addEventListener("pointerout", (e) => {
    const row = e.target.closest?.("li.card[data-asin]");
    if (row && !row.contains(e.relatedTarget)) hideSimilar();
  });
  document.addEventListener("focusin", (e) => {
    const row = e.target.closest?.("li.card[data-asin]");
    if (row) showSimilar(row); else hideSimilar();
  });
  /* The panel is out of flow now, so anything that moves the row has to move
     the panel with it. Repositioning rather than hiding matters because the
     page scrolls smoothly: `scrollThread` animates for most of a second after
     every turn, and hiding on scroll made the panel un-summonable for that
     whole window. It is dismissed only once its row has actually left. */
  function trackSimilar() {
    if (!similarPanel || !similarFor) return;
    const r = similarFor.getBoundingClientRect();
    if (!similarFor.isConnected || r.bottom < 0 || r.top > window.innerHeight) {
      hideSimilar();
      return;
    }
    placeSimilar(similarFor, similarPanel);
  }
  for (const event of ["scroll", "resize"]) {
    window.addEventListener(event, trackSimilar, { passive: true, capture: true });
  }

  function addUserTurn(text) {
    el.empty?.remove();
    const wrap = document.createElement("div");
    wrap.className = "turn turn-user";
    wrap.innerHTML = `<div class="bubble">${esc(text)}</div>`;
    el.thread.append(wrap);
    scrollThread();
  }

  /* ── Notes between a question and its answer ─────────────────────── */
  /* These sit in the one place on the page where length is expensive: between
     what someone typed and what came back. The rigour has to stay — a search
     box that quietly changes your query is worse than one that fails — but it
     does not have to be read first. So each note is one line that says what
     changed, and a `why?` disclosure holding the reasoning. Native <details>,
     so it needs no script and reaches the keyboard for free. */

  function note(kind, headline, summary, why) {
    const wrap = document.createElement("div");
    wrap.className = "turn turn-note";
    wrap.innerHTML = `
      <div class="assist-note ${kind}">
        <b>${headline}</b>
        ${summary ? `<p class="note-line">${summary}</p>` : ""}
        ${why ? `<details class="note-why">
            <summary>why?</summary>
            <div class="note-why-body">${why}</div>
          </details>` : ""}
      </div>`;
    el.thread.append(wrap);
    return wrap;
  }

  function addAssistNote(a) {
    const fixes = Object.entries(a.corrections || {});
    // The one line: what it searched, and anything it had to infer to get there.
    const summary = [
      a.via ? `matched through &ldquo;${esc(a.via)}&rdquo;` : "",
      a.ceiling != null ? `over ${money(a.ceiling)} flagged, not filtered` : "",
      a.switched_from ? "earlier constraints dropped" : "",
      fixes.length
        ? `spelling: ${fixes.map(([f, t]) => `<s>${esc(f)}</s> ${esc(t)}`).join(", ")}`
        : "",
    ].filter(Boolean).join(" · ");

    const why = `
      <p>The agent is built to be told a catalog category — the way the
      evaluator opens every scored session — so the demo resolved one first and
      sent it this, verbatim:</p>
      <code>${esc(a.message)}</code>
      ${a.via ? `<p>Nothing you typed names a category, so it routed through
        <b>&ldquo;${esc(a.via)}&rdquo;</b>, an occasion word rather than a
        product word. That is an inference, which is why it is labelled.</p>` : ""}
      ${a.ceiling != null ? `<p>You named a ceiling of <b>${money(a.ceiling)}</b>.
        The catalog states a price as <em>budget around $X</em> — a target, not
        a limit — and the scored ranker does no numeric comparison at all, so
        rows above it are flagged rather than removed.</p>` : ""}
      ${a.switched_from ? `<p>That changed the subject. The agent fixes its
        category for the life of a session by design, so a new subject is a new
        session and the earlier constraints went with it.</p>` : ""}`;

    note(
      a.switched_from ? "is-switch" : "",
      a.switched_from
        ? `New search — left <s>${esc(a.switched_from)}</s> for <span>${esc(a.category)}</span> · ${a.pool.toLocaleString()} products`
        : `Searching <span>${esc(a.category)}</span> · ${a.pool.toLocaleString()} products`,
      summary,
      why,
    );
  }

  function addRewriteNote(typed, sent) {
    note("", `Read &ldquo;${esc(typed)}&rdquo; as a rejection`, "",
      `<p>The customer simulator has exactly one way of saying that and the
       parser reads only that one, so the demo sent:</p>
       <code>${esc(sent)}</code>`);
  }

  function addUnmatchedNote(words, category) {
    const quoted = words.map((w) => `&ldquo;${esc(w)}&rdquo;`).join(", ");
    const wrap = note("is-scope",
      `No category here for ${quoted}`,
      `Still showing <em>${esc(category || "your current search")}</em>.`,
      `<p>This catalog holds 50,000 <em>Clothing, Shoes &amp; Jewelry</em>
       products and nothing else. A word that names no category cannot narrow
       the search, and the agent locks its category on the opening turn — so
       refining will not rescue it.</p>`);
    wrap.querySelector(".assist-note").insertAdjacentHTML("beforeend",
      '<button class="chip" type="button" data-reset-search>Start a new search</button>');
  }

  function addScopeWarning(typed, turn) {
    const wrap = turn > 1
      ? note("is-scope", "Still no catalog category for this search",
          "Everything below is a plain text match.",
          `<p>The agent fixes its category on the opening turn, so refining will
           not rescue it — the results are a text match against
           <em>Clothing, Shoes &amp; Jewelry</em> rather than a category result.</p>`)
      : note("is-scope",
          `Nothing here matches &ldquo;${esc(typed)}&rdquo; as a category`,
          "The answer below is the closest text match, not a category result.",
          `<p>This catalog is 50,000 <em>Clothing, Shoes &amp; Jewelry</em>
           products — no electronics, no phones, no home goods. Start typing a
           garment or an accessory and the box will offer the categories that
           do exist.</p>`);
    if (turn > 1) {
      wrap.querySelector(".assist-note").insertAdjacentHTML("beforeend",
        '<button class="chip" type="button" data-reset-search>Start a new search</button>');
    }
  }

  function addAgentTurn(data) {
    const wrap = document.createElement("div");
    wrap.className = "turn turn-agent";
    const asking = Boolean(data.ask_attribute);
    const probe = data.decision?.mode === "probe";
    const shown = probe ? 1 : 6;
    const cards = (data.results || []).slice(0, shown);
    const shared = sharedEvidence(cards);
    const rows = cards.map((r) => productRow(r, "", shared)).join("");

    // What every row here has in common, stated once. The `why` entries are
    // identical objects across cards for a shared text, so the first card's
    // copy carries the right tier and rarity.
    const sharedChips = [...shared].map((text) => {
      const w = (cards[0].why || []).find((x) => x.text === text);
      return `<i class="ev${w?.decisive ? " is-decisive" : ""}${
        w?.tier === "text" ? " is-term" : ""}">${esc(text)}</i>`;
    }).join("");

    // Concrete values for the attribute it asked about. Clicking one sends the
    // simulator's own disclosure frame, which the parser reads exactly.
    //
    // In the recorded build only one continuation exists, so offering ten
    // values that mostly dead-end would be a worse lie than offering the one
    // that works. The others are still listed, greyed, so the real disclosure
    // surface is visible.
    const next = cannedNext();
    const opts = STATIC_SRC
      ? (next
          ? `<button class="chip" type="button" data-say="${esc(next)}">${esc(next)}</button>`
          : "") +
        (data.options || []).slice(0, 5).map((value) =>
          `<i class="chip is-unrecorded" title="Not recorded in this build">${esc(value)}</i>`
        ).join("")
      : (data.options || []).map((value) =>
          `<button class="chip" type="button" data-say="For that, what matters is: ${esc(value)}.">${esc(value)}</button>`
        ).join("");

    wrap.innerHTML = `
      <div class="turn-badge">
        turn ${data.turn} · <b>${esc(data.track)}</b>
        · spec ${num(data.specificity, 2)}
        ${asking ? ` · asks <b>${esc(data.ask_attribute)}</b>` : " · no question left"}
      </div>
      <div class="bubble${asking ? " is-clarify" : ""}">
        ${esc(data.message)}
        ${sharedChips ? `<p class="shared-why">
            <span class="mono">all ${cards.length} match</span>${sharedChips}</p>` : ""}
        ${rows ? `<ol class="results-inline${probe ? " is-probe" : ""}">${rows}</ol>` : ""}
        ${(data.results || []).length > shown
          ? `<p class="more mono">+ ${data.results.length - shown} more returned</p>` : ""}
        ${opts ? `<div class="opt-row"><span class="opt-label mono">${
          STATIC_SRC ? "continue" : "or pick one"}</span>${opts}</div>` : ""}
      </div>`;
    el.thread.append(wrap);
    scrollThread();
  }

  function addSmallTalkTurn(data) {
    el.empty?.remove();
    const wrap = document.createElement("div");
    wrap.className = "turn turn-agent turn-chat";
    const chips = (data.chips || []).map((c) => (
      c.action === "reset"
        ? `<button class="chip" type="button" data-reset-search>${esc(c.label)}</button>`
        : `<button class="chip" type="button" data-say="${esc(c.say)}">${esc(c.label)}</button>`
    )).join("");
    wrap.innerHTML = `
      <div class="turn-badge">
        <span class="chat-tag">chat</span> · ${esc(String(data.intent).replace("_", " "))}
        · <b>no turn spent</b> · ${data.turns_remaining > 0
            ? `still ${data.turns_remaining} of 10 left`
            : "the 10-turn budget is already spent"}
      </div>
      <div class="bubble is-chat">
        ${esc(data.message)}
        ${chips ? `<div class="opt-row"><span class="opt-label mono">try</span>${chips}</div>` : ""}
      </div>`;
    el.thread.append(wrap);
    scrollThread();
  }

  function addBudgetNotice(data) {
    el.empty?.remove();
    const wrap = document.createElement("div");
    wrap.className = "turn turn-agent turn-chat is-budget";
    wrap.innerHTML = `
      <div class="turn-badge">
        <span class="chat-tag">limit</span> · turn budget spent · <b>0 of 10 left</b>
      </div>
      <div class="bubble is-chat is-budget">
        ${esc(data.message)}
        <div class="opt-row">
          <span class="opt-label mono">next</span>
          <button class="chip" type="button" data-reset-search>start a new search</button>
        </div>
      </div>`;
    el.thread.append(wrap);
    scrollThread();
  }

  function scrollThread() {
    el.thread.scrollTo({ top: el.thread.scrollHeight, behavior: calm ? "auto" : "smooth" });
  }

  /* ── State panels ────────────────────────────────────────────────── */

  function renderRouting(data) {
    // The needle maps specificity 0..1 onto the meter, and the zone widths in
    // CSS mirror src/route.py's 0.30 / 0.55 thresholds.
    const pct = Math.max(3, Math.min(97, (data.specificity ?? 0) * 100));
    el.needle.style.left = `${pct}%`;
    el.needle.dataset.track = data.track;
    el.trackLabel.textContent = `${data.track} · ${num(data.specificity, 2)}`;

    const notes = {
      buy: "Constraints are firm, so exact phrase matching dominates and diversity is switched off.",
      browse: "Still exploring — cosine similarity carries the ranking and MMR spreads the results.",
      blend: "The evidence does not separate the tracks yet, so it hedges rather than committing.",
    };
    el.trackNote.textContent = notes[data.track] || "";

    const w = data.weights || {};
    el.weightList.innerHTML = Object.entries(w)
      .filter(([, v]) => v > 0)
      .map(([k, v]) => `<li><span>${esc(k)}</span><b>${v.toFixed(2)}</b></li>`)
      .join("");
  }

  function renderConstraints(c) {
    if (!c) return;
    const values = {
      category: c.category
        ? c.category + (c.category_exact ? ` (${c.category_pool.toLocaleString()} products)` : " — not a catalog category")
        : null,
      scenario: c.scenario_label,
      ruled_out: c.ruled_out ? `${c.ruled_out} products` : null,
      exhausted: (c.exhausted || []).join(", ") || null,
    };

    for (const [key, val] of Object.entries(values)) {
      const row = $(`.slot[data-slot="${key}"]`, el.slots);
      if (!row) continue;
      const dd = $("dd", row);
      const text = val || "—";
      row.classList.toggle("is-set", Boolean(val));
      if (text !== lastSlots[key] && val) {
        row.classList.remove("just-changed");
        void row.offsetWidth; // restart the flash animation
        row.classList.add("just-changed");
      }
      dd.textContent = text;
      dd.title = text;
      lastSlots[key] = text;
    }

    const phrases = c.phrases || [];
    el.slotCount.textContent = String(phrases.length);
    el.phraseCloud.innerHTML = phrases.length
      ? phrases.map((p) => `
          <span class="phrase" title="evidence weight ${p.weight}">
            ${esc(p.text)}<b class="mono">${p.weight.toFixed(2)}</b>
          </span>`).join("")
      : '<p class="state-note">Nothing disclosed yet.</p>';
  }

  function renderGains(data) {
    const gains = data.gains || {};
    const entries = Object.entries(gains);
    el.askLabel.textContent = data.ask_attribute || "none";

    if (!entries.length) {
      el.gains.innerHTML = '<li class="muted">No attribute can teach it anything more.</li>';
      return;
    }
    const top = Math.max(...entries.map(([, v]) => v), 0.001);
    el.gains.innerHTML = entries.map(([name, value]) => {
      const chosen = name === data.ask_attribute;
      return `
        <li class="${chosen ? "is-chosen" : ""}">
          <span>${esc(name)}</span>
          <i style="width:${Math.max(2, (value / top) * 100)}%"></i>
          <b class="mono">${value.toFixed(2)}</b>
        </li>`;
    }).join("");
  }

  function renderDecision(d) {
    if (!d) return;
    el.decisionMode.textContent = d.mode === "probe" ? "probing" : "full list";
    el.decisionMode.style.color = d.mode === "probe" ? "var(--amber)" : "var(--mint)";
    el.decisionHead.textContent = d.headline || "—";
    el.decisionWhy.textContent = d.why || "";
  }

  function renderRetrieval(trace) {
    if (!trace) return;
    const rows = [
      ["pool", (trace.pool ?? 0).toLocaleString()],
      ["in category", (trace.in_bucket ?? 0).toLocaleString()],
      ["phrase hits", (trace.phrase_docs ?? 0).toLocaleString()],
      ["excluded", (trace.excluded ?? 0).toLocaleString()],
      ["margin", num(trace.margin, 3)],
    ];
    el.retrievalKv.innerHTML = rows.map(([k, v]) =>
      `<li><span>${esc(k)}</span><b title="${esc(v)}">${esc(v)}</b></li>`).join("");
  }

  function renderComparison(ours, base) {
    el.oursResults.innerHTML =
      (ours?.results || []).slice(0, 6).map((r) => productRow(r)).join("")
      || '<li class="muted">No results.</li>';
    el.oursTag.textContent = ours?.track ? `${ours.track} track` : "—";

    if (!base) return;
    if (!base.ready) {
      el.baseTag.textContent = "indexing…";
      el.baseResults.innerHTML =
        `<li class="muted">${esc(base.note || "The official starter is still indexing.")}</li>`;
      return;
    }
    el.baseTag.textContent = "no state";
    // Items the baseline shares with us are dimmed, so the divergence stands
    // out rather than the overlap.
    const oursIds = new Set((ours?.results || []).map((r) => r.parent_asin));
    el.baseResults.innerHTML = (base.results || []).slice(0, 6)
      .map((r) => productRow(r, oursIds.has(r.parent_asin) ? "is-shared" : ""))
      .join("") || '<li class="muted">No results.</li>';
  }

  /* ── The suggestion panel ────────────────────────────────────────── */
  /* Two jobs, and they are not the same job.
   *
   * Before a session has a category, the catalog is 50,000 Clothing, Shoes &
   * Jewelry products and nothing else, so an unguided box invites queries
   * nothing can answer ("headphones") and the page looks broken while behaving
   * correctly. Here it offers only categories that exist, shows the exact
   * opener each one becomes, and labels the ones it reached by inference
   * ("winter" -> gloves) rather than passing them off as literal matches.
   *
   * Once a category is locked the agent will not change it, so offering more
   * categories would invite a topic change nobody asked for. From that point
   * the panel switches to the constraints the *live candidates* can still
   * disclose — the same values the agent computes for its own question — so
   * every row teaches it something.
   */

  let taRows = [];          // [{ kind, say, label, meta, hint }]
  let taNote = "";          // the note above the rows, kept so arrowing through
                            // the list does not have to read it back out of
                            // the DOM and silently lose it
  let taIndex = -1;
  let taTimer = null;
  let taSeq = 0;
  let sessionLocked = false;  // set once the agent holds a real category

  function closeTypeahead() {
    // Cancel any in-flight debounce too, or a keystroke 100ms before Enter
    // reopens the list over the reply that just arrived.
    clearTimeout(taTimer);
    taSeq += 1;
    el.typeahead.hidden = true;
    el.msg.setAttribute("aria-expanded", "false");
    el.msg.removeAttribute("aria-activedescendant");
    taRows = [];
    taNote = "";
    taIndex = -1;
  }

  function renderTypeahead(header = taNote) {
    taNote = header || "";
    if (!taRows.length && !taNote) { closeTypeahead(); return; }
    const rows = taRows.map((row, i) => {
      const active = i === taIndex;
      return `
        <button class="ta-item${active ? " is-active" : ""}" type="button"
                role="option" id="ta-opt-${i}" aria-selected="${active}"
                data-ta="${i}"${row.title ? ` title="${esc(row.title)}"` : ""}>
          <span class="ta-main">
            <span class="ta-label">${row.label}</span>
            ${row.hint ? `<span class="ta-hint">${row.hint}</span>` : ""}
          </span>
          <span class="ta-side">
            ${row.meta ? `<b class="mono ta-meta">${esc(row.meta)}</b>` : ""}
            ${row.aside ? `<i class="mono ta-aside">${esc(row.aside)}</i>` : ""}
          </span>
        </button>`;
    }).join("");
    el.typeahead.innerHTML = taNote + rows;
    el.typeahead.hidden = false;
    el.msg.setAttribute("aria-expanded", "true");
    if (taIndex >= 0) el.msg.setAttribute("aria-activedescendant", `ta-opt-${taIndex}`);
    else el.msg.removeAttribute("aria-activedescendant");
  }

  /* Header block: says what the list is, or why there isn't one. */
  function taHeader(text, tone) {
    if (!text) return "";
    return `<p class="ta-note${tone ? ` is-${tone}` : ""}">${text}</p>`;
  }

  /* Before a category: catalog categories, grouped by how they were reached. */
  async function refreshCategories(value) {
    const seq = ++taSeq;
    const data = await get(`/categories?q=${encodeURIComponent(value)}`);
    if (seq !== taSeq || busy) return;

    const cats = data.categories || [];
    taRows = cats.map((c) => {
      // What is actually on the shelf, not just that the shelf exists.
      // "Accessories Belts, 258" and "Belt Buckles, 18" are equally plausible
      // to someone who has seen neither, and picking wrong costs a session —
      // the agent locks its category on turn one.
      const traits = (c.traits || []).join(" · ");
      const band = c.price_low != null && c.price_high != null
        ? `${money(c.price_low)}–${money(c.price_high)}` : "";
      return {
        kind: "category",
        say: c.opener,
        label: c.via
          ? `${esc(c.category)} <em class="ta-via">via “${esc(c.via)}”</em>`
          : esc(c.category),
        meta: c.count.toLocaleString(),
        hint: traits ? `mostly ${esc(traits)}` : "",
        aside: [band, c.rating ? `${num(c.rating, 1)}★` : ""].filter(Boolean).join(" · "),
        title: [
          c.requirement ? `opens as a buying search — ${c.requirement}`
                        : "opens open-ended",
          band ? `${band} across ${c.priced.toLocaleString()} priced products` : "",
        ].filter(Boolean).join("\n"),
      };
    });
    taIndex = -1;

    let header = "";
    if (data.small_talk) {
      // "hi" is a greeting that happens to prefix "Hiking Boots". Lead with the
      // greeting; keep the categories underneath as a did-you-mean.
      header = taHeader(
        `Press Enter to say <b>${esc(data.query)}</b> — I'll answer, and it won't `
        + `spend a turn.${cats.length ? " Or pick a category:" : ""}`, "chat");
    } else if (data.note) {
      header = taHeader(esc(data.note), cats.length ? "warn" : "empty");
    } else if (!value) {
      header = taHeader("Biggest categories in the catalog — or keep typing.");
    } else if (Object.keys(data.corrections || {}).length) {
      const fixes = Object.entries(data.corrections)
        .map(([from, to]) => `<s>${esc(from)}</s> ${esc(to)}`).join(", ");
      header = taHeader(`Reading that as ${fixes}.`, "warn");
    }
    renderTypeahead(header);
  }

  /* After a category: the constraints the live candidates can still disclose. */
  async function refreshRefinements(value) {
    const seq = ++taSeq;
    const data = await get(
      `/refinements?session_id=${encodeURIComponent(sessionId)}&q=${encodeURIComponent(value)}`);
    if (seq !== taSeq || busy) return;
    if (!data.ready) { sessionLocked = false; return refreshCategories(value); }

    const values = (data.values || []).map((v) => ({
      kind: "value",
      say: v.say,
      label: esc(v.value),
      meta: v.attribute,
      hint: null,
    }));

    // Typing a *different* product mid-session is not a refinement, and the
    // agent locks its category for the life of a session — so this cannot be
    // answered by narrowing. Offering the switch openly beats silently ranking
    // belts for someone who just typed "sneakers".
    let switches = [];
    if (value && !values.length) {
      const alt = await get(`/categories?q=${encodeURIComponent(value)}`);
      if (seq !== taSeq) return;
      switches = (alt.categories || [])
        // A two-product campaign slice is not somewhere to restart a session.
        .filter((c) => !c.noise && !c.thin)
        .slice(0, 4)
        .map((c) => ({
          kind: "switch",
          say: c.opener,
          label: `<span class="ta-action">start over in</span> ${esc(c.category)}`,
          meta: c.count.toLocaleString(),
          hint: "drops the current search",
        }));
    }

    taRows = [
      ...values,
      ...switches,
      ...(data.actions || []).filter((a) => a.say).map((a) => ({
        kind: "action",
        say: a.say,
        label: `<span class="ta-action">${esc(a.label)}</span>`,
        meta: null,
        hint: null,
      })),
    ];
    taIndex = -1;
    let header;
    if (values.length) {
      header = `Constraints <b>${esc(data.category)}</b> can still disclose — every one `
        + `narrows the pool${data.asked ? `, and the first few answer the `
        + `<b>${esc(data.asked)}</b> question it just asked` : ""}.`;
    } else if (switches.length) {
      header = `Nothing in <b>${esc(data.category)}</b> matches that, and it reads `
        + `as a different product — which means a new search.`;
    } else if (value) {
      header = `No constraint in <b>${esc(data.category)}</b> mentions `
        + `&ldquo;${esc(value)}&rdquo;, and it names no other category either. `
        + `Send it anyway and it re-ranks on the raw text.`;
    } else {
      header = `Nothing left to disclose in <b>${esc(data.category)}</b>. Say anything `
        + `and it re-ranks on the text.`;
    }
    renderTypeahead(taHeader(header, values.length ? null : "warn"));
  }

  async function refreshTypeahead() {
    const value = el.msg.value.trim();
    // A fully-formed customer frame is not a search — leave it alone.
    if (/^(i'?m looking for|for that|actually|i don'?t have)/i.test(value)) {
      closeTypeahead();
      return;
    }
    try {
      if (sessionLocked) await refreshRefinements(value);
      else await refreshCategories(value);
    } catch {
      closeTypeahead();
    }
  }

  function chooseTypeahead(index) {
    const row = taRows[index];
    if (!row) return;
    closeTypeahead();
    send(row.say);
  }

  const scheduleTypeahead = (delay = 120) => {
    clearTimeout(taTimer);
    taTimer = setTimeout(refreshTypeahead, delay);
  };

  /* Focus we gave ourselves, rather than focus the reader asked for. Without
     this the panel reopened over every answer the moment it arrived, because
     `send` refocuses the box when a turn finishes. */
  let quietFocus = false;
  function refocusQuietly() {
    quietFocus = true;
    el.msg.focus({ preventScroll: true });
    // Cleared on a timer rather than in the handler: `focus` does not fire at
    // all if the box already had focus, and a flag left set would swallow the
    // reader's next genuine click into it.
    setTimeout(() => { quietFocus = false; }, 0);
  }

  el.msg?.addEventListener("input", () => { if (!STATIC_SRC) scheduleTypeahead(); });
  // Opening the box on focus is the difference between a search bar you have
  // to already know the answer to and one that tells you what it holds.
  el.msg?.addEventListener("focus", () => {
    if (quietFocus || STATIC_SRC) return;
    scheduleTypeahead(0);
  });
  el.msg?.addEventListener("blur", () => setTimeout(closeTypeahead, 150));
  document.addEventListener("pointerdown", (e) => {
    if (!e.target.closest(".composer-wrap")) closeTypeahead();
  });
  el.typeahead?.addEventListener("mousedown", (e) => {
    const item = e.target.closest("[data-ta]");
    // mousedown, not click: blur would close the list first.
    if (item) { e.preventDefault(); chooseTypeahead(Number(item.dataset.ta)); }
  });

  /* ── Turn flow ───────────────────────────────────────────────────── */

  let started = false;

  async function ensureSession() {
    if (started) return;
    await post("/reset", {
      session_id: sessionId,
      profile: PROFILES[profileKey].profile,
      mode: modeKey,
    });
    started = true;
  }

  async function send(text) {
    if (busy || !text.trim()) return;
    busy = true;
    closeTypeahead();
    el.send.disabled = true;
    setStatus("thinking");
    addUserTurn(text);
    el.msg.value = "";

    // Pin the session this turn belongs to. "New session" while a request is in
    // flight used to clear the thread and then have the old reply land in it,
    // against a session id the server had already forgotten.
    const forSession = sessionId;
    const t0 = performance.now();
    try {
      await ensureSession();
      // The baseline call is best-effort: it is a comparison aid, and the
      // conversation must work whether or not the starter has finished
      // indexing.
      const [data, base] = await Promise.all([
        post("/chat", { session_id: forSession, message: text, assist: true }),
        post("/baseline", { session_id: forSession, message: text }).catch(() => null),
      ]);
      if (forSession !== sessionId) return;

      // A greeting is not a search. It costs no turn, touches no state, and
      // leaves every panel showing what the last real turn showed.
      if (data.kind === "small_talk") {
        addSmallTalkTurn(data);
        el.latency.textContent = `${Math.round(performance.now() - t0)} ms · no turn spent`;
        setStatus("ready");
        return;
      }
      // The ten-turn cap is enforced by the server, not by a disabled text box.
      if (data.kind === "budget_spent") {
        addBudgetNotice(data);
        el.latency.textContent = "";
        setStatus("ready");
        return;
      }

      el.latency.textContent =
        `${data.latency_ms} ms agent · ${Math.round(performance.now() - t0)} ms round trip`;
      if (data.rewrote_as) {
        addRewriteNote(text, data.rewrote_as);
      }
      if (data.assist) {
        addAssistNote(data.assist);
      } else if (data.unmatched?.length) {
        addUnmatchedNote(data.unmatched, data.constraints.category);
      } else if (data.constraints && !data.constraints.category_exact) {
        // Every turn, not just the first. A session that never resolved a
        // category never will — the agent locks the slot — so going quiet
        // after turn 1 leaves you watching it answer a question it cannot
        // answer, which is what makes it feel like it is repeating itself.
        addScopeWarning(data.constraints.category || text, data.turn);
      }
      addAgentTurn(data);
      // The detail panel marks which of a product's constraints this session
      // has already heard, so a turn invalidates every cached card.
      similarCache.clear();
      renderRouting(data);
      renderConstraints({ ...data.constraints, scenario_label: data.scenario });
      renderGains(data);
      renderDecision(data.decision);
      renderProfilePanel(data);
      renderRetrieval(data.trace);
      renderComparison(data, base);
      el.turns.textContent = `turn ${data.turn} / 10`;
      el.turns.classList.toggle("is-spent", data.turns_remaining <= 0);
      // Once the agent holds a real category it will not change it, so the
      // suggestion panel stops offering categories and starts offering the
      // constraints this pool can still disclose.
      sessionLocked = Boolean(data.constraints?.category_exact);
      setStatus("ready");

      if (data.turns_remaining <= 0) {
        // The box stays usable: conversation is free, and being unable to say
        // "thanks, that's it" is a worse ending than being told the budget is
        // spent when you try to search again.
        el.msg.placeholder = "Turn budget spent — say anything, or start a new session.";
      }
    } catch (err) {
      if (forSession !== sessionId) return;
      setStatus("error");
      const wrap = document.createElement("div");
      wrap.className = "turn turn-agent";
      wrap.innerHTML = err.notRecorded
        ? `<div class="bubble is-error">This is the recorded build, so it can only
             replay conversations that were captured from the live agent — it cannot
             answer a new sentence. Pick one of the conversations below, or clone the
             repository and run <code>make serve</code> to type anything you like.
             <div class="opt-row"><span class="opt-label mono">recorded</span>
             ${(CANNED?.scripts || []).map((sc) =>
               `<button class="chip" type="button" data-say="${esc(sc.recorded[0].said)}"
                  >${esc(sc.label)}</button>`).join("")}</div></div>`
        : `<div class="bubble is-error">Couldn't reach the agent: ${esc(err.message)}. ` +
          `Check that <code>python3 server.py</code> is running and the catalog exists.</div>`;
      el.thread.append(wrap);
      scrollThread();
    } finally {
      if (forSession === sessionId) {
        busy = false;
        el.send.disabled = false;
        // Refocus without reopening the suggestion panel over the answer that
        // just arrived: the focus is ours, not the reader's.
        refocusQuietly();
      }
    }
  }

  async function resetSession() {
    sessionId = newSession();
    started = false;
    sessionLocked = false;
    lastSlots = {};
    similarCache.clear();
    closeTypeahead();
    el.thread.innerHTML = "";
    if (el.empty) el.thread.append(el.empty);
    el.msg.value = "";
    el.msg.disabled = false;
    el.msg.placeholder = "Type a category, an occasion, or just say hi …";
    el.turns.classList.remove("is-spent");
    hideSimilar();
    el.turns.textContent = "turn 0 / 10";
    el.latency.textContent = "";
    el.oursResults.innerHTML = '<li class="muted">No results yet.</li>';
    el.baseResults.innerHTML = '<li class="muted">No results yet.</li>';
    el.gains.innerHTML = "";
    el.askLabel.textContent = "—";
    el.slotCount.textContent = "0";
    el.phraseCloud.innerHTML = '<p class="state-note">Nothing disclosed yet.</p>';
    renderConstraints({ phrases: [] });
    setStatus("idle");
    el.msg.focus();
  }

  /* ── Profiles ────────────────────────────────────────────────────── */

  function renderProfiles() {
    el.profileChips.innerHTML = Object.entries(PROFILES).map(([key, p]) =>
      `<button class="chip${key === profileKey ? " is-on" : ""}" type="button"
               data-profile="${esc(key)}">${esc(p.label)}</button>`).join("");
    el.profileSummary.textContent = PROFILES[profileKey].summary;
    el.modeChips.innerHTML = Object.entries(MODES).map(([key, m]) =>
      `<button class="chip${key === modeKey ? " is-on" : ""}" type="button"
               data-mode="${esc(key)}">${esc(m.label)}</button>`).join("");
    el.modeSummary.textContent = MODES[modeKey].summary;
  }

  function renderProfilePanel(data) {
    const tags = data.profile_tags || [];
    if (data.profile_weight != null) profileWeight = data.profile_weight;
    el.profileWeight.textContent = tags.length ? `weight ${data.profile_weight}` : "off";
    if (!tags.length) {
      el.profilePanel.innerHTML =
        '<p class="state-note">No profile selected, so ranking uses only what you said.</p>';
      return;
    }
    const hits = (data.results || []).filter((r) => (r.matched_tags || []).length).length;
    el.profilePanel.innerHTML = `
      <div class="phrase-cloud">
        ${tags.map((t) => `<span class="phrase is-tag">${esc(t)}<b class="mono">${data.profile_weight}</b></span>`).join("")}
      </div>
      <p class="state-note">
        These enter the query vector at ${data.profile_weight} against 1.00 for anything you
        state, so they break ties and can never overrule you.
        <b>${hits} of ${(data.results || []).length}</b> returned products carry one in their own text.
        Measured over 200 sessions the whole layer is worth −0.0002 of score — kept because a
        personalisation layer that <em>could</em> overrule a stated constraint would be a worse
        product even if it scored better.
      </p>`;
  }

  /* ── Replay ──────────────────────────────────────────────────────── */

  async function loadSessions() {
    try {
      const { sessions } = await get("/sessions");
      el.replayPick.innerHTML = sessions.map((s) =>
        `<option value="${esc(s.sample_id)}">${esc(s.sample_id)} · ${esc(s.scenario)} · ${esc(s.difficulty)}</option>`
      ).join("");
    } catch {
      el.replayPick.innerHTML = '<option value="">no sessions available</option>';
    }
  }

  function renderReplay(run) {
    const t = run.target;
    const verdict = run.hit
      ? `<b class="ok">converted on turn ${run.hit_turn} at rank ${run.hit_rank}</b>
         · reciprocal rank ${run.reciprocal_rank}`
      : '<b class="bad">no conversion within 10 turns</b>';

    el.replayTarget.innerHTML = `
      <div class="panel-head">
        <h3>Hidden target</h3>
        <span class="mono">${esc(run.sample_id)} · ${esc(run.scenario)}</span>
      </div>
      <div class="target-body">
        <p class="target-title">${esc(t.title)}</p>
        <p class="mono target-meta">
          ${esc(t.parent_asin)} · ${money(t.price)} · ${num(t.rating, 1)}★ (${(t.rating_count || 0).toLocaleString()})
        </p>
        <div class="target-cards">
          <div>
            <h4>hard constraints</h4>
            <ul>${(t.hard_constraints || []).map((v) => `<li>${esc(v)}</li>`).join("") || "<li>—</li>"}</ul>
          </div>
          <div>
            <h4>soft preferences</h4>
            <ul>${(t.soft_preferences || []).map((v) => `<li>${esc(v)}</li>`).join("") || "<li>—</li>"}</ul>
          </div>
        </div>
        <p class="verdict">${verdict}</p>
        <p class="state-note">${esc(run.profile?.summary || "")}</p>
      </div>`;

    el.replayTurns.innerHTML = run.turns.map((turn) => {
      const rows = (turn.results || []).slice(0, 5).map((r) => {
        const isTarget = r.parent_asin === t.parent_asin;
        return productRow(r, isTarget ? "is-target" : "");
      }).join("");
      return `
        <li class="replay-turn${turn.target_rank ? " is-hit" : ""}">
          <div class="rt-head">
            <span class="mono">turn ${turn.turn}</span>
            <span class="mono rt-track" data-track="${esc(turn.track)}">${esc(turn.track)}</span>
            ${turn.ask_attribute ? `<span class="mono rt-ask">asks ${esc(turn.ask_attribute)}</span>` : ""}
            <span class="mono rt-pool">pool ${(turn.trace?.pool ?? 0).toLocaleString()}</span>
            ${turn.decision?.mode === "probe" ? '<span class="mono rt-probe">probing</span>' : ""}
            ${turn.target_rank ? `<span class="mono rt-found">target at rank ${turn.target_rank}</span>` : ""}
          </div>
          <p class="rt-customer"><span>customer</span>${esc(turn.customer)}</p>
          <p class="rt-agent"><span>agent</span>${esc(turn.message)}</p>
          <ol class="results-inline">${rows}</ol>
        </li>`;
    }).join("");

    el.replayOut.hidden = false;
  }

  async function runReplay() {
    const sampleId = el.replayPick.value;
    if (!sampleId || el.replayRun.disabled) return;
    el.replayRun.disabled = true;
    el.replayStatus.textContent = "running the organizer's simulator…";
    try {
      const t0 = performance.now();
      const run = await post("/replay", { sample_id: sampleId });
      renderReplay(run);
      el.replayStatus.textContent =
        `${run.turns.length} turns in ${Math.round(performance.now() - t0)} ms`;
    } catch (err) {
      el.replayStatus.textContent = `failed: ${err.message}`;
    } finally {
      el.replayRun.disabled = false;
    }
  }

  /* ── Benchmark tables ────────────────────────────────────────────── */

  function fillHero(b) {
    const map = {
      score: num(b.ours.technical_score, 5),
      hr: num(b.ours.hit_rate_at_10, 3),
      mrr: num(b.ours.mrr, 4),
      mttc: num(b.ours.mttc, 3),
      tokens: b.ours.tokens == null ? "—" : b.ours.tokens.toLocaleString(),
    };
    for (const [key, value] of Object.entries(map)) {
      const node = $(`[data-stat="${key}"]`);
      if (node) node.textContent = value;
    }
  }

  function fillScenarios(b) {
    const body = $("#scenarioTable tbody");
    const rows = Object.entries(b.scenarios || {});
    if (!rows.length) { body.innerHTML = '<tr><td colspan="5" class="muted">No artifact.</td></tr>'; return; }
    const order = ["buying", "browsing", "intent_override", "boundary"];
    rows.sort((a, x) => order.indexOf(a[0]) - order.indexOf(x[0]));
    body.innerHTML = rows.map(([name, m]) => `
      <tr>
        <td>${esc(name.replace("_", " "))}</td>
        <td class="mono">${m.sample_count}</td>
        <td class="mono">${num(m.hit_rate_at_10, 3)}</td>
        <td class="mono">${num(m.mrr, 4)}</td>
        <td class="mono">${num(m.mttc, 2)}</td>
      </tr>`).join("");
  }

  function fillAblation(b) {
    const body = $("#ablationTable tbody");
    const arms = b.ablation || [];
    if (!arms.length) { body.innerHTML = '<tr><td colspan="3" class="muted">No artifact.</td></tr>'; return; }
    const full = arms.find((a) => a.arm === "full system")?.technical_score ?? 0;
    body.innerHTML = arms.map((a) => {
      const delta = a.technical_score - full;
      const isFull = a.arm === "full system";
      // Colour by magnitude, not sign: the point of the table is which rows
      // are outside the noise band at n=200.
      const cls = isFull ? "" : Math.abs(delta) >= 0.01 ? "big" : "small";
      return `
        <tr class="${isFull ? "is-full" : ""}">
          <td>${esc(a.arm)}</td>
          <td class="mono">${num(a.technical_score, 5)}</td>
          <td class="mono d-${cls}">${isFull ? "—" : (delta >= 0 ? "+" : "") + delta.toFixed(3)}</td>
        </tr>`;
    }).join("");
  }

  function fillRobust(b) {
    const body = $("#robustTable tbody");
    const rows = b.robustness || [];
    if (!rows.length) { body.innerHTML = '<tr><td colspan="3" class="muted">No artifact.</td></tr>'; return; }
    const byLevel = {};
    rows.forEach((r) => {
      byLevel[r.paraphrase] = byLevel[r.paraphrase] || {};
      byLevel[r.paraphrase][r.arm] = r.technical_score;
    });
    const order = ["none", "light", "medium", "heavy"];
    const label = { none: "verbatim (as scored)", light: "light paraphrase",
                    medium: "medium paraphrase", heavy: "heavy paraphrase" };
    body.innerHTML = order.filter((k) => byLevel[k]).map((k) => `
      <tr>
        <td>${esc(label[k])}</td>
        <td class="mono">${byLevel[k].unhardened == null ? "—" : num(byLevel[k].unhardened, 5)}</td>
        <td class="mono strong">${byLevel[k].hardened == null ? "—" : num(byLevel[k].hardened, 5)}</td>
      </tr>`).join("");
  }

  async function loadBenchmark() {
    try {
      const b = await get("/benchmark");
      if (b.ours?.technical_score != null) fillHero(b);
      fillScenarios(b);
      fillAblation(b);
      fillRobust(b);
    } catch {
      // The page is still useful without the committed artifacts — the tables
      // keep their "loading" row rather than showing invented numbers.
    }
  }

  async function loadStarters() {
    if (STATIC_SRC) {
      await loadCanned();
      el.chips.innerHTML = `
        <div class="starter-group">
          <span class="starter-label mono">recorded conversations — pick one</span>
          <div class="chips">${CANNED.scripts.map((sc) => `
            <button class="chip chip-natural" type="button"
                    data-say="${esc(sc.recorded[0].said)}">
              ${esc(sc.label)}<span class="chip-why">${esc(sc.why)}</span>
            </button>`).join("")}</div>
        </div>`;
      el.msg.placeholder = "Pick a recorded conversation above …";
      el.scopeNote.innerHTML =
        'Recorded from the live agent — <b>every turn below is a real response</b>. ' +
        'Clone the repo and run <code>make serve</code> to type your own.';
      return;
    }
    try {
      const { examples, natural } = await get("/suggestions");
      // Two rows, because they demonstrate different things. The first is how
      // the evaluator opens every scored session; the second is how a person
      // would say the same thing, which is what the assist layer is for.
      const human = (natural || []).slice(0, 4).map((n) => `
        <button class="chip chip-natural" type="button" data-say="${esc(n.text)}"
                title="resolves to ${esc(n.resolves_to)}${n.via ? ` — via “${esc(n.via)}”` : ""}">
          ${esc(n.text)}
          <span class="chip-why">${esc(n.why)}</span>
        </button>`).join("");
      const scored = (examples || []).slice(0, 2).map((e) => `
        <button class="chip chip-scored" type="button" data-say="${esc(e)}"
                title="${esc(e)}">${esc(e)}</button>`).join("");
      el.chips.innerHTML = `
        <div class="starter-group">
          <span class="starter-label mono">how a person says it</span>
          <div class="chips">${human}</div>
        </div>
        <div class="starter-group">
          <span class="starter-label mono">how the evaluator says it</span>
          <div class="chips">${scored}</div>
        </div>`;
    } catch {
      // Suggestions come from the live catalog, so a failure here just means
      // no chips — never a broken page.
    }
  }

  async function loadHealth() {
    try {
      const h = await get("/health");
      el.footMeta.textContent =
        `${h.catalog_size.toLocaleString()} products indexed in ${h.index_seconds}s · ` +
        `reranker: ${h.reranker} · scored headlessly — this page is the walkthrough, not the scored path.`;
    } catch { /* leave the static text */ }
  }

  /* ── Wiring ──────────────────────────────────────────────────────── */

  el.send?.addEventListener("click", () => send(el.msg.value));
  el.msg?.addEventListener("keydown", (e) => {
    const open = !el.typeahead.hidden && taRows.length;
    if (open && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      // -1 is "nothing highlighted", and it is a real position in the cycle:
      // arrowing past the last row returns you to what you actually typed.
      const step = e.key === "ArrowDown" ? 1 : -1;
      const span = taRows.length + 1;
      taIndex = ((taIndex + 1 + step) % span + span) % span - 1;
      renderTypeahead();
      el.typeahead.querySelector(".ta-item.is-active")
        ?.scrollIntoView({ block: "nearest" });
      return;
    }
    if (e.key === "Escape") {
      if (open) { e.preventDefault(); closeTypeahead(); return; }
      el.msg.value = "";
      return;
    }
    if (e.key === "Tab" && open && taIndex >= 0) {
      // Tab completes into the box instead of sending, so a suggestion can be
      // edited before it goes.
      e.preventDefault();
      el.msg.value = taRows[taIndex].say;
      closeTypeahead();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (open && taIndex >= 0) {
        chooseTypeahead(taIndex);
      } else {
        closeTypeahead();
        send(el.msg.value);
      }
    }
  });
  el.reset?.addEventListener("click", resetSession);
  el.replayRun?.addEventListener("click", runReplay);
  el.replayRandom?.addEventListener("click", () => {
    const options = el.replayPick.options;
    if (!options.length) return;
    el.replayPick.selectedIndex = Math.floor(Math.random() * options.length);
    runReplay();
  });

  // Delegated: chips exist both at load and inside rendered turns.
  document.addEventListener("click", (e) => {
    const say = e.target.closest("[data-say]");
    if (say) { send(say.dataset.say); return; }

    const profile = e.target.closest("[data-profile]");
    if (profile) {
      profileKey = profile.dataset.profile;
      renderProfiles();
      // The profile is handed to the agent at reset, so changing it starts a
      // fresh session rather than silently applying from the next turn.
      resetSession();
      return;
    }

    if (e.target.closest("[data-reset-search]")) { resetSession(); return; }

    const mode = e.target.closest("[data-mode]");
    if (mode) {
      modeKey = mode.dataset.mode;
      renderProfiles();
      resetSession();
    }
  });

  initCursor();
  initMagnetic();
  initTilt();
  initKinetic();
  initScrollProgress();
  initTicker();
  renderProfiles();
  loadStarters();
  loadSessions();
  loadBenchmark();
  loadHealth();
})();
