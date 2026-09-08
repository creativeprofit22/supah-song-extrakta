/* One controller shared by the local page and the speaker-free assertion page. */
"use strict";
window.mountBlindAB = function (root, options = {}) {
  const request = options.fetch || window.fetch.bind(window);
  const Context = Object.hasOwn(options, "AudioContext") ? options.AudioContext : window.AudioContext;
  const frame = options.requestAnimationFrame || window.requestAnimationFrame.bind(window);
  const cancel = options.cancelAnimationFrame || window.cancelAnimationFrame.bind(window);
  const events = options.window || window, page = options.document || document;
  const later = options.setTimeout || window.setTimeout.bind(window);
  const clear = options.clearTimeout || window.clearTimeout.bind(window);
  const get = (name) => root.querySelector(`[data-${name}]`);
  const play = get("play"), reveal = get("reveal"), status = get("status");
  const slots = [...root.querySelectorAll("[data-slot]")];
  let metadata, pcm, buffers, context, nodes = [], gains = [], selected = "A";
  let state = "loading", revealed = null, busy = false, revealing = false;
  // A lost POST response cannot prove blinding, even if an older GET says false.
  let revealAttempted = false, refreshing = null, refreshAbort, refreshTimer, deadline;
  const identityState = () => revealed === true ? "revealed" : revealAttempted ? "Reveal pending" :
    revealed === null ? "state unknown" : "blind";
  let startTime = 0, generation = 0, animation = 0, disposed = false;

  function show() {
    if (disposed) return;
    const text = `${state[0].toUpperCase() + state.slice(1)} / ${identityState()}${busy ? " / pending" : ""}.` +
      (state === "error" ? " Playback unavailable. Check browser support or restart the local session." : "");
    if (status.textContent !== text) status.textContent = text;
    play.disabled = busy || state === "loading" || state === "error";
    play.textContent = state === "playing" ? "Pause" : state === "ended" ? "Replay" : "Play";
    for (const button of slots) {
      const active = button.dataset.slot === selected;
      button.disabled = busy || state === "loading" || state === "error";
      button.setAttribute("aria-pressed", String(active));
      button.textContent = button.dataset.slot + (active ? " (selected)" : "");
    }
    reveal.disabled = state === "loading" || state === "error" || revealing;
  }
  function disconnect() {
    generation++;
    for (const node of nodes) { node.onended = null; try { node.stop(); } catch (_) {} node.disconnect(); }
    for (const gain of gains) gain.disconnect();
    nodes = []; gains = [];
    cancel(animation);
  }
  function fail() {
    if (disposed) return;
    state = "error";
    disconnect();
    if (context) context.close().catch(() => {});
    show();
  }
  async function response(route, init) {
    const result = await request(route, {cache: "no-store", credentials: "omit", ...init});
    if (!result.ok) throw new Error("Request failed");
    return result;
  }
  function refreshSession() {
    if (disposed) return Promise.resolve();
    if (refreshing) return refreshing;
    clear(refreshTimer);
    refreshAbort = new AbortController();
    deadline = later(() => refreshAbort.abort(), 5000);
    refreshing = (async () => {
      try {
        const next = await (await response("session", {signal: refreshAbort.signal})).json();
        if (disposed) return;
        if (next.sample_rate !== 48000 || next.channels !== 2 || next.loading !== false ||
            typeof next.revealed !== "boolean" || !Number.isInteger(next.frames) ||
            next.frames < 1 || next.frames > 2880000 ||
            !Number.isInteger(next.source_start_frame) || next.source_start_frame < 0 ||
            !Number.isInteger(next.source_end_frame) ||
            next.source_end_frame - next.source_start_frame !== next.frames ||
            (metadata && ["frames", "source_start_frame", "source_end_frame"].some((key) => next[key] !== metadata[key])))
          throw new Error("Invalid scope");
        if (!metadata) metadata = next;
        if (revealed !== true) revealed = next.revealed;
      } catch (_) {
        if (revealed !== true) revealed = null;
      } finally {
        clear(deadline);
        refreshing = null;
        if (!disposed) {
          show();
          if (page.visibilityState !== "hidden") refreshTimer = later(refreshSession, 2000);
        }
      }
    })();
    return refreshing;
  }
  function reconcile() {
    clear(refreshTimer);
    if (page.visibilityState !== "hidden") void refreshSession();
  }
  events.addEventListener("focus", reconcile);
  page.addEventListener("visibilitychange", reconcile);
  function position() {
    if (disposed || state === "error") return;
    const seconds = state === "ended" ? metadata.frames / 48000 :
      nodes.length ? Math.max(0, Math.min(metadata.frames / 48000, context.currentTime - startTime)) : 0;
    const sourceFrame = Math.min(metadata.source_end_frame,
      metadata.source_start_frame + Math.floor(seconds * 48000));
    get("position").textContent = `Elapsed: ${seconds.toFixed(3)} s / ${(metadata.frames / 48000).toFixed(3)} s; source frame ${sourceFrame}`;
    if (state === "playing") animation = frame(position);
  }
  function switchSlot(slot) {
    if (busy || !metadata || state === "error" || state === "loading") return;
    selected = slot;
    if (gains.length) {
      const now = context.currentTime;
      gains.forEach((gain, index) => gain.gain.setValueAtTime((index === 0 ? "A" : "B") === selected ? 1 : 0, now));
    }
    show();
  }
  async function toggle() {
    if (busy || disposed || state === "loading" || state === "error") return;
    busy = true; show();
    try {
      if (!context) {
        if (!Context) throw new Error("Web Audio unavailable");
        context = new Context({sampleRate: 48000});
        if (context.sampleRate !== 48000) throw new Error("Unsupported rate");
        buffers = pcm.map((samples) => {
          const buffer = context.createBuffer(2, metadata.frames, 48000);
          for (let channel = 0; channel < 2; channel++) {
            const output = buffer.getChannelData(channel);
            for (let i = 0; i < metadata.frames; i++) output[i] = samples[i * 2 + channel];
          }
          return buffer;
        });
        pcm = null;
      }
      if (state === "playing") {
        await context.suspend();
        if (state !== "ended") state = "paused";
      } else {
        await context.resume();
        if (disposed || state === "error") return;
        if (state !== "paused") {
          disconnect();
          const currentGeneration = generation;
          gains = buffers.map(() => context.createGain());
          nodes = buffers.map((buffer, i) => {
            const node = context.createBufferSource(); node.buffer = buffer;
            node.connect(gains[i]); gains[i].connect(context.destination);
            return node;
          });
          const now = context.currentTime;
          startTime = now + 0.03;
          nodes.forEach((node, i) => {
            gains[i].gain.setValueAtTime((i === 0 ? "A" : "B") === selected ? 1 : 0, now);
            node.onended = () => {
              if (generation !== currentGeneration || state === "ended" || disposed) return;
              state = "ended"; cancel(animation); position(); show();
            };
            node.start(startTime, 0, metadata.frames / 48000);
          });
        }
        state = "playing";
      }
      cancel(animation); position();
    } catch (_) { fail(); }
    finally { busy = false; if (state !== "error") show(); }
  }
  async function revealIdentities() {
    if (revealing || disposed || state === "loading" || state === "error") return;
    revealing = true; revealAttempted = true; show();
    try {
      const identities = await (await response("reveal", {method: "POST"})).json();
      if (disposed) return;
      revealed = true;
      get("identities").textContent = JSON.stringify(identities, null, 2);
      reveal.textContent = "Identities revealed";
    } catch (_) { if (!disposed) void refreshSession(); }
    finally { revealing = false; show(); }
  }
  play.addEventListener("click", toggle);
  reveal.addEventListener("click", revealIdentities);
  slots.forEach((button) => button.addEventListener("click", () => switchSlot(button.dataset.slot)));
  const ready = (async () => {
    try {
      await refreshSession();
      if (disposed) return;
      if (!metadata) throw new Error("Session unavailable");
      get("scope").textContent = `Source frames [${metadata.source_start_frame}, ${metadata.source_end_frame}) / 48 kHz stereo`;
      pcm = await Promise.all(["A", "B"].map(async (slot) => {
        const bytes = await (await response(slot)).arrayBuffer();
        if (bytes.byteLength !== metadata.frames * 8) throw new Error("Incomplete audio");
        const view = new DataView(bytes), samples = new Float32Array(metadata.frames * 2);
        for (let i = 0; i < samples.length; i++) {
          samples[i] = view.getFloat32(i * 4, true);
          if (!Number.isFinite(samples[i])) throw new Error("Invalid samples");
        }
        return samples;
      }));
      await refreshSession();
      if (disposed) return;
      if (!Context) throw new Error("Web Audio unavailable");
      state = "ready"; show(); position();
    } catch (_) { fail(); }
  })();
  return {ready, toggle, reveal: revealIdentities, switchSlot, dispose() {
    disposed = true;
    clear(refreshTimer); clear(deadline); refreshAbort?.abort();
    events.removeEventListener("focus", reconcile);
    page.removeEventListener("visibilitychange", reconcile);
    disconnect(); if (context) context.close().catch(() => {});
  }};
};
const blindRoot = document.querySelector("[data-blind-ab]");
if (blindRoot) {
  const controller = window.mountBlindAB(blindRoot);
  window.addEventListener("pagehide", () => controller.dispose(), {once: true});
}
