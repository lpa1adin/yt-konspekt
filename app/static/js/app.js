/* YT-Конспект: живые карточки задач (SSE), выбор папки, экспорт, редактор, тема. Без сборки. */
(function () {
  "use strict";

  const STAGES = {
    info: "данные о видео", subtitles: "субтитры", audio: "скачиваю аудио",
    transcribing: "распознаю речь", summarizing: "пишу конспект", exporting: "сохраняю файл",
  };
  const STATUSES = { queued: "в очереди", running: "в работе", done: "готово", failed: "ошибка", cancelled: "отменено" };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  async function api(url, body, method = "POST") {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const r = await fetch(url, opts);
    let data = {};
    try { data = await r.json(); } catch (e) { /* пустой ответ */ }
    if (!r.ok && data.ok === undefined) data.ok = false;
    return data;
  }

  // ───────── тосты ─────────
  function toast(text, opts = {}) {
    const box = $("#toasts");
    if (!box) return;
    const el = document.createElement("div");
    el.className = "toast" + (opts.error ? " toast--error" : "");
    const span = document.createElement("span");
    span.textContent = text;
    el.appendChild(span);
    (opts.actions || []).forEach(([label, fn]) => {
      const b = document.createElement("button");
      b.className = "link-btn"; b.type = "button"; b.textContent = label;
      b.addEventListener("click", fn);
      el.appendChild(b);
    });
    box.appendChild(el);
    setTimeout(() => el.remove(), opts.timeout || 7000);
  }
  window.ytkToast = toast;

  // ───────── тема ─────────
  $("#theme-toggle")?.addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    localStorage.setItem("theme", root.dataset.theme);
  });

  // ───────── форма ссылок: Enter — отправить, Shift+Enter — новая строка ─────────
  const urls = $("#urls");
  if (urls) {
    urls.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        if (urls.value.trim()) urls.form.requestSubmit();
      }
    });
    document.body.addEventListener("jobs-changed", () => { urls.value = ""; });
  }

  // ───────── живые карточки задач через SSE ─────────
  function refreshCard(id) {
    const el = document.getElementById("job-" + id);
    if (!el || !window.htmx) return;
    htmx.ajax("GET", "/partials/jobs/" + id, { target: el, swap: "outerHTML" });
  }

  function applyEvent(ev) {
    if (ev.type === "pull") return onPull(ev);
    const el = document.getElementById("job-" + ev.job_id);
    if (!el) {
      if ($("#jobs") && window.htmx) htmx.trigger(document.body, "jobs-changed");
      return;
    }
    if (el.dataset.status !== ev.status || (ev.waiting && !el.classList.contains("job--waiting"))) {
      const wrap = el.closest("[data-reload-on-done]");
      if (wrap && ev.status === "done") { location.reload(); return; }
      refreshCard(ev.job_id);
      return;
    }
    const bar = $('[data-role="bar"]', el);
    if (bar) bar.style.width = Math.round((ev.progress || 0) * 100) + "%";
    const stage = $('[data-role="stage"]', el);
    if (stage) stage.textContent = STAGES[ev.stage] || "";
    const msg = $('[data-role="message"]', el);
    if (msg && ev.message) msg.textContent = ev.message;
    const pill = $('[data-role="status"]', el);
    if (pill) pill.textContent = STATUSES[ev.status] || ev.status;
    const stream = $('[data-role="stream"]', el);
    if (stream) {
      if (ev.text && ev.stage === "summarizing") {
        stream.hidden = false;
        stream.textContent = ev.text;
        stream.scrollTop = stream.scrollHeight;
      } else if (ev.stage !== "summarizing") {
        stream.hidden = true;
      }
    }
  }

  let es = null;
  function connect() {
    if (!window.EventSource) return;
    es = new EventSource("/api/events");
    es.onmessage = (m) => { try { applyEvent(JSON.parse(m.data)); } catch (e) { console.warn(e); } };
    es.addEventListener("hello", () => {
      // переподключились — источник истины таблица jobs: перечитываем карточки
      $$("[data-job-id]").forEach((el) => {
        if (["queued", "running"].includes(el.dataset.status)) refreshCard(el.dataset.jobId);
      });
    });
  }
  connect();

  // ───────── выбор и проверка папки ─────────
  function fieldMsg(id, text, ok) {
    const m = $(`.field-msg[data-for="${id}"]`);
    if (m) { m.textContent = text || ""; m.classList.toggle("ok", !!ok); }
  }

  document.addEventListener("click", async (e) => {
    const pick = e.target.closest("[data-pick-folder]");
    if (pick) {
      const input = document.getElementById(pick.dataset.pickFolder);
      pick.disabled = true;
      fieldMsg(input.id, "открываю окно выбора папки…", true);
      const r = await api("/api/pick-folder", { initial: input.value });
      pick.disabled = false;
      if (r.path) {
        input.value = r.path;
        fieldMsg(input.id, r.ok ? "" : r.error, r.ok);
      } else {
        fieldMsg(input.id, r.error || "Папка не выбрана — впишите путь вручную.", false);
      }
      return;
    }

    const exp = e.target.closest("[data-export-id]");
    if (exp) {
      const r = await api(`/api/exports/${exp.dataset.exportId}/${exp.dataset.exportAction}`);
      if (!r.ok) toast(r.error || "Не получилось", { error: true });
      return;
    }

    const pa = e.target.closest("[data-path-action]");
    if (pa) {
      const r = await api("/api/open-path", { path: pa.dataset.path, action: pa.dataset.pathAction });
      if (!r.ok) toast(r.error || "Файл не найден", { error: true });
      return;
    }

    const openD = e.target.closest("[data-open-dialog]");
    if (openD) { document.getElementById(openD.dataset.openDialog)?.showModal(); return; }
    const closeD = e.target.closest("[data-close-dialog]");
    if (closeD) { closeD.closest("dialog")?.close(); return; }

    const tab = e.target.closest("[data-tab]");
    if (tab) {
      $$("[data-tab]").forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        const panel = document.getElementById(t.dataset.tab);
        if (panel) panel.hidden = !on;
      });
      document.dispatchEvent(new CustomEvent("ytk:tab", { detail: tab.dataset.tab }));
      return;
    }

    const reset = e.target.closest("[data-reset-prompt]");
    if (reset) {
      const name = reset.dataset.resetPrompt;
      const r = await api(`/settings/prompts/${name}/reset`);
      if (r.ok) { document.getElementById("prompt_" + name).value = r.text; toast("Промпт сброшен к умолчанию"); }
      return;
    }

    const pull = e.target.closest("[data-pull-model]");
    if (pull) { startPull(pull.dataset.pullModel, pull.closest(".pull")); }
  });

  document.addEventListener("change", async (e) => {
    const input = e.target.closest("[data-check-folder]");
    if (!input) return;
    const r = await api("/api/check-folder", { path: input.value });
    fieldMsg(input.id, r.ok ? "папка доступна для записи" : r.error, r.ok);
  });

  // диалог «Переделать» закрываем после отправки
  document.body.addEventListener("htmx:afterRequest", (e) => {
    const form = e.target.closest && e.target.closest("[data-close-dialog-on-submit]");
    if (form) form.closest("dialog")?.close();
  });
  document.body.addEventListener("htmx:responseError", (e) => {
    let msg = "Ошибка запроса";
    try { msg = JSON.parse(e.detail.xhr.responseText).error || msg; } catch (err) { /* не JSON */ }
    toast(msg, { error: true });
  });

  // ───────── версии, редактор, экспорт ─────────
  $("#version-select")?.addEventListener("change", (e) => {
    const u = new URL(location.href);
    u.searchParams.set("v", e.target.value);
    location.href = u.toString();
  });

  const editToggle = $("#edit-toggle");
  if (editToggle) {
    const view = $("#summary-view"), editor = $("#editor");
    const show = (on) => { editor.hidden = !on; view.hidden = on; editToggle.disabled = on; };
    editToggle.addEventListener("click", () => {
      $('[data-tab="tab-summary"]')?.click();
      show(true);
      $("#md-editor").focus();
    });
    $("#cancel-edit").addEventListener("click", () => show(false));
    $("#save-edit").addEventListener("click", async (e) => {
      const r = await api(`/api/summaries/${e.target.dataset.summaryId}`, { markdown: $("#md-editor").value }, "PUT");
      if (r.ok) location.href = r.url; else toast(r.error || "Не сохранилось", { error: true });
    });
  }

  $("#export-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = e.target;
    const btn = $('button[type="submit"]', f);
    btn.disabled = true;
    const r = await api(`/api/summaries/${f.dataset.summaryId}/export`,
      { format: f.format.value, out_dir: f.out_dir.value });
    btn.disabled = false;
    if (r.ok) {
      f.closest("dialog").close();
      toast("Сохранено: " + r.name, {
        timeout: 12000,
        actions: [
          ["Открыть", () => api(`/api/exports/${r.export_id}/open`)],
          ["Показать в папке", () => api(`/api/exports/${r.export_id}/reveal`)],
        ],
      });
    } else {
      toast((r.error || "Не удалось сохранить") + (r.hint ? " " + r.hint : ""), { error: true, timeout: 12000 });
    }
  });

  // открывать обычные ссылки конспекта в новой вкладке (таймкоды перехватывает player.js)
  $$(".md a[href^='http']").forEach((a) => {
    if (!/youtu\.be\//.test(a.href)) { a.target = "_blank"; a.rel = "noopener"; }
  });

  // ───────── скачивание модели (экран «Состояние») ─────────
  const pullBoxes = {};
  async function startPull(name, box) {
    if (!name) return;
    pullBoxes[name] = box;
    const bar = $(".progress", box);
    if (bar) bar.hidden = false;
    $(".pull__msg", box).textContent = "начинаю…";
    const r = await api("/api/models/pull", { name });
    if (!r.ok) $(".pull__msg", box).textContent = r.error || "ошибка";
  }
  function onPull(ev) {
    const box = pullBoxes[ev.model] || $(`.pull[data-model="${CSS.escape(ev.model)}"]`);
    if (!box) return;
    const bar = $(".progress__bar", box);
    if (bar) { bar.parentElement.hidden = false; bar.style.width = Math.round((ev.progress || 0) * 100) + "%"; }
    $(".pull__msg", box).textContent = ev.message || "";
    if (ev.status === "done") { toast("Модель " + ev.model + " скачана"); setTimeout(() => location.reload(), 1200); }
    if (ev.status === "failed") toast(ev.message || "Не удалось скачать модель", { error: true });
  }
  $("#pull-other")?.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = e.target.name.value.trim();
    startPull(name, e.target);
  });
})();
