/* Плеер YouTube (IFrame API, youtube-nocookie.com): клик по таймкоду → seekTo, подсветка блока расшифровки. */
(function () {
  "use strict";
  const holder = document.getElementById("player");
  if (!holder) return;
  const videoId = holder.dataset.videoId;
  let player = null;
  let ready = false;

  window.onYouTubeIframeAPIReady = function () {
    player = new YT.Player("player", {
      host: "https://www.youtube-nocookie.com",
      videoId: videoId,
      playerVars: { rel: 0, modestbranding: 1, playsinline: 1 },
      events: { onReady: () => { ready = true; } },
    });
  };
  const tag = document.createElement("script");
  tag.src = "https://www.youtube.com/iframe_api";
  tag.onerror = () => {
    holder.innerHTML = '<p style="color:#fff;padding:1rem">Плеер не загрузился (нет интернета?). Таймкоды откроются на YouTube.</p>';
  };
  document.head.appendChild(tag);

  // ссылки вида https://youtu.be/ID?t=754 (и watch?v=ID&t=754) остаются настоящими — для экспорта
  function parseLink(href) {
    try {
      const u = new URL(href);
      let id = null;
      if (u.hostname === "youtu.be") id = u.pathname.slice(1);
      else if (/(^|\.)youtube(-nocookie)?\.com$/.test(u.hostname)) id = u.searchParams.get("v");
      const t = u.searchParams.get("t");
      if (id !== videoId || t === null) return null;
      return parseInt(t, 10) || 0;
    } catch (e) { return null; }
  }

  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    const t = parseLink(a.href);
    if (t === null) return;
    if (!ready || !player || !player.seekTo) { a.target = "_blank"; return; }
    e.preventDefault();
    player.seekTo(t, true);
    player.playVideo();
    if (window.matchMedia("(max-width: 960px)").matches) {
      holder.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });

  // подсветка текущего блока расшифровки: опрос getCurrentTime() раз в секунду
  const blocks = Array.from(document.querySelectorAll(".tblock[data-t]"));
  const times = blocks.map((b) => parseInt(b.dataset.t, 10));
  const follow = document.getElementById("follow-player");
  const panel = document.getElementById("tab-transcript");
  let current = -1;
  setInterval(() => {
    if (!ready || !blocks.length || !player.getCurrentTime) return;
    const now = player.getCurrentTime();
    let idx = -1;
    for (let i = 0; i < times.length; i++) { if (times[i] <= now) idx = i; else break; }
    if (idx === current) return;
    if (current >= 0) blocks[current].classList.remove("is-current");
    current = idx;
    if (idx < 0) return;
    blocks[idx].classList.add("is-current");
    const playing = player.getPlayerState && player.getPlayerState() === 1;
    if (playing && follow && follow.checked && panel && !panel.hidden) {
      blocks[idx].scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, 1000);
})();
