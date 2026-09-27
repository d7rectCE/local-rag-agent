// Applies the saved theme and accent before the first paint (the page's CSP forbids inline scripts).
(function () {
  var ACCENTS = {
    dark: ['#8b97ff', '#4fd1c5', '#f2a66b', '#b69cff'],
    light: ['#4453c6', '#0f766e', '#b4532a', '#7c3aed'],
  };
  function read(key, fallback) {
    try { return localStorage.getItem(key) || fallback; } catch (e) { return fallback; }
  }
  function apply() {
    var pref = read('rag.theme', 'system');
    var dark = pref === 'dark' || (pref === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    var theme = dark ? 'dark' : 'light';
    var i = parseInt(read('rag.accent', '0'), 10) || 0;
    var accent = ACCENTS[theme][Math.max(0, Math.min(3, i))];
    var n = parseInt(accent.slice(1), 16);
    var rgb = ((n >> 16) & 255) + ',' + ((n >> 8) & 255) + ',' + (n & 255);
    var root = document.documentElement;
    root.dataset.theme = theme;
    root.style.setProperty('--accent', accent);
    root.style.setProperty('--accent-rgb', rgb);
    root.style.setProperty('--soft', 'rgba(' + rgb + ',' + (dark ? '0.16' : '0.11') + ')');
  }
  window.ragTheme = { apply: apply, ACCENTS: ACCENTS };
  apply();
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply);
})();
