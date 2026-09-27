// Local RAG web UI: a thin client of the local API (same origin, no build step).
// Everything that comes from the model, the files or the web is untrusted: it is inserted as text,
// or as Markdown rendered by marked and cleaned by DOMPurify (no images, no styles, no scripts).

const ICONS = {
  layers: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 5v6h-6"/>',
  sliders: '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
  chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
  bulb: '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z"/>',
  chevron: '<path d="M6 9l6 6 6-6"/>',
  check: '<path d="M5 12l5 5L20 7"/>',
  terminal: '<path d="M4 17l6-5-6-5M12 19h8"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  paperclip: '<path d="M21 11.5l-8.6 8.6a5 5 0 0 1-7.1-7.1l8.6-8.6a3.3 3.3 0 0 1 4.7 4.7l-8.6 8.6a1.7 1.7 0 0 1-2.4-2.4l7.9-7.9"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  code: '<path d="M8 8l-4 4 4 4M16 8l4 4-4 4M14 5l-4 14"/>',
  'arrow-up': '<path d="M12 19V5M5 12l7-7 7 7"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
  pencil: '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 16v-5M12 8h.01"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  upload: '<path d="M12 15V3M7 8l5-5 5 5"/><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>',
};

function icon(name, size = 16, width = 1.8) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ''}</svg>`;
}

const ICON_SIZES = { layers: [18, 2], plus: [16, 2], x: [14, 2.2], 'arrow-up': [18, 2.2], menu: [18, 2], activity: [18, 2] };

function initIcons(root = document) {
  for (const el of root.querySelectorAll('[data-icon]')) {
    const [size, width] = ICON_SIZES[el.dataset.icon] || [17, 1.8];
    el.insertAdjacentHTML('afterbegin', icon(el.dataset.icon, size, width));
    el.removeAttribute('data-icon');
  }
}

// tiny DOM builder: text is always set with textContent
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'icon') el.insertAdjacentHTML('afterbegin', icon(...(Array.isArray(v) ? v : [v])));
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, '');
    else el.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const $ = (sel, root = document) => root.querySelector(sel);

// --------------------------------------------------------------------------- formatting

const nf = new Intl.NumberFormat('ru-RU');
const fmtNum = (n) => nf.format(n || 0);
const fmtSec = (s) => `${(s || 0).toFixed(1).replace('.', ',')} с`;
function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
function fmtSize(bytes) {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1).replace('.', ',')} МБ`;
}
const basename = (p) => (p || '').split(/[\\/]/).filter(Boolean).pop() || p;
function extOf(name) {
  const m = /\.([a-z0-9]+)$/i.exec(name || '');
  return m ? m[1].toLowerCase() : '';
}
function extBadge(name) {
  const e = extOf(name);
  return e === 'py' ? '.py' : (e || 'файл').slice(0, 4);
}

const TYPE_LABELS = {
  '.py': 'Python', '.ipynb': 'Ноутбуки', '.pdf': 'PDF', '.docx': 'Word', '.doc': 'Word 97–2003',
  '.txt': 'Текст', '.md': 'Markdown', '.log': 'Логи',
};
const ROUTES = { auto: 'Авто', corpus: 'Мои файлы', general: 'Общие знания' };
const REASONING = { off: 'Выкл', auto: 'Авто', on: 'Вкл' };
const LEVEL = { light: 'лёгкий', deep: 'глубокий', none: '' };

// --------------------------------------------------------------------------- state

const PREFS_KEY = 'rag.prefs';
function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}'); } catch { return {}; }
}
function savePrefs() {
  const p = { route: S.route, reasoning: S.reasoning, web: S.web, sandboxNet: S.sandboxNet, model: S.model, settings: S.settings };
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* private mode */ }
}

const prefs = loadPrefs();
const S = {
  status: null,
  policy: null,
  corpora: [],
  dialogs: [],
  dialog: null,       // {id, title, turns: [...]}
  uploads: [],        // uploads of the current dialog (the dialog is the upload session)
  attachments: [],    // upload ids attached to the next question
  uploading: [],      // {key, name, error}
  route: prefs.route || 'auto',
  reasoning: prefs.reasoning || null,
  web: !!prefs.web,
  sandboxNet: !!prefs.sandboxNet,  // code may run with network; every such task is confirmed
  model: prefs.model || null,      // null: the configured model
  models: null,                    // {default, models: [...]} from /models
  settings: { agent: 'auto', webMode: 'auto', code: 'auto', top_k: null, mode: null, rerank: null, symbols: null, ...(prefs.settings || {}) },
  busy: null,         // {kind, text, uploads, started}
  error: null,        // {kind, text, message}
  selected: null,     // id of the answer turn shown in the right panel
  wasIndexing: false,
};

// --------------------------------------------------------------------------- API

async function api(method, path, body, { form } = {}) {
  const opts = { method, headers: {} };
  if (form) opts.body = form;
  else if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  let r;
  try {
    r = await fetch(path, opts);
  } catch (e) {
    throw new Error('Сервер не отвечает. Запустите: rag serve');
  }
  let data = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    const d = data && data.detail;
    const msg = typeof d === 'string' ? d : (d ? JSON.stringify(d) : `ошибка ${r.status}`);
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return data;
}

function toast(msg, kind = '') {
  const el = h('div', { class: `toast ${kind}`, text: msg });
  $('#toasts').append(el);
  setTimeout(() => el.remove(), kind === 'error' ? 7000 : 3500);
}

async function guarded(fn, okMsg) {
  try {
    const res = await fn();
    if (okMsg) toast(okMsg);
    return res;
  } catch (e) {
    toast(e.message, 'error');
    return undefined;
  }
}

// --------------------------------------------------------------------------- loading

async function loadStatus() {
  const st = await api('GET', '/status');
  S.status = st;
  if (!S.reasoning) S.reasoning = st.reasoning?.mode || 'auto';
  return st;
}
async function loadPolicy() { S.policy = await api('GET', '/policy'); }
async function loadModels() {
  S.models = await api('GET', '/models');
  if (S.model && !S.models.models.some((m) => m.name === S.model)) { S.model = null; savePrefs(); } // removed from Ollama
}
async function loadDialogs() { S.dialogs = await api('GET', '/dialogs'); renderDialogs(); }
async function loadUploads() {
  S.uploads = S.dialog ? await api('GET', `/uploads?session=${encodeURIComponent(S.dialog.id)}`).catch(() => []) : [];
}

async function openDialog(id) {
  if (S.busy) return;
  const d = await guarded(() => api('GET', `/dialogs/${id}`));
  if (!d) return;
  S.dialog = d;
  S.error = null;
  S.selected = null;
  await loadUploads();
  // the files of the last question with attachments are still attached (they stay in the dialog's context)
  const last = [...(d.turns || [])].reverse().find((t) => t.kind === 'question' && (t.payload?.uploads || []).length);
  S.attachments = (last?.payload?.uploads || []).filter((id) => S.uploads.some((u) => u.id === id));
  closeSidebar();
  renderAll();
  scrollToBottom();
}

async function reloadDialog() {
  if (!S.dialog) return;
  S.dialog = await api('GET', `/dialogs/${S.dialog.id}`);
  await loadUploads();
}

let dialogPromise = null;
async function ensureDialog() {
  if (S.dialog) return S.dialog;
  if (!dialogPromise) {  // one dialog even if a file and a question arrive at the same time
    dialogPromise = api('POST', '/dialogs', { title: '' })
      .then((d) => { S.dialog = { ...d, turns: [] }; S.uploads = []; return S.dialog; })
      .finally(() => { dialogPromise = null; });
  }
  return dialogPromise;
}

function newDialog() {
  if (S.busy) return;
  S.dialog = null;
  S.uploads = [];
  S.attachments = [];
  S.error = null;
  S.selected = null;
  closeSidebar();
  renderAll();
  $('#msg').focus();
}

// --------------------------------------------------------------------------- sidebar

function corpusInfo() {
  return S.status?.corpus || null;
}
function indexing() {
  const p = S.status?.progress;
  return p && ['scanning', 'indexing', 'extracting'].includes(p.state) ? p : null;
}

// cards polled every few seconds are rebuilt only when their data changed: a rebuild under the
// pointer would swallow a click
const lastKeys = {};
function changed(name, data) {
  const key = JSON.stringify(data);
  if (lastKeys[name] === key) return false;
  lastKeys[name] = key;
  return true;
}

function renderFolder(force = false) {
  const el = $('#folder');
  const c = corpusInfo();
  const p = S.status?.progress || {};
  if (!changed('folder', [c && [c.root, c.stats?.n_files, c.vectors, c.catalog?.experiments], p.state, p.files_done, p.files_total, p.current]) && !force) return;
  el.replaceChildren();
  const prog = indexing();
  let chip;
  if (prog) {
    const pct = prog.files_total ? Math.round((100 * prog.files_done) / prog.files_total) : 0;
    chip = h('span', { class: 'chip-status warn busy', text: prog.state === 'extracting' ? 'каталог' : `индексация ${pct}%` });
  } else if (S.status?.progress?.state === 'error') chip = h('span', { class: 'chip-status danger', text: 'ошибка' });
  else if (c) chip = h('span', { class: 'chip-status ok', text: 'актуален' });
  else chip = h('span', { class: 'chip-status idle', text: 'нет папки' });
  el.append(h('div', { class: 'card-head' }, h('h2', { text: 'Рабочая папка' }), chip));

  if (!c) {
    el.append(
      h('p', { class: 'muted small', style: 'margin:0', text: 'Выберите папку с кодом, ноутбуками и документами — агент проиндексирует её и будет отвечать со ссылками.' }),
      h('button', { type: 'button', class: 'btn accent', onclick: openFolderModal, icon: ['folder', 14, 2] }, 'Выбрать папку'),
    );
    return;
  }
  el.append(
    h('button', { type: 'button', class: 'folder-item', title: c.root, onclick: openFolderModal },
      h('span', { class: 'folder-icon', icon: 'folder' }),
      h('span', { class: 'folder-name' }, h('b', { text: basename(c.root) }), h('span', { text: c.root }))),
  );
  if (prog) {
    const pct = prog.files_total ? (100 * prog.files_done) / prog.files_total : 5;
    el.append(
      h('div', { class: 'progress' }, h('span', { style: `width:${Math.max(3, pct).toFixed(0)}%` })),
      h('div', { class: 'muted small', text: prog.current ? `${prog.files_done} из ${prog.files_total} · ${basename(prog.current)}` : `${prog.files_done} из ${prog.files_total}` }),
    );
  }
  const nFiles = c.stats?.n_files || 0;
  const nNodes = c.vectors || c.stats?.n_nodes || 0;
  el.append(h('div', { class: 'stats' },
    h('div', {}, h('b', { text: fmtNum(nFiles) }), h('span', { text: plural(nFiles, 'файл', 'файла', 'файлов') })),
    h('div', {}, h('b', { text: fmtNum(nNodes) }), h('span', { text: plural(nNodes, 'фрагмент', 'фрагмента', 'фрагментов') }))));
  const cat = c.catalog || {};
  if (cat.experiments) {
    el.append(h('button', { type: 'button', class: 'text-link', onclick: openCatalogModal },
      `Каталог: ${cat.experiments} ${plural(cat.experiments, 'эксперимент', 'эксперимента', 'экспериментов')} →`));
  }
  el.append(h('div', { class: 'row' },
    prog
      ? h('button', { type: 'button', class: 'btn', onclick: () => guarded(() => api('POST', '/index/cancel')), icon: ['stop', 14, 2] }, 'Остановить')
      : h('button', { type: 'button', class: 'btn', onclick: () => startIndex(c.root, c.prefs), icon: ['refresh', 14, 2] }, 'Индексировать'),
    h('button', { type: 'button', class: 'icon-circle', 'aria-label': 'Исключения и настройки папки', title: 'Исключения и настройки папки', onclick: openFolderModal, icon: 'sliders' })));
}

async function startIndex(root, prefsForRoot, force = false) {
  const p = prefsForRoot || corpusInfo()?.prefs || S.status?.defaults || {};
  const res = await guarded(() => api('POST', '/index', { root, include_ext: p.include_ext, exclude: p.exclude, force }));
  if (res) {
    S.wasIndexing = true;
    await refreshStatus();
  }
}

function dayKey(iso) {
  const d = new Date(iso);
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
}
function groupLabel(iso) {
  const today = dayKey(new Date().toISOString());
  const k = dayKey(iso);
  if (k === today) return 'Сегодня';
  if (k === today - 86400000) return 'Вчера';
  if (today - k < 7 * 86400000) return 'На этой неделе';
  return 'Раньше';
}

function renderDialogs() {
  const nav = $('#dialogs');
  nav.replaceChildren();
  if (!S.dialogs.length) {
    nav.append(h('div', { class: 'empty-note', text: 'Диалогов пока нет' }));
    return;
  }
  let group = null;
  for (const d of S.dialogs) {
    const g = groupLabel(d.updated_at);
    if (g !== group) {
      nav.append(h('h2', { text: g }));
      group = g;
    }
    const active = S.dialog && S.dialog.id === d.id;
    nav.append(h('div', { class: `dialog-item${active ? ' active' : ''}` },
      h('button', { type: 'button', class: 'open', 'aria-current': active ? 'true' : null, title: d.title || 'Без названия', onclick: () => openDialog(d.id) }, d.title || 'Без названия'),
      h('span', { class: 'item-actions' },
        h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Переименовать', title: 'Переименовать', onclick: () => renameDialog(d), icon: ['pencil', 14] }),
        h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Удалить', title: 'Удалить', onclick: () => deleteDialog(d), icon: ['trash', 14] }))));
  }
}

const shortModel = (name) => (name || '').replace(/^ollama:/, '');

function renderModel() {
  const el = $('#model');
  const llm = S.status?.llm || {};
  const installed = S.models?.models?.map((m) => m.name);
  if (!changed('model', [llm.name, llm.available, S.model, installed, S.status?.gpu, S.status?.embedder?.device, document.documentElement.dataset.theme])) return;
  el.replaceChildren();
  const dev = S.status?.gpu || (S.status?.embedder?.device === 'cpu' ? 'CPU' : S.status?.embedder?.device || '');
  const dark = document.documentElement.dataset.theme === 'dark';
  const name = S.model || shortModel(llm.name) || '—';
  const available = S.model ? (installed ? installed.includes(S.model) : true) : llm.available !== false;
  el.append(
    h('div', { class: 'model-icon', icon: 'chip' }),
    h('button', { type: 'button', class: 'model-name', title: 'Выбрать модель', 'aria-label': `Модель ${name}. Выбрать другую`, onclick: openModelModal },
      h('b', { text: name }), h('span', { text: ['Ollama', dev].filter(Boolean).join(' · ') + (S.model ? ' · выбрана' : '') })),
    h('span', { class: `dot${available ? '' : ' off'}`, role: 'img', 'aria-label': available ? 'Модель запущена' : 'Модель недоступна', title: available ? 'Модель доступна' : 'Модель недоступна: запустите Ollama' }),
    h('button', { type: 'button', class: 'mini-btn', 'aria-label': dark ? 'Светлая тема' : 'Тёмная тема', title: dark ? 'Светлая тема' : 'Тёмная тема', onclick: toggleTheme, icon: [dark ? 'sun' : 'moon', 15] }),
    h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Параметры ответа', title: 'Параметры ответа', onclick: openSettingsModal, icon: ['gear', 15] }),
  );
}

function setTheme(pref) {
  try { localStorage.setItem('rag.theme', pref); } catch { /* ignore */ }
  window.ragTheme.apply();
  renderModel();
}
function toggleTheme() {
  setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
}

// --------------------------------------------------------------------------- header, composer controls

function segmented(container, options, value, onPick, label) {
  container.replaceChildren();
  if (label) container.append(label);
  for (const [k, text] of Object.entries(options)) {
    container.append(h('button', { type: 'button', 'aria-pressed': String(k === value), onclick: () => onPick(k) }, text));
  }
}

function renderHeader() {
  const d = S.dialog;
  $('#dialog-title').textContent = d?.title || 'Новый диалог';
  const parts = [];
  const n = d?.turns?.length || 0;
  if (n) parts.push(`${n} ${plural(n, 'сообщение', 'сообщения', 'сообщений')}`);
  const c = corpusInfo();
  const usedWeb = (d?.turns || []).some((t) => t.kind === 'answer' && (t.payload?.sources || []).some((s) => s.file_type === 'web'));
  if (c) parts.push(`папка ${basename(c.root)}${usedWeb ? ' и веб' : ''}`);
  else parts.push('папка не выбрана');
  $('#dialog-meta').textContent = parts.join(' · ');
  segmented($('#route'), ROUTES, S.route, (k) => { S.route = k; savePrefs(); renderHeader(); });
}

function renderControls() {
  const label = h('span', { class: 'seg-label', icon: ['bulb', 14] }, 'Думать');
  segmented($('#reasoning'), REASONING, S.reasoning || 'auto', (k) => { S.reasoning = k; savePrefs(); renderControls(); }, label);
  $('#web-toggle').setAttribute('aria-pressed', String(S.web));
  $('#web-toggle').title = S.web ? (S.settings.webMode === 'always' ? 'Интернет: искать всегда' : 'Интернет: когда нужен') : 'Интернет выключен';
  const net = $('#net-toggle');
  const netAllowed = S.policy?.sandbox_network !== 'never' && S.settings.code !== 'off';
  net.hidden = !netAllowed;
  net.setAttribute('aria-pressed', String(S.sandboxNet && netAllowed));
  net.title = S.sandboxNet
    ? 'Код может выходить в интернет (скачать данные, pip install); каждый такой запуск нужно разрешить'
    : 'Код в песочнице работает без интернета';
  renderAttachments();
  renderSend();
}

function renderSend() {
  const btn = $('#send');
  const empty = !$('#msg').value.trim();
  btn.disabled = !!S.busy || empty || S.uploading.some((u) => !u.error);
}

function renderAttachments() {
  const box = $('#attachments');
  box.replaceChildren();
  for (const id of S.attachments) {
    const u = S.uploads.find((x) => x.id === id);
    if (!u) continue;
    const meta = u.fits_context ? fmtSize(u.size) : `${fmtSize(u.size)} · ${u.n_fragments} фрагм.`;
    box.append(h('span', { class: 'att', title: u.warnings?.length ? u.warnings.join('\n') : u.name },
      h('span', { class: 'ext', text: extBadge(u.name) }), h('b', { text: u.name }), h('span', { class: 'meta', text: meta }),
      h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Убрать файл из контекста', title: 'Файл в контексте диалога: агент учитывает его в следующих вопросах. Убрать (файл останется в сессии)', onclick: () => { S.attachments = S.attachments.filter((x) => x !== id); renderControls(); }, icon: ['x', 12, 2.2] })));
  }
  for (const u of S.uploading) {
    box.append(h('span', { class: `att ${u.error ? 'error' : 'loading'}`, title: u.error || 'Разбираю файл…' },
      h('span', { class: 'ext', text: extBadge(u.name) }), h('b', { text: u.name }),
      h('span', { class: 'meta', text: u.error ? 'не принят' : 'разбор…' }),
      u.error ? h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Убрать', onclick: () => { S.uploading = S.uploading.filter((x) => x !== u); renderControls(); }, icon: ['x', 12, 2.2] }) : null));
  }
}

// --------------------------------------------------------------------------- markdown with citations

const CITE_RE = /\[(\d{1,3})\]/g;

function renderMarkdown(text, sources) {
  const byN = new Map((sources || []).map((s) => [s.n, s]));
  // citations become placeholders outside code, then chips after sanitizing
  const parts = String(text || '').split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  const prepared = parts.map((p, i) => (i % 2 ? p : p.replace(CITE_RE, (m, n) => (byN.has(+n) ? `\u2063${n}\u2063` : m)))).join('');
  let html;
  if (window.marked && window.DOMPurify) {
    html = window.DOMPurify.sanitize(window.marked.parse(prepared, { gfm: true, breaks: false }), {
      FORBID_TAGS: ['img', 'style', 'iframe', 'form', 'input', 'video', 'audio', 'svg', 'math', 'object', 'embed', 'button'],
      FORBID_ATTR: ['style', 'class', 'id', 'srcset'],
    });
  }
  const div = h('div', { class: 'md' });
  if (html === undefined) div.textContent = prepared.replace(/\u2063/g, '');
  else div.innerHTML = html;
  for (const a of div.querySelectorAll('a[href]')) {
    a.target = '_blank';
    a.rel = 'noopener noreferrer nofollow';
  }
  // placeholders -> chips (never inside code)
  const walker = document.createTreeWalker(div, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) {
    const n = walker.currentNode;
    if (n.nodeValue.includes('\u2063') && !n.parentElement.closest('code, pre')) nodes.push(n);
  }
  for (const n of nodes) {
    const frag = document.createDocumentFragment();
    n.nodeValue.split(/\u2063(\d{1,3})\u2063/g).forEach((piece, i) => {
      if (i % 2 === 0) {
        if (piece) frag.append(piece);
      } else {
        const s = byN.get(+piece);
        frag.append(h('button', { type: 'button', class: `cite${s?.file_type === 'web' ? ' web' : ''}`, title: s ? `${s.file_path}${s.location ? ' · ' + s.location : ''}` : '', onclick: () => s && openSourceModal(s) }, piece));
      }
    });
    n.replaceWith(frag);
  }
  return div;
}

// --------------------------------------------------------------------------- thread

function sourceKind(s) {
  if (s.file_type === 'web') return 'веб';
  if (s.file_type === 'catalog') return 'каталог';
  if (s.file_type === 'sandbox') return 'вычисление';
  if (s.node_type === 'log_chunk' || /\.log$/i.test(s.file_path)) return 'лог';
  return { py: 'код', ipynb: 'ноутбук', pdf: 'pdf', docx: 'docx', txt: 'текст', png: 'изображение' }[s.file_type] || s.file_type;
}
function sourceName(s) {
  if (s.file_type === 'web') {
    try { return new URL(s.file_path).hostname.replace(/^www\./, ''); } catch { return s.file_path; }
  }
  if (s.file_type === 'catalog') return 'SQL-запрос';
  if (s.file_type === 'sandbox') return 'код в песочнице';
  return basename(s.file_path.replace(/^upload:/, ''));
}
function sourceLoc(s) {
  if (s.file_type === 'web') return (s.title || '').replace(/^web · /, '') || s.location;
  return s.location || s.title || '';
}

function routeLabel(a) {
  const web = (a.sources || []).some((s) => s.file_type === 'web');
  if (a.route === 'code') return 'код-агент';
  if (a.route === 'general') return 'общие знания';
  if (a.route === 'web') return (a.sources || []).some((s) => s.file_type !== 'web') ? 'веб + мои файлы' : 'веб';
  if (a.route === 'upload') return 'загруженный файл';
  return web ? 'мои файлы + веб' : 'мои файлы';
}

function sourceCard(s) {
  return h('button', { type: 'button', class: `source${s.file_type === 'web' ? ' web' : ''}`, id: `src-${s.n}`, title: s.file_path, onclick: () => openSourceModal(s) },
    h('span', { class: 'lbl', text: `${s.n} · ${sourceKind(s)}` }),
    h('span', { class: 'fn', text: sourceName(s) }),
    h('span', { class: 'loc', text: sourceLoc(s) || '—' }));
}

function questionOf(turns, idx) {
  for (let i = idx - 1; i >= 0; i--) if (turns[i].role === 'user') return turns[i];
  return null;
}

function renderAnswer(turn, turns, idx) {
  const a = turn.payload || {};
  const art = h('article', { class: `answer${S.selected === turn.id ? ' selected' : ''}`, dataset: { turn: turn.id } });
  art.addEventListener('click', (e) => {
    if (e.target.closest('button, a, summary')) return;
    S.selected = turn.id;
    renderPanel();
  });
  const agent = (a.trace || []).some((st) => st.name === 'agent' || st.name === 'code_agent');
  const def = S.models?.default || shortModel(S.status?.llm?.name);
  const model = shortModel(a.model);
  art.append(h('div', { class: 'answer-head' },
    h('span', { class: 'avatar', icon: [a.route === 'code' ? 'terminal' : 'layers', 13, 2.2] }),
    h('span', { class: 'who', text: agent ? 'Агент' : 'Ответ' }),
    h('span', { class: 'badge', text: routeLabel(a) }),
    model && def && model !== def ? h('span', { class: 'mono', title: 'Модель этого ответа', text: model }) : null,
    h('span', { text: fmtSec(a.latency_s) }),
    h('button', { type: 'button', class: 'trace-link', onclick: () => { S.selected = turn.id; renderPanel(); openTraceModal(a); } }, 'трасса')));

  if (a.reasoning) {
    const budget = S.status?.reasoning?.budget_tokens?.[a.reasoning_level];
    const meta = [LEVEL[a.reasoning_level], `${fmtNum(a.reasoning_tokens)}${budget ? ' / ' + fmtNum(budget) : ''} ток.`, a.reasoning_truncated ? 'обрезано' : ''].filter(Boolean).join(' · ');
    const text = h('pre', { class: 'reasoning-text', hidden: true, text: a.reasoning });
    const note = h('p', { class: 'reasoning-note', hidden: true, text: 'Черновик модели, а не объяснение ответа: подтверждают ответ только ссылки на источники.' });
    const chip = h('button', { type: 'button', class: 'reasoning-chip', 'aria-expanded': 'false', onclick: () => {
      const open = chip.getAttribute('aria-expanded') !== 'true';
      chip.setAttribute('aria-expanded', String(open));
      text.hidden = !open;
      note.hidden = !open;
    } },
    h('span', { class: 'ic', icon: ['bulb', 13, 2] }), h('b', { text: 'Черновик рассуждений' }), h('span', { text: meta }),
    h('span', { class: 'chev', icon: ['chevron', 14, 2] }));
    art.append(chip, text, note);
  }

  if (a.pending && a.pending.length) {
    const p = a.pending[0];
    const q = questionOf(turns, idx);
    const what = p.tool === 'web_search' ? `поиск в интернете: «${p.args?.query || ''}»`
      : p.tool === 'run_code' && p.args?.network ? `код с доступом в интернет: «${p.args.task || ''}»`
        : `${p.tool} ${JSON.stringify(p.args || {})}`;
    art.append(h('div', { class: 'notice warn' },
      h('span', { class: 'ic', icon: ['alert', 16, 2] }),
      h('div', { class: 'body' },
        h('b', { text: 'Нужно ваше подтверждение' }),
        h('span', {}, 'Агент хочет выполнить ', h('code', { text: what }), '.'),
        h('span', { class: 'small muted', text: p.reason || '' }),
        h('div', { class: 'actions' },
          h('button', { type: 'button', class: 'btn-pill accent', disabled: !!S.busy || !q, onclick: () => q && ask(q.content, { confirmed: a.pending.map((x) => x.key), replaceLast: true, from: q.payload }) }, 'Разрешить'),
          h('button', { type: 'button', class: 'btn-pill', onclick: (e) => { e.currentTarget.closest('.notice').remove(); } }, 'Не надо')))));
  }

  if (a.code) {  // the chat handed the task to the code agent: its result in words, then the card
    if (a.answer && a.answer !== a.code.summary) art.append(renderMarkdown(a.answer, []));
    art.append(codeCard(a.code));
  }
  else if (a.answer && !(a.pending || []).some((p) => p.tool === 'run_code')) art.append(renderMarkdown(a.answer, a.sources));
  if ((a.suggest || []).includes('web') && !(a.pending || []).length) {
    const q = questionOf(turns, idx);
    art.append(h('div', { class: 'notice' }, h('span', { class: 'ic', icon: ['globe', 16, 2] }),
      h('div', { class: 'body' },
        h('span', { text: 'Для ответа, похоже, нужны свежие данные из интернета, а он выключен.' }),
        h('div', { class: 'actions' },
          h('button', { type: 'button', class: 'btn-pill accent', disabled: !!S.busy || !q, onclick: () => {
            if (!q) return;
            S.web = true;
            savePrefs();
            renderControls();
            renderAccessCard();
            ask(q.content, { replaceLast: true, from: { ...(q.payload || {}), web: S.settings.webMode || 'auto' } });
          } }, 'Включить интернет и спросить снова')))));
  }
  if (a.general) {
    art.append(h('div', { class: 'general-block' },
      h('span', { class: 'general-label', icon: ['info', 13, 2] }, 'Из общих знаний модели, не из ваших файлов'),
      renderMarkdown(a.general, [])));
  }
  if (a.notice) art.append(h('div', { class: 'notice' }, h('span', { class: 'ic', icon: ['info', 16, 2] }), h('div', { class: 'body', text: a.notice })));
  for (const c of a.conflicts || []) {
    art.append(h('div', { class: 'notice warn' }, h('span', { class: 'ic', icon: ['alert', 16, 2] }),
      h('div', { class: 'body' }, h('b', { text: `Расхождение: ${c.topic || 'архив и интернет'}` }),
        h('dl', { class: 'conflict' }, h('dt', { text: 'в файлах' }), h('dd', { text: c.archive || '' }), h('dt', { text: 'в интернете' }), h('dd', { text: c.web || '' })))));
  }
  const sources = a.sources || [];
  const cited = sources.filter((s) => s.cited).sort((x, y) => x.n - y.n);
  if (cited.length) art.append(h('div', { class: 'sources' }, cited.map(sourceCard)));
  if (sources.length > cited.length) {
    art.append(h('button', { type: 'button', class: 'text-link more-sources', onclick: () => openSourcesModal(a) },
      cited.length ? `Все найденные фрагменты (${sources.length}) →` : `Найденные фрагменты (${sources.length}) →`));
  }
  return art;
}

function testsPassed(res) {
  let n = null;
  for (const st of res.steps || []) {
    const m = /(\d+) passed/.exec(st.observation || '');
    if (m) n = +m[1];
  }
  return n;
}

function diffView(diff) {
  const box = h('div', { class: 'diff', role: 'region', 'aria-label': 'Изменения' });
  const lines = (diff || '').split('\n');
  let shown = 0;
  for (const ln of lines) {
    if (shown > 600) { box.append(h('div', { class: 'hunk', text: `… ещё ${lines.length - shown} строк` })); break; }
    if (ln.startsWith('diff --git')) { box.append(h('div', { class: 'file', text: '▸ ' + ln.replace(/^diff --git a\/(\S+) b\/.*$/, '$1') })); shown++; continue; }
    if (/^(index |--- |\+\+\+ |new file mode|deleted file mode|similarity|rename )/.test(ln)) continue;
    const cls = ln.startsWith('@@') ? 'hunk' : ln.startsWith('+') ? 'add' : ln.startsWith('-') ? 'del' : '';
    box.append(h('div', { class: cls, text: ln || ' ' }));
    shown++;
  }
  return box;
}

function renderCode(turn) {  // a code task of the earlier explicit code mode
  return codeCard(turn.payload || {});
}

function codeCard(r) {
  const card = h('article', { class: 'code-card', 'aria-label': 'Код-агент' });
  const passed = testsPassed(r);
  let status;
  if (r.status === 'done') status = h('span', { class: 'tag ok push', icon: ['check', 13, 2.4] }, passed ? `${passed} ${plural(passed, 'тест прошёл', 'теста прошли', 'тестов прошли')}` : 'готово');
  else if (r.status === 'failed') status = h('span', { class: 'tag danger push', text: 'не удалось' });
  else if (r.status === 'limit') status = h('span', { class: 'tag warn push', text: 'лимит шагов' });
  else status = h('span', { class: 'tag danger push', text: 'ошибка' });
  card.append(h('div', { class: 'code-head' },
    h('span', { class: 'code-icon', icon: ['terminal', 15, 2] }), h('span', { class: 'title', text: 'Код-агент' }),
    r.network ? h('span', { class: 'tag warn', title: 'Сеть была разрешена для этой задачи', text: 'с сетью' }) : h('span', { class: 'tag', text: 'без сети' }),
    h('span', { class: 'tag', text: 'корпус: чтение' }),
    r.pipeline && r.pipeline !== 'free' ? h('span', { class: 'tag accent', text: `конвейер ${r.pipeline}` }) : null,
    status));
  if (r.diff) card.append(diffView(r.diff));
  const arts = (r.artifacts || []).filter((p) => !(r.changed || []).some(([, q]) => q === p) || /\.(png|svg|jpe?g|csv)$/i.test(p));
  if (arts.length) {
    const box = h('div', { class: 'artifacts' });
    for (const p of arts) {
      const url = `/code/${encodeURIComponent(r.task_id)}/file?path=${encodeURIComponent(p)}`;
      if (/\.(png|jpe?g)$/i.test(p)) box.append(h('a', { href: url, target: '_blank', rel: 'noopener', title: p }, h('img', { src: url, alt: p })));
      else box.append(h('a', { href: url, target: '_blank', rel: 'noopener', text: p }));
    }
    card.append(box);
  }
  const changed = r.changed || [];
  const applied = (r.applied || []).length;
  const pending = changed.length && !applied && !r.rejected;
  const summary = [
    r.summary,
    `${r.runs || 0} ${plural(r.runs || 0, 'запуск', 'запуска', 'запусков')}, неудачных ${r.failed_runs || 0} · ${fmtSec(r.latency_s)}`,
    pending ? 'Запись в папку — только после подтверждения.' : '',
  ].filter(Boolean).join('\n');
  const lines = summary.split('\n');
  const short = lines.length > 5 ? [...lines.slice(0, 4), '…'].join('\n') : summary;
  const sumEl = h('span', { class: 'summary', text: short });
  const foot = h('div', { class: 'code-foot' }, sumEl);
  if (short !== summary) {
    const more = h('button', { type: 'button', class: 'text-link', onclick: () => {
      const full = sumEl.textContent === short;
      sumEl.textContent = full ? summary : short;
      more.textContent = full ? 'свернуть' : 'подробнее';
    } }, 'подробнее');
    foot.append(more);
  }
  if (applied) foot.append(h('span', { class: 'tag ok', icon: ['check', 13, 2.4] }, `применено: ${applied} ${plural(applied, 'файл', 'файла', 'файлов')}`));
  else if (r.rejected) foot.append(h('span', { class: 'tag', text: 'отклонено' }));
  else if (changed.length) {
    foot.append(
      h('button', { type: 'button', class: 'btn-pill', onclick: () => codeAction(r.task_id, 'reject') }, 'Отклонить'),
      h('button', { type: 'button', class: 'btn-pill accent', onclick: () => codeAction(r.task_id, 'apply') }, 'Применить'));
  }
  card.append(foot);
  if ((r.steps || []).length) {
    const list = h('div', { class: 'steps-list', hidden: true });
    for (const st of r.steps) {
      const args = st.args ? Object.entries(st.args).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(' ') : '';
      list.append(h('div', { class: 'step' }, h('b', { text: `${st.step}. ${st.action}` }), args ? h('span', { class: 'muted mono small', text: ' ' + args.slice(0, 200) }) : null,
        st.thought ? h('div', { class: 'muted', text: st.thought }) : null, st.observation ? h('pre', { text: st.observation }) : null));
    }
    if ((r.checkpoints || []).length > 1 && !applied) {
      const sel = h('select', { class: 'input', 'aria-label': 'Контрольная точка' }, r.checkpoints.map((c) => h('option', { value: c.commit, text: `${c.message || c.commit} · ${c.commit.slice(0, 7)}` })));
      list.append(h('div', { class: 'row' }, sel, h('button', { type: 'button', class: 'btn inline', onclick: () => codeRollback(r.task_id, sel.value) }, 'Откатить копию')));
    }
    const toggle = h('button', { type: 'button', class: 'text-link', style: 'margin: 0 16px 12px', onclick: () => { list.hidden = !list.hidden; toggle.textContent = list.hidden ? `Шаги агента (${r.steps.length}) →` : 'Скрыть шаги'; } }, `Шаги агента (${r.steps.length}) →`);
    card.append(toggle, list);
  }
  return card;
}

async function codeAction(taskId, action) {
  const res = await guarded(() => api('POST', `/code/${encodeURIComponent(taskId)}/${action}`), action === 'apply' ? 'Изменения перенесены в папку' : 'Изменения отклонены');
  if (res) {
    await reloadDialog();
    renderThread();
    renderPanel();
  }
}
async function codeRollback(taskId, commit) {
  const res = await guarded(() => api('POST', `/code/${encodeURIComponent(taskId)}/rollback`, { commit }), 'Рабочая копия откачена');
  if (res) { await reloadDialog(); renderThread(); }
}

function renderQuestion(turn) {
  const ids = turn.payload?.uploads || [];
  const files = ids.map((id) => S.uploads.find((u) => u.id === id)).filter(Boolean);
  return h('div', { class: 'bubble-wrap' },
    files.map((u) => h('span', { class: 'file-chip' }, h('span', { class: 'ext', text: extBadge(u.name) }),
      h('span', { class: 'name' }, h('b', { text: u.name }), fmtSize(u.size)))),
    h('div', { class: 'bubble' }, turn.kind === 'code_task' ? h('span', { class: 'tag-in', text: 'код' }) : null, turn.content));
}

function renderWelcome() {
  const c = corpusInfo();
  const examples = [
    { k: 'Мои файлы', q: 'В каком эксперименте лучший ROC-AUC и чем он отличается от остальных?' },
    { k: 'Код', q: 'Где определена функция подсчёта метрик и где она вызывается?' },
    { k: 'Интернет', q: 'Какая сейчас последняя версия scikit-learn?', web: true },
    { k: 'Запуск кода', q: 'Собери метрики из всех ноутбуков в CSV и построй по ним график' },
  ];
  return h('div', { class: 'welcome' },
    h('div', { class: 'brand-logo', icon: ['layers', 24, 2] }),
    h('h2', { text: c ? 'С чего начнём?' : 'Выберите рабочую папку' }),
    h('p', { text: c ? `Спросите о файлах из «${basename(c.root)}» или о чём угодно ещё. Агент сам решит, искать ли в файлах, в интернете или запустить код в песочнице. Ответы по файлам — со ссылками на строку, ячейку или страницу, изменения в папке — только после вашего подтверждения.` : 'Агент проиндексирует код, ноутбуки и документы и будет отвечать по ним со ссылками. Общие вопросы работают и без папки.' }),
    c ? h('div', { class: 'examples' }, examples.map((e) => h('button', { type: 'button', class: 'example', onclick: () => {
      if (e.web) S.web = true;
      savePrefs();
      $('#msg').value = e.q;
      renderControls();
      autoGrow();
      $('#msg').focus();
    } }, h('span', { class: 'k', text: e.k }), e.q)))
      : h('button', { type: 'button', class: 'btn accent inline', onclick: openFolderModal, icon: ['folder', 14, 2] }, 'Выбрать папку'));
}

let tickTimer = null;
function renderThread() {
  const thread = $('#thread');
  thread.replaceChildren();
  const turns = S.dialog?.turns || [];
  if (!turns.length && !S.busy && !S.error) {
    thread.append(renderWelcome());
    return;
  }
  turns.forEach((t, i) => {
    if (t.kind === 'question' || t.kind === 'code_task') thread.append(renderQuestion(t));
    else if (t.kind === 'answer') thread.append(renderAnswer(t, turns, i));
    else if (t.kind === 'code') thread.append(renderCode(t));
  });
  if (S.busy) {
    if (!S.busy.replace) thread.append(renderQuestion({ kind: S.busy.kind === 'code' ? 'code_task' : 'question', content: S.busy.text, payload: { uploads: S.busy.uploads } }));
    const label = h('span', { class: 'busy-label' });
    const update = () => {
      const s = (Date.now() - S.busy.started) / 1000;
      label.textContent = `${S.busy.kind === 'code' ? 'Код-агент работает в песочнице' : s > 20 ? 'Работаю: поиск, рассуждение или запуск кода' : 'Ищу и думаю'} · ${Math.floor(s)} с`;
    };
    update();
    clearInterval(tickTimer);
    tickTimer = setInterval(() => (S.busy ? update() : clearInterval(tickTimer)), 1000);
    thread.append(h('article', { class: 'answer' },
      h('div', { class: 'answer-head' }, h('span', { class: 'avatar', icon: [S.busy.kind === 'code' ? 'terminal' : 'layers', 13, 2.2] }), h('span', { class: 'who', text: S.busy.kind === 'code' ? 'Код-агент' : 'Агент' })),
      h('div', { class: 'thinking' }, h('span', { class: 'dots' }, h('span'), h('span'), h('span')), label)));
  }
  if (S.error) {
    const e = S.error;
    thread.append(h('div', { class: 'notice danger' }, h('span', { class: 'ic', icon: ['alert', 16, 2] }),
      h('div', { class: 'body' }, h('b', { text: 'Не удалось получить ответ' }), h('span', { text: e.message }),
        h('div', { class: 'actions' }, h('button', { type: 'button', class: 'btn-pill', onclick: () => { S.error = null; send(e.text, e.kind); } }, 'Повторить')))));
  }
}

function scrollToBottom() {
  const m = $('#messages');
  requestAnimationFrame(() => { m.scrollTop = m.scrollHeight; });
}

// --------------------------------------------------------------------------- right panel

function traceGroups(a) {
  const groups = [];
  const add = (key, name, color, dur, value) => {
    let g = groups.find((x) => x.key === key);
    if (!g) { g = { key, name, color, dur: 0, value: '' }; groups.push(g); }
    g.dur += dur || 0;
    if (value !== undefined) g.value = value;
  };
  let agentSteps = 0;
  for (const st of a.trace || []) {
    const d = st.detail || {};
    switch (st.name) {
      case 'route': add('route', 'Маршрутизатор', 'c-router', st.duration_s, a.route === 'upload' ? 'загруженный файл' : a.route === 'code' ? 'запуск кода' : (d.route === 'general' ? 'общие знания' : 'мои файлы') + (d.web ? ' + веб' : '')); break;
      case 'route_code': add('route', 'Маршрутизатор', 'c-router', st.duration_s); break;
      case 'code_agent': add('code', 'Код-агент', 'c-agent', st.duration_s, `${d.runs || 0} ${plural(d.runs || 0, 'запуск', 'запуска', 'запусков')}${d.network ? ' · с сетью' : ''}`); break;
      case 'retrieve': add('search', 'Поиск', 'c-search', st.duration_s, `${(d.hits || []).length} фрагм.`); break;
      case 'sql': add('sql', 'SQL к каталогу', 'c-search', st.duration_s, d.error ? 'ошибка' : `${d.rows} ${plural(d.rows || 0, 'строка', 'строки', 'строк')}`); break;
      case 'crag': add('crag', 'Проверка релевантности', 'c-ok', st.duration_s, { correct: 'релевантно', ambiguous: 'частично', incorrect: 'не найдено' }[d.verdict] || d.verdict || ''); break;
      case 'agent': agentSteps += 1; add('agent', 'Агент', 'c-agent', st.duration_s, `${agentSteps} ${plural(agentSteps, 'шаг', 'шага', 'шагов')}`); break;
      case 'web_search': add('web', 'Веб-шлюз', 'c-web', st.duration_s, `${(d.results || []).length} результатов`); break;
      case 'web_read': {
        const pages = (d.pages || []).filter((p) => p.passages).length;
        add('web', 'Веб-шлюз', 'c-web', st.duration_s, `${pages} ${plural(pages, 'страница', 'страницы', 'страниц')}`);
        break;
      }
      case 'web_error': add('web', 'Веб-шлюз', 'c-web', st.duration_s, 'поиск недоступен'); break;
      case 'web_policy': case 'web_blocked': add('web', 'Веб-шлюз', 'c-web', st.duration_s, 'ждёт подтверждения'); break;
      case 'upload': add('upload', 'Загрузка', 'c-upload', st.duration_s, { whole: 'целиком', index: 'по индексу', parts: 'по частям', 'index+parts': 'индекс + части' }[d.route] || d.route || ''); break;
      case 'upload_parts': add('upload', 'Загрузка', 'c-upload', st.duration_s, 'по частям'); break;
      case 'generate': case 'generate_general': case 'escalate':
        add('gen', a.reasoning_tokens ? 'Рассуждение и ответ' : 'Ответ', 'c-gen', st.duration_s,
          a.reasoning_tokens ? `${fmtNum(a.reasoning_tokens)} ток.` : `${fmtNum(d.completion_tokens || 0)} ток.`);
        break;
      default: break;
    }
  }
  if (a.route !== 'general' && a.route !== 'code' && a.answerable) {
    const markers = new Set([...(a.answer || '').matchAll(CITE_RE)].map((m) => m[1])).size;
    add('cite', 'Проверка ссылок', 'c-ok', 0, markers ? `${(a.citations || []).length} из ${markers}` : 'без ссылок');
  }
  return groups;
}

function selectedAnswer() {
  const turns = S.dialog?.turns || [];
  const pick = S.selected ? turns.find((t) => t.id === S.selected && t.kind === 'answer') : null;
  return pick || [...turns].reverse().find((t) => t.kind === 'answer') || null;
}

function renderTraceCard() {
  const el = $('#trace-card');
  el.replaceChildren();
  const t = selectedAnswer();
  const a = t?.payload;
  el.append(h('div', { class: 'card-head' }, h('h2', { text: 'Ход ответа' }), a ? h('span', { class: 'mono', text: fmtSec(a.latency_s) }) : null));
  if (!a) {
    el.append(h('p', { class: 'panel-empty', style: 'margin:0', text: 'Здесь появятся шаги ответа: маршрут, поиск, веб, рассуждение и проверка ссылок.' }));
    return;
  }
  const groups = traceGroups(a);
  const total = groups.reduce((s, g) => s + g.dur, 0) || 1;
  const bar = h('div', { class: 'bar', 'aria-hidden': 'true' });
  for (const g of groups) {
    const w = g.key === 'cite' ? 4 : Math.max(2, (100 * g.dur) / total);
    if (g.dur > 0 || g.key === 'cite') bar.append(h('span', { class: g.color, style: `width:${w.toFixed(1)}%` }));
  }
  el.append(bar, h('ol', { class: 'steps' }, groups.map((g) => h('li', { title: g.dur ? fmtSec(g.dur) : '' },
    h('span', { class: `sw ${g.color}` }), h('span', { class: 'nm', text: g.name }), h('span', { class: 'vl', text: g.value })))));
}

function renderFilesCard() {
  const el = $('#files-card');
  el.replaceChildren();
  el.append(h('div', { class: 'card-head' }, h('h2', { text: 'Файлы сессии' }),
    h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Загрузить файл', title: 'Загрузить файл', onclick: () => $('#file-input').click(), icon: ['upload', 15] })));
  const rows = [];
  for (const t of S.dialog?.turns || []) {
    const r = t.kind === 'code' ? t.payload : t.kind === 'answer' ? t.payload?.code : null;
    if (r && (r.changed || []).length && !(r.applied || []).length && !r.rejected) {
      for (const [st, path] of r.changed) {
        rows.push(h('div', { class: 'file-row' }, h('span', { class: 'ext', text: extBadge(path) }),
          h('div', { class: 'txt' }, h('b', { text: basename(path), title: path }), h('span', { class: 'pending', text: `ждёт подтверждения${st === 'A' ? ' · новый' : st === 'D' ? ' · удалён' : ''}` }))));
      }
    }
  }
  for (const u of S.uploads) {
    const status = u.duplicate_of ? `уже в корпусе: ${basename(u.duplicate_of)}` : u.fits_context ? 'целиком в контексте' : `сессионный индекс · ${u.n_fragments} фрагм.`;
    rows.push(h('div', { class: 'file-row' }, h('span', { class: 'ext', text: extBadge(u.name) }),
      h('div', { class: 'txt' }, h('b', { text: u.name, title: u.name }), h('span', { text: status })),
      h('span', { class: 'row-actions' },
        S.attachments.includes(u.id) ? null : h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Прикрепить к вопросу', title: 'Прикрепить к следующему вопросу', onclick: () => { S.attachments.push(u.id); renderControls(); }, icon: ['paperclip', 14] }),
        corpusInfo() && !u.duplicate_of ? h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Добавить в корпус', title: 'Добавить в рабочую папку и проиндексировать', onclick: () => addToCorpus(u), icon: ['folder', 14] }) : null,
        h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Удалить файл', title: 'Удалить из сессии', onclick: () => deleteUpload(u), icon: ['trash', 14] }))));
  }
  if (!rows.length) rows.push(h('p', { class: 'panel-empty', style: 'margin:0', text: 'Прикрепите файл к вопросу — он доступен только в этом диалоге и удаляется через сутки.' }));
  el.append(...rows);
}

function renderAccessCard() {
  const el = $('#access-card');
  el.replaceChildren();
  el.append(h('h2', { text: 'Доступ агента' }));
  const p = S.policy;
  const c = corpusInfo();
  const row = (label, text, cls, title) => h('div', { class: 'access-row', title: title || '' }, h('span', { text: label }), h('span', { class: `pill ${cls}`, text }));
  el.append(row(c ? `Папка ${basename(c.root)}` : 'Рабочая папка', c ? 'чтение' : 'не выбрана', c ? 'ok' : 'tag', 'Агент читает файлы корпуса и ничего в них не меняет'));
  el.append(row('Запись файлов', 'спрашивать', 'warn', 'Изменения код-агента попадают в папку только по кнопке «Применить»'));
  let sandbox;
  if (p?.sandbox !== 'no_network') sandbox = ['нет Docker', 'danger', 'Запустите Docker Desktop, чтобы агент мог исполнять код'];
  else if (S.settings.code === 'off') sandbox = ['не запускать', 'tag', 'Выполнение кода выключено в параметрах'];
  else if (S.sandboxNet && p?.sandbox_network !== 'never') sandbox = ['сеть по запросу', 'warn', 'Код исполняется в контейнере Docker; доступ в интернет — только после вашего подтверждения для каждой задачи'];
  else sandbox = ['без сети', 'tag', 'Код исполняется в контейнере Docker без сети, с лимитами'];
  el.append(row('Песочница', ...sandbox));
  const web = S.web ? (S.settings.webMode === 'always' ? ['всегда', 'accent'] : ['адаптивно', 'accent']) : ['выключен', 'tag'];
  el.append(row('Интернет', ...web, 'Наружу уходят только поисковые запросы из вопроса и загрузки страниц; запрос с именами из ваших файлов требует подтверждения'));
}

function renderPanel() {
  renderTraceCard();
  renderFilesCard();
  renderAccessCard();
  for (const art of document.querySelectorAll('.answer[data-turn]')) {
    art.classList.toggle('selected', selectedAnswer() && +art.dataset.turn === selectedAnswer().id);
  }
}

function renderAll() {
  renderFolder();
  renderDialogs();
  renderModel();
  renderHeader();
  renderControls();
  renderThread();
  renderPanel();
}

// --------------------------------------------------------------------------- sending

function askBody(text, opts = {}) {
  const st = S.settings;
  const from = opts.from || {};
  return {
    question: text,
    dialog_id: S.dialog.id,
    session: S.dialog.id,
    route: opts.replaceLast ? (from.route || S.route) : S.route,
    reasoning: opts.replaceLast ? (from.reasoning || S.reasoning) : S.reasoning,
    web: opts.replaceLast ? (from.web || 'off') : (S.web ? st.webMode : 'off'),
    uploads: opts.replaceLast ? (from.uploads || []) : [...S.attachments],
    agent: st.agent || null,
    top_k: st.top_k || null,
    mode: st.mode || null,
    rerank: st.rerank,
    symbols: st.symbols,
    code: opts.replaceLast ? (from.code || st.code || null) : (st.code || null),
    // a confirmation re-asks exactly as asked: the same network toggle and model
    sandbox_net: opts.replaceLast ? !!from.sandbox_net : S.sandboxNet,
    model: opts.replaceLast ? (from.model || null) : S.model,
    confirmed: opts.confirmed || [],
    replace_last: !!opts.replaceLast,
    upload_fallback: true,  // files stay attached: a question they do not answer is routed as usual
  };
}

async function ask(text, opts = {}) {
  return send(text, 'ask', opts);
}

async function send(text, kind = 'ask', opts = {}) {
  text = (text || '').trim();
  if (!text || S.busy) return;
  // busy at once: a second Enter while the dialog is being created must not send the question twice
  S.error = null;
  S.busy = { kind, text, uploads: opts.replaceLast ? [] : [...S.attachments], started: Date.now(), replace: !!opts.replaceLast };
  if (!opts.keepInput) {
    $('#msg').value = '';
    autoGrow();
  }
  try {
    await ensureDialog();
  } catch (e) {
    S.busy = null;
    $('#msg').value = text;
    autoGrow();
    toast(e.message, 'error');
    return;
  }
  if (opts.replaceLast && S.dialog.turns.length && S.dialog.turns[S.dialog.turns.length - 1].role === 'assistant') {
    S.dialog.turns = S.dialog.turns.slice(0, -1);
  }
  renderAll();
  scrollToBottom();
  try {
    if (kind === 'code') await api('POST', '/code', { task: text, dialog_id: S.dialog.id, model: S.model });
    else await api('POST', '/ask', askBody(text, opts));
    // attached files stay in the dialog's context for the next questions until the user removes them
    await reloadDialog();
    S.selected = null;
  } catch (e) {
    S.error = { kind, text, message: e.message };
    await reloadDialog().catch(() => {});
  } finally {
    S.busy = null;
    renderAll();
    scrollToBottom();
    loadDialogs().catch(() => {});
  }
}

// --------------------------------------------------------------------------- uploads

async function uploadFiles(files) {
  if (!files.length) return;
  try {
    await ensureDialog();
  } catch (e) {
    toast(e.message, 'error');
    return;
  }
  renderHeader();
  for (const file of files) {
    const entry = { key: Math.random(), name: file.name, error: null };
    S.uploading.push(entry);
    renderControls();
    const form = new FormData();
    form.append('file', file, file.name);
    try {
      const info = await api('POST', `/uploads?session=${encodeURIComponent(S.dialog.id)}`, undefined, { form });
      S.uploading = S.uploading.filter((x) => x !== entry);
      S.attachments.push(info.id);
      if (info.duplicate_of) toast(`${info.name} уже есть в корпусе: ${info.duplicate_of}`);
    } catch (e) {
      entry.error = e.message;
      toast(`${file.name}: ${e.message}`, 'error');
    }
    await loadUploads();
    renderControls();
    renderPanel();
  }
  loadDialogs().catch(() => {});
}

async function deleteUpload(u) {
  const ok = await guarded(() => api('DELETE', `/uploads/${encodeURIComponent(u.id)}?session=${encodeURIComponent(S.dialog.id)}`), 'Файл удалён из сессии');
  if (ok) {
    S.attachments = S.attachments.filter((x) => x !== u.id);
    await loadUploads();
    renderControls();
    renderPanel();
  }
}

async function addToCorpus(u) {
  const res = await guarded(() => api('POST', `/uploads/${encodeURIComponent(u.id)}/corpus?session=${encodeURIComponent(S.dialog.id)}`), `${u.name} добавлен в рабочую папку`);
  if (res) await refreshStatus();
}

// --------------------------------------------------------------------------- modal

const modal = $('#modal');
function openModal(title, body, footer = [], { wide = false } = {}) {
  modal.replaceChildren(h('div', { class: 'modal-inner' },
    h('div', { class: 'modal-head' }, h('h2', { id: 'modal-title', text: title }),
      h('button', { type: 'button', class: 'mini-btn', 'aria-label': 'Закрыть', onclick: closeModal, icon: ['x', 16, 2] })),
    h('div', { class: 'modal-body' }, body),
    footer.length ? h('div', { class: 'modal-foot' }, footer) : null));
  modal.style.width = wide ? 'min(1000px, calc(100vw - 32px))' : '';
  if (!modal.open) modal.showModal();
}
function closeModal() { if (modal.open) modal.close(); }
modal.addEventListener('click', (e) => { if (e.target === modal) closeModal(); });

function openSourceModal(s) {
  const kv = h('dl', { class: 'kv' },
    h('dt', { text: s.file_type === 'web' ? 'Страница' : 'Файл' }),
    h('dd', {}, s.file_type === 'web' ? h('a', { href: s.file_path, target: '_blank', rel: 'noopener noreferrer nofollow', text: s.file_path }) : h('span', { class: 'mono', text: s.file_path })),
    s.location ? [h('dt', { text: 'Место' }), h('dd', { text: s.location })] : null,
    s.title && s.file_type !== 'web' ? [h('dt', { text: 'Фрагмент' }), h('dd', { text: s.title })] : null,
    h('dt', { text: 'Релевантность' }), h('dd', { text: (s.score ?? 0).toFixed(3) }));
  openModal(`${s.n} · ${sourceKind(s)} · ${sourceName(s)}`, [kv,
    s.file_type === 'web' ? h('p', { class: 'small muted', style: 'margin:0', text: 'Текст веб-страницы — недоверенный источник: агент не выполняет инструкций из него.' }) : null,
    h('pre', { class: 'frag', text: s.text || '' })]);
}

function openSourcesModal(a) {
  const list = h('div', { class: 'list' }, (a.sources || []).map((s) => h('button', { type: 'button', onclick: () => openSourceModal(s) },
    h('span', { class: `cite${s.file_type === 'web' ? ' web' : ''}`, text: String(s.n) }),
    h('span', { style: 'display:flex;flex-direction:column;min-width:0' }, h('b', { text: `${sourceName(s)}${s.cited ? '' : ' · не процитирован'}` }), h('span', { class: 'path', text: `${sourceLoc(s)} · релевантность ${(s.score ?? 0).toFixed(3)}` })))));
  openModal(`Найденные фрагменты (${(a.sources || []).length})`, list);
}

function openTraceModal(a) {
  const steps = h('div', { class: 'trace-steps' }, (a.trace || []).map((st) => h('details', {},
    h('summary', {}, h('b', { text: st.name }), h('span', { class: 'mono muted', text: fmtSec(st.duration_s) })),
    h('pre', { text: JSON.stringify(st.detail || {}, null, 2) }))));
  const kv = h('dl', { class: 'kv' },
    h('dt', { text: 'Вопрос' }), h('dd', { text: a.standalone_question || a.question || '' }),
    h('dt', { text: 'Модель' }), h('dd', { class: 'mono', text: a.model || '' }),
    h('dt', { text: 'Маршрут' }), h('dd', { text: routeLabel(a) }),
    h('dt', { text: 'Время' }), h('dd', { text: fmtSec(a.latency_s) }),
    h('dt', { text: 'Подтверждён ссылками' }), h('dd', { text: a.grounded === null || a.grounded === undefined ? '—' : a.grounded ? 'да' : 'нет' }));
  openModal('Полная трасса ответа', [kv, steps], [], { wide: true });
}

async function openCatalogModal() {
  const data = await guarded(() => api('GET', '/catalog'));
  if (!data) return;
  const rows = [];
  for (const e of data.experiments || []) {
    for (const m of e.metrics || []) rows.push([e.file_path, e.dataset, e.model, m.name, m.split, m.variant, m.value, m.cell]);
  }
  const table = h('div', { class: 'table-wrap' }, h('table', {},
    h('thead', {}, h('tr', {}, ['ноутбук', 'датасет', 'модель', 'метрика', 'выборка', 'вариант', 'значение', 'ячейка'].map((c) => h('th', { text: c })))),
    h('tbody', {}, rows.map((r) => h('tr', {}, r.map((c) => h('td', { text: c === null || c === undefined ? '' : String(c) })))))));
  openModal(`Каталог экспериментов (${data.experiments_count ?? (data.experiments || []).length})`, [
    h('p', { class: 'small muted', style: 'margin:0', text: `Извлечено из ${data.notebooks ?? '?'} ноутбуков: ${data.values ?? rows.length} значений метрик и гиперпараметров, каждое сверено со своей ячейкой; отброшено непроверенных — ${data.dropped ?? 0}. Агрегатные вопросы («лучший», «сколько», «все запуски») отвечаются SQL-запросом к этому каталогу.` }),
    table], [], { wide: true });
}

async function openFolderModal() {
  const c = corpusInfo();
  const input = h('input', { class: 'input mono', id: 'root-input', value: c?.root || '', placeholder: 'C:\\Users\\…\\WorkSpace' });
  const typesBox = h('div', { class: 'checks' });
  const excl = h('textarea', { class: 'textarea', id: 'excl-input', rows: 6 });
  const force = h('input', { type: 'checkbox' });
  const recent = h('div', { class: 'list' });
  const problems = h('div');

  const fillPrefs = (p) => {
    typesBox.replaceChildren(...Object.entries(TYPE_LABELS).map(([ext, label]) => h('label', { class: 'check' },
      h('input', { type: 'checkbox', value: ext, checked: (p.include_ext || []).includes(ext) }), `${label} (${ext})`)));
    excl.value = (p.exclude || []).join('\n');
  };
  fillPrefs(c?.prefs || S.status?.defaults || {});
  const loadPrefsFor = async (root) => {
    if (!root) return;
    const p = await api('GET', `/corpus/prefs?root=${encodeURIComponent(root)}`).catch(() => null);
    if (p) fillPrefs(p);
  };
  input.addEventListener('change', () => loadPrefsFor(input.value.trim()));

  const corpora = await api('GET', '/corpora').catch(() => []);
  for (const k of corpora) {
    if (c && k.root === c.root) continue;
    recent.append(h('button', { type: 'button', onclick: async () => {
      const res = await guarded(() => api('POST', '/corpus/open', { root: k.root }), `Открыта папка ${basename(k.root)}`);
      if (res) { closeModal(); await refreshStatus(); }
    } }, h('span', { class: 'folder-icon', icon: 'folder' }), h('span', { style: 'display:flex;flex-direction:column;min-width:0' }, h('b', { text: basename(k.root) }), h('span', { class: 'path', text: k.root }))));
  }
  if (c) {
    api('GET', '/index/problems').then((list) => {
      if (!list || !list.length) return;
      problems.append(h('details', {}, h('summary', { class: 'small', text: `Файлы с проблемами при индексации: ${list.length}` }),
        h('div', { class: 'list', style: 'margin-top:8px' }, list.slice(0, 200).map((p) => h('div', { class: 'small' }, h('span', { class: 'mono', text: p.path || p.file_path || '' }), ' — ', p.message || p.error || p.status || '')))));
    }).catch(() => {});
  }
  const pick = h('button', { type: 'button', class: 'btn inline', icon: ['folder', 14, 2], onclick: async () => {
    pick.disabled = true;
    const res = await guarded(() => api('POST', '/pick-folder', { initial: input.value.trim() }));
    pick.disabled = false;
    if (res && res.path) { input.value = res.path; loadPrefsFor(res.path); }
  } }, 'Выбрать…');
  const body = [
    h('div', { class: 'field' }, h('label', { for: 'root-input', text: 'Папка' }), h('div', { class: 'row' }, input, pick),
      h('span', { class: 'hint', text: 'Все данные остаются на этом компьютере. Индекс хранится отдельно для каждой папки.' })),
    recent.childElementCount ? h('div', { class: 'field' }, h('span', { class: 'label', text: 'Ранее проиндексированные' }), recent) : null,
    h('div', { class: 'field' }, h('span', { class: 'label', text: 'Типы файлов' }), typesBox),
    h('div', { class: 'field' }, h('label', { for: 'excl-input', text: 'Исключения' }), excl,
      h('span', { class: 'hint', text: 'По одному шаблону в строке. Без «/» — любая папка или файл с таким именем (например, «Аккаунты», «*.egg-info»); с «/» — путь (например, «data/raw/**»). Скрытые папки пропускаются всегда.' })),
    h('label', { class: 'switch-row' }, h('span', { text: 'Переиндексировать всё с нуля' }), h('span', { class: 'switch' }, force, h('span'))),
    problems,
  ];
  const go = h('button', { type: 'button', class: 'btn accent', icon: ['refresh', 14, 2], onclick: async () => {
    const root = input.value.trim();
    if (!root) { toast('Укажите папку', 'error'); return; }
    const include = [...typesBox.querySelectorAll('input:checked')].map((x) => x.value);
    if (!include.length) { toast('Выберите хотя бы один тип файлов', 'error'); return; }
    const exclude = excl.value.split('\n').map((x) => x.trim()).filter(Boolean);
    closeModal();
    await startIndex(root, { include_ext: include, exclude }, force.checked);
  } }, 'Индексировать');
  openModal('Рабочая папка', body, [h('button', { type: 'button', class: 'btn ghost', onclick: closeModal }, 'Отмена'), go]);
}

function openSettingsModal() {
  const st = S.settings;
  const retr = S.status?.retrieval || {};
  const themePref = (() => { try { return localStorage.getItem('rag.theme') || 'system'; } catch { return 'system'; } })();
  const accentIdx = (() => { try { return parseInt(localStorage.getItem('rag.accent') || '0', 10) || 0; } catch { return 0; } })();
  const themeSeg = h('div', { class: 'segmented', role: 'group', 'aria-label': 'Тема' });
  const drawTheme = (v) => segmented(themeSeg, { system: 'Как в системе', light: 'Светлая', dark: 'Тёмная' }, v, (k) => { setTheme(k); drawTheme(k); drawAccents(); });
  drawTheme(themePref);
  const accents = h('div', { class: 'accents', role: 'group', 'aria-label': 'Цвет акцента' });
  const drawAccents = () => {
    const theme = document.documentElement.dataset.theme;
    let cur = 0;
    try { cur = parseInt(localStorage.getItem('rag.accent') || '0', 10) || 0; } catch { /* ignore */ }
    accents.replaceChildren(...window.ragTheme.ACCENTS[theme].map((color, i) => h('button', { type: 'button', 'aria-label': `Акцент ${i + 1}`, 'aria-pressed': String(i === cur), style: `background:${color}`, onclick: () => {
      try { localStorage.setItem('rag.accent', String(i)); } catch { /* ignore */ }
      window.ragTheme.apply();
      drawAccents();
    } })));
  };
  drawAccents();
  void accentIdx;

  const agentSeg = h('div', { class: 'segmented', role: 'group', 'aria-label': 'Агент' });
  const drawAgent = (v) => segmented(agentSeg, { off: 'Выкл', auto: 'Авто', always: 'Всегда' }, v, (k) => { st.agent = k; savePrefs(); drawAgent(k); });
  drawAgent(st.agent || 'auto');
  const codeSeg = h('div', { class: 'segmented', role: 'group', 'aria-label': 'Выполнение кода' });
  const drawCode = (v) => segmented(codeSeg, { auto: 'Когда нужно', off: 'Не запускать' }, v, (k) => { st.code = k; savePrefs(); drawCode(k); renderControls(); renderAccessCard(); });
  drawCode(st.code || 'auto');
  const webSeg = h('div', { class: 'segmented', role: 'group', 'aria-label': 'Интернет' });
  const drawWeb = (v) => segmented(webSeg, { auto: 'Когда нужен', always: 'Всегда искать' }, v, (k) => { st.webMode = k; savePrefs(); drawWeb(k); renderControls(); renderAccessCard(); });
  drawWeb(st.webMode || 'auto');
  const modeSeg = h('div', { class: 'segmented', role: 'group', 'aria-label': 'Режим поиска' });
  const drawMode = (v) => segmented(modeSeg, { dense: 'Плотный', sparse: 'Разреженный', hybrid: 'Гибридный' }, v, (k) => { st.mode = k; savePrefs(); drawMode(k); });
  drawMode(st.mode || retr.mode || 'dense');
  const topVal = h('b', { class: 'mono', text: String(st.top_k || retr.top_k || 6) });
  const top = h('input', { type: 'range', min: 2, max: 12, value: st.top_k || retr.top_k || 6, 'aria-label': 'Фрагментов в контексте' });
  top.addEventListener('input', () => { st.top_k = +top.value; topVal.textContent = top.value; savePrefs(); });
  const sw = (label, key, def, hint) => {
    const inp = h('input', { type: 'checkbox', checked: st[key] ?? def });
    inp.addEventListener('change', () => { st[key] = inp.checked; savePrefs(); });
    return h('label', { class: 'switch-row', title: hint }, h('span', { text: label }), h('span', { class: 'switch' }, inp, h('span')));
  };
  openModal('Параметры', [
    h('div', { class: 'field' }, h('span', { class: 'label', text: 'Тема' }), themeSeg),
    h('div', { class: 'field' }, h('span', { class: 'label', text: 'Акцент' }), accents),
    h('div', { class: 'field' }, h('span', { class: 'label', text: 'Агент' }), agentSeg,
      h('span', { class: 'hint', text: 'Агент сам вызывает поиск, точный поиск имён, SQL к каталогу, чтение файлов и расчёт в песочнице. «Авто» — только для агрегатных и многошаговых вопросов.' })),
    h('div', { class: 'field' }, h('span', { class: 'label', text: 'Выполнение кода' }), codeSeg,
      h('span', { class: 'hint', text: 'Агент сам решает, когда запустить код: задачу с файлами (построить график, исправить ошибку, запустить ноутбук) выполняет код-агент в копии папки, расчёт для ответа — в песочнице. В вашу папку изменения попадают только по кнопке «Применить». Сеть для кода включается переключателем под полем ввода.' })),
    h('div', { class: 'field' }, h('span', { class: 'label', text: 'Интернет, когда включён' }), webSeg,
      h('span', { class: 'hint', text: '«Когда нужен» — только для свежих и внешних фактов или когда в файлах ничего нет. Запрос с именами из ваших файлов всегда требует подтверждения.' })),
    h('div', { class: 'field' }, h('span', { class: 'label', text: 'Поиск по файлам' }), modeSeg,
      h('div', { class: 'range-row' }, h('span', { class: 'small', text: 'Фрагментов в ответе' }), top, topVal),
      sw('Реранкер', 'rerank', retr.rerank ?? true, 'Кросс-энкодер переупорядочивает найденное: точнее, но медленнее'),
      sw('Точный поиск имён', 'symbols', retr.symbols ?? true, 'Для имён функций и классов из вопроса добавляет их определения и места вызова')),
  ], [
    h('button', { type: 'button', class: 'btn ghost', onclick: () => { S.settings = { agent: 'auto', webMode: 'auto', code: 'auto', top_k: null, mode: null, rerank: null, symbols: null }; savePrefs(); closeModal(); renderControls(); renderAccessCard(); toast('Параметры сброшены'); } }, 'По умолчанию'),
    h('button', { type: 'button', class: 'btn accent', onclick: closeModal }, 'Готово'),
  ]);
}

async function openModelModal() {
  if (!S.models) await guarded(loadModels);
  const def = S.models?.default || shortModel(S.status?.llm?.name);
  const cur = S.model || def;
  const list = h('div', { class: 'list models' });
  for (const m of S.models?.models || []) {
    const meta = [m.parameters, m.quantization, m.size ? fmtSize(m.size) : ''].filter(Boolean).join(' · ');
    list.append(h('button', { type: 'button', 'aria-pressed': String(m.name === cur), onclick: () => {
      S.model = m.name === def ? null : m.name;
      savePrefs();
      closeModal();
      renderModel();
      toast(`Модель: ${m.name}`);
    } },
    h('span', { class: 'folder-icon', icon: m.name === cur ? 'check' : 'chip' }),
    h('span', { style: 'display:flex;flex-direction:column;min-width:0' },
      h('b', {}, m.name, m.name === def ? h('span', { class: 'tag', style: 'margin-left:8px', text: 'по умолчанию' }) : null),
      h('span', { class: 'path', text: meta }))));
  }
  if (!list.childElementCount) list.append(h('p', { class: 'small muted', style: 'margin:0', text: 'Ollama не отвечает или в ней нет моделей: ollama pull qwen3.5:9b' }));
  openModal('Модель', [
    list,
    h('p', { class: 'hint', style: 'margin:0' }, `Выбранная модель отвечает на новые вопросы в этом браузере и работает код-агентом. Индексация и каталог экспериментов остаются на модели по умолчанию (${def}), на ней же сделаны все замеры в отчётах. Две модели одновременно в память видеокарты могут не поместиться: Ollama выгрузит прежнюю, и первый ответ новой будет дольше.`),
  ], [h('button', { type: 'button', class: 'btn accent', onclick: closeModal }, 'Готово')]);
}

function renameDialog(d) {
  const input = h('input', { class: 'input', value: d.title || '', 'aria-label': 'Название' });
  const save = async () => {
    const res = await guarded(() => api('PATCH', `/dialogs/${d.id}`, { title: input.value }));
    if (res) {
      closeModal();
      if (S.dialog && S.dialog.id === d.id) S.dialog.title = res.title;
      await loadDialogs();
      renderHeader();
    }
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') save(); });
  openModal('Переименовать диалог', input, [h('button', { type: 'button', class: 'btn ghost', onclick: closeModal }, 'Отмена'), h('button', { type: 'button', class: 'btn accent', onclick: save }, 'Сохранить')]);
  setTimeout(() => input.select(), 30);
}

function deleteDialog(d) {
  openModal('Удалить диалог?', h('p', { style: 'margin:0' }, `«${d.title || 'Без названия'}» и загруженные в него файлы будут удалены. Файлы рабочей папки не затрагиваются.`), [
    h('button', { type: 'button', class: 'btn ghost', onclick: closeModal }, 'Отмена'),
    h('button', { type: 'button', class: 'btn accent', onclick: async () => {
      const ok = await guarded(() => api('DELETE', `/dialogs/${d.id}`));
      closeModal();
      if (ok) {
        if (S.dialog && S.dialog.id === d.id) newDialog();
        await loadDialogs();
      }
    } }, 'Удалить'),
  ]);
}

// --------------------------------------------------------------------------- status polling

async function refreshStatus() {
  const before = indexing();
  const rootBefore = corpusInfo()?.root || null;
  try {
    await loadStatus();
  } catch (e) {
    return;
  }
  const rootChanged = (corpusInfo()?.root || null) !== rootBefore;
  const now = indexing();
  if (before && !now) {
    const p = S.status.progress;
    if (p.state === 'done') toast(`Индексация завершена: новых ${p.new}, изменённых ${p.changed}, без изменений ${p.unchanged}`);
    else if (p.state === 'error') toast('Индексация завершилась с ошибкой', 'error');
    loadPolicy().catch(() => {});
  }
  renderFolder();
  renderModel();
  renderHeader();
  if (rootChanged && !S.dialog?.turns?.length && !S.busy) renderThread(); // the welcome text names the folder
  if (rootChanged) renderAccessCard();
}

function schedulePoll() {
  setTimeout(async () => {
    await refreshStatus();
    schedulePoll();
  }, indexing() ? 1000 : 10000);
}

// --------------------------------------------------------------------------- events

function autoGrow() {
  const t = $('#msg');
  t.style.height = 'auto';
  t.style.height = `${Math.min(220, t.scrollHeight)}px`;
  renderSend();
}

function closeSidebar() {
  $('#sidebar').classList.remove('open');
  $('#panel').classList.remove('open');
  $('#scrim').hidden = true;
}

function bindEvents() {
  $('#new-dialog').addEventListener('click', newDialog);
  $('#composer').addEventListener('submit', (e) => { e.preventDefault(); send($('#msg').value); });
  $('#msg').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (!$('#send').disabled) send($('#msg').value);
    }
  });
  $('#msg').addEventListener('input', autoGrow);
  $('#attach').addEventListener('click', () => $('#file-input').click());
  $('#file-input').addEventListener('change', (e) => { uploadFiles([...e.target.files]); e.target.value = ''; });
  $('#folder-btn').addEventListener('click', openFolderModal);
  $('#web-toggle').addEventListener('click', () => { S.web = !S.web; savePrefs(); renderControls(); renderAccessCard(); });
  $('#net-toggle').addEventListener('click', () => {
    S.sandboxNet = !S.sandboxNet;
    savePrefs();
    renderControls();
    renderAccessCard();
    if (S.sandboxNet) toast('Код сможет выходить в интернет — каждый такой запуск нужно будет разрешить');
  });
  $('#open-trace').addEventListener('click', () => { const t = selectedAnswer(); if (t) openTraceModal(t.payload); else toast('Ответов в этом диалоге пока нет'); });
  $('#open-sidebar').addEventListener('click', () => { $('#sidebar').classList.add('open'); $('#scrim').hidden = false; });
  $('#close-sidebar').addEventListener('click', closeSidebar);
  $('#open-panel').addEventListener('click', () => { $('#panel').classList.add('open'); $('#scrim').hidden = false; });
  $('#scrim').addEventListener('click', closeSidebar);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSidebar(); });
  const comp = $('#composer');
  comp.addEventListener('dragover', (e) => { e.preventDefault(); comp.classList.add('drag'); });
  comp.addEventListener('dragleave', () => comp.classList.remove('drag'));
  comp.addEventListener('drop', (e) => { e.preventDefault(); comp.classList.remove('drag'); uploadFiles([...e.dataTransfer.files]); });
}

async function boot() {
  initIcons();
  bindEvents();
  renderControls();
  try {
    await Promise.all([loadStatus(), loadPolicy().catch(() => {}), loadModels().catch(() => {}), loadDialogs()]);
  } catch (e) {
    toast(e.message, 'error');
  }
  if (indexing()) S.wasIndexing = true;
  if (S.dialogs.length) await openDialog(S.dialogs[0].id);
  else renderAll();
  schedulePoll();
}

boot();
