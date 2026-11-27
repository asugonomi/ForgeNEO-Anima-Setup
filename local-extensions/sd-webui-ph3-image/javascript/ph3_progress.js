// Status independent of the host progressbar preference; completion comes from Gradio.
let ph3ProgressRun = null;
function ph3ProgressElements() {
    const root = gradioApp().querySelector('#ph3_activity');
    return root ? {root, text: root.querySelector('[data-ph3-text]'), bar: root.querySelector('[data-ph3-bar]')} : null;
}
function ph3Finish() {
    const run = ph3ProgressRun;
    ph3ProgressRun = null;
    if (run) { clearTimeout(run.timer); if (run.controller) run.controller.abort(); }
    const el = ph3ProgressElements();
    if (el) { el.root.hidden = true; el.root.setAttribute('aria-busy', 'false'); }
    return [];
}
function ph3Submit(...args) {
    ph3Finish();
    const id = randomId();
    args[args.length - 1] = id;
    const run = {id, started: Date.now(), timer: null, controller: null};
    ph3ProgressRun = run;
    const paint = (text, progress) => {
        const el = ph3ProgressElements();
        if (!el || ph3ProgressRun !== run) return;
        el.root.hidden = false;
        el.root.setAttribute('aria-busy', 'true');
        el.text.textContent = text;
        const value = Math.max(0, Math.min(1, Number(progress) || 0));
        el.bar.style.width = `${value * 100}%`;
        el.bar.parentNode.setAttribute('aria-valuenow', String(Math.round(value * 100)));
    };
    paint('処理を開始しています…', 0);
    const poll = async () => {
        if (ph3ProgressRun !== run) return;
        run.controller = new AbortController();
        const timeout = setTimeout(() => run.controller.abort(), 8000);
        try {
            const response = await fetch('./internal/progress', {method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id_task: id, live_preview: false}), signal: run.controller.signal});
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const state = await response.json();
            if (ph3ProgressRun !== run) return;
            const seconds = Math.floor((Date.now() - run.started) / 1000);
            const phase = state.completed ? '結果を受信しています…' : state.active
                ? (state.textinfo || '処理中…') : state.queued ? '生成待ち…' : '処理の開始待ち…';
            const percent = state.progress > 0 ? ` · ${Math.round(state.progress * 100)}%` : '';
            paint(`${phase}${percent} · 経過 ${seconds}秒`, state.progress);
        } catch (error) {
            if (ph3ProgressRun === run) paint('進捗の応答待ちです。接続を再確認中…', 0);
        } finally {
            clearTimeout(timeout);
            if (ph3ProgressRun === run) run.timer = setTimeout(poll, 700);
        }
    };
    poll();
    return args;
}
