import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type Analysis, type Job, type JobItem } from "../lib/api";
import { duration, int, num, pct } from "../lib/format";
import { rememberJob } from "../lib/history";
import { BarList, Confusion, Donut, RocChart } from "../components/Charts";
import { ResultView } from "../components/Result";

type Filter = "all" | "ai" | "real" | "low" | "wrong" | "error";

function Progress({ job }: { job: Job }) {
  const k = job.total ? job.done / job.total : 0;
  const r = 54, L = 2 * Math.PI * r;
  return (
    <div className="card progress-card">
      <svg viewBox="0 0 140 140" width="150" aria-hidden>
        <circle cx="70" cy="70" r={r} stroke="var(--paper-2)" strokeWidth="12" fill="none" />
        <circle cx="70" cy="70" r={r} stroke="var(--teal)" strokeWidth="12" fill="none" strokeLinecap="round"
          strokeDasharray={`${k * L} ${L}`} transform="rotate(-90 70 70)" style={{ transition: "stroke-dasharray .6s" }} />
        <text x="70" y="78" textAnchor="middle" fontSize="26" fontFamily="Fraunces" fill="var(--ink)">{Math.round(k * 100)}%</text>
      </svg>
      <div className="stack">
        <span className="eyebrow">{job.status === "queued" ? "Waiting in line" : "Analysing"}</span>
        <h2>{int(job.done)} of {int(job.total)} images</h2>
        <p className="muted">{job.eta_seconds != null ? `About ${duration(job.eta_seconds)} left.` : "Starting up…"} You can leave this page open; the report appears here when it is done.</p>
      </div>
    </div>
  );
}

function Hist({ counts }: { counts: number[] }) {
  const m = Math.max(...counts, 1);
  return (
    <div className="mini-hist">
      {counts.map((c, i) => (
        <div key={i} title={`${i * 10}-${i * 10 + 10}%: ${c}`}>
          <span style={{ height: `${(c / m) * 100}%`, background: i >= 5 ? "var(--coral)" : "var(--sage)" }} />
        </div>
      ))}
      <small>0%</small><small style={{ gridColumn: "10" }}>100%</small>
    </div>
  );
}

export default function Batch() {
  const { id = "" } = useParams();
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<"index" | "p_desc" | "p_asc">("index");
  const [open, setOpen] = useState<{ item: JobItem; a?: Analysis; err?: string } | null>(null);

  useEffect(() => {
    let stop = false, t: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const j = await api.job(id);
        if (stop) return;
        setJob(j);
        if (j.status === "done") rememberJob(j);
        if (j.status === "queued" || j.status === "running") t = setTimeout(poll, 1000);
      } catch (e) { if (!stop) setErr((e as Error).message); }
    };
    poll();
    return () => { stop = true; clearTimeout(t); };
  }, [id]);

  useEffect(() => {
    if (!open || open.a || open.err) return;
    api.jobItem(id, open.item.index).then((a) => setOpen((o) => o && { ...o, a })).catch((e) => setOpen((o) => o && { ...o, err: e.message }));
  }, [open, id]);

  const rows = useMemo(() => {
    if (!job) return [];
    let r = job.items.filter((it) =>
      filter === "all" ? true : filter === "error" ? !!it.error : filter === "low" ? it.band === "low"
        : filter === "wrong" ? it.correct === false : it.verdict === filter);
    if (q) r = r.filter((it) => it.path.toLowerCase().includes(q.toLowerCase()));
    if (sort !== "index") r = [...r].sort((a, b) => ((a.p_ai ?? -1) - (b.p_ai ?? -1)) * (sort === "p_asc" ? 1 : -1));
    return r;
  }, [job, filter, q, sort]);

  if (err) return <section className="wrap page-head"><h1>Batch not found</h1><p className="lede">{err}</p><p style={{ marginTop: 20 }}><Link className="btn" to="/detect?mode=folder">Start a new batch</Link></p></section>;
  if (!job) return <section className="wrap page-head"><div className="skeleton" style={{ height: 200 }} /></section>;

  const s = job.summary;
  const L = s?.labelled;
  return (
    <>
      <section className="wrap page-head">
        <div className="crumbs"><Link to="/detect?mode=folder">Detector</Link><span>/</span><span>Batch report</span></div>
        <span className="eyebrow">{job.source === "zip" ? "ZIP archive" : "Folder"} · {job.created.replace("T", " ")}</span>
        <h1 style={{ maxWidth: "none" }}>{job.name}</h1>
      </section>

      <section className="wrap">
        {job.status !== "done" && job.status !== "failed" && <Progress job={job} />}
        {job.status === "failed" && <div className="error-box">This batch could not be finished. Please try again.</div>}

        {s && (
          <div className="stack" style={{ ["--gap" as string]: "28px" }}>
            <div className="card batch-hero">
              <Donut ai={s.ai} real={s.real} />
              <div className="batch-stats">
                <div className="stat"><b>{int(s.scored)}</b><span>images analysed</span></div>
                <div className="stat"><b style={{ color: "var(--coral)" }}>{int(s.ai)}</b><span>look AI-generated</span></div>
                <div className="stat"><b style={{ color: "var(--sage)" }}>{int(s.real)}</b><span>look real</span></div>
                <div className="stat"><b style={{ color: "var(--saffron)" }}>{int(s.bands.low)}</b><span>uncertain</span></div>
              </div>
              <div className="batch-actions">
                <a className="btn" href={api.csvUrl(job.id)} download>Download CSV report</a>
                <p className="small muted">{s.errors > 0 && `${s.errors} unreadable · `}{s.skipped > 0 && `${s.skipped} non-image files skipped · `}done in {duration(job.seconds)}</p>
              </div>
            </div>

            <div className="grid-3 align-stretch">
              <div className="card flat stack">
                <h4>Score spread</h4>
                <p className="small muted">How many images fell in each 10% band of AI probability.</p>
                <Hist counts={s.histogram} />
              </div>
              <div className="card flat stack">
                <h4>Confidence</h4>
                <BarList max={Math.max(s.scored, 1)} format={(v) => int(v)} rows={[
                  { label: "High", value: s.bands.high, color: "var(--teal)" },
                  { label: "Moderate", value: s.bands.moderate, color: "var(--saffron)" },
                  { label: "Uncertain", value: s.bands.low, color: "var(--coral)" }]} />
                {s.with_ai_markers > 0 && <p className="note warn small">{s.with_ai_markers} files name an AI tool in their metadata.</p>}
              </div>
              <div className="card flat stack">
                <h4>Lens usage</h4>
                <p className="small muted">Average fusion weight across the batch.</p>
                <BarList max={0.4} rows={Object.entries(s.mean_gates).map(([k, v]) => ({ label: k[0].toUpperCase() + k.slice(1), value: v }))} />
              </div>
            </div>

            {L && (
              <div className="card">
                <div className="section-head" style={{ marginBottom: 24 }}>
                  <span className="eyebrow">Your labelled data</span>
                  <h3>Folder names told us the true answer for {int(L.images)} images</h3>
                  <p className="muted small">Images inside folders named real / fake (or 0_real / 1_fake) are scored against that label. {int(L.real)} real, {int(L.ai)} AI.</p>
                </div>
                <div className="grid-3 align-start">
                  <div className="mini-stats col">
                    <div><b>{pct(L.accuracy)}</b><span>accuracy</span></div>
                    <div><b>{L.auc != null ? num(L.auc) : "–"}</b><span>ROC-AUC</span></div>
                    <div><b>{pct(L.precision)}</b><span>precision</span></div>
                    <div><b>{pct(L.recall)}</b><span>recall (AI caught)</span></div>
                    <div><b>{pct(L.specificity)}</b><span>real kept real</span></div>
                    <div><b>{num(L.f1)}</b><span>F1 score</span></div>
                  </div>
                  <Confusion {...L.confusion} />
                  {L.roc ? <RocChart fpr={L.roc.fpr} tpr={L.roc.tpr} auc={L.roc.auc} /> : <p className="small muted">A ROC curve needs both real and AI images.</p>}
                </div>
              </div>
            )}

            {s.folders.length > 1 && (
              <div className="card flat">
                <h4 style={{ marginBottom: 14 }}>By folder</h4>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Folder</th><th className="num">Images</th><th className="num">Flagged AI</th><th className="num">Share</th><th className="num">Mean AI prob.</th></tr></thead>
                    <tbody>{s.folders.map((f) => (
                      <tr key={f.folder}><td>{f.folder}</td><td className="num">{int(f.images)}</td><td className="num">{int(f.ai)}</td>
                        <td className="num">{pct(f.ai / f.images)}</td><td className="num">{pct(f.mean_p_ai)}</td></tr>))}</tbody>
                  </table>
                </div>
              </div>
            )}

            <div className="card flat">
              <div className="table-tools">
                <div className="tabs">
                  {([["all", "All"], ["ai", "AI"], ["real", "Real"], ["low", "Uncertain"], ...(L ? [["wrong", "Wrong"]] : []), ...(s.errors ? [["error", "Errors"]] : [])] as [Filter, string][]).map(([k, l]) => (
                    <button key={k} aria-selected={filter === k} onClick={() => setFilter(k)}>{l}</button>
                  ))}
                </div>
                <input className="search" placeholder="Search file names" value={q} onChange={(e) => setQ(e.target.value)} />
                <select className="search" value={sort} onChange={(e) => setSort(e.target.value as typeof sort)} aria-label="Sort">
                  <option value="index">Original order</option>
                  <option value="p_desc">Most AI-like first</option>
                  <option value="p_asc">Most real-like first</option>
                </select>
              </div>
              <div className="table-wrap" style={{ maxHeight: 620 }}>
                <table>
                  <thead><tr><th></th><th>File</th><th>Verdict</th><th className="num">AI prob.</th><th>Confidence</th>{L && <th>Label</th>}<th className="num">Size</th></tr></thead>
                  <tbody>
                    {rows.slice(0, 500).map((it) => (
                      <tr key={it.index} className="clickable" onClick={() => !it.error && setOpen({ item: it })}>
                        <td><img className="thumb" src={api.thumbUrl(job.id, it.index)} alt="" loading="lazy" onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} /></td>
                        <td style={{ whiteSpace: "normal", maxWidth: 360 }}><b>{it.filename}</b><br /><span className="small muted">{it.path}</span></td>
                        <td>{it.error ? <span className="chip low">{it.error}</span> : <span className={`chip ${it.verdict}`}>{it.verdict === "ai" ? "AI-generated" : "Real"}</span>}</td>
                        <td className="num">{pct(it.p_ai)}</td>
                        <td>{it.band && <span className={`chip ${it.band === "low" ? "low" : "teal"}`}>{it.band}</span>}</td>
                        {L && <td>{it.ground_truth == null ? "–" : <span className={it.correct ? "ok" : "bad"}>{it.ground_truth ? "AI" : "real"} {it.correct ? "✓" : "✗"}</span>}</td>}
                        <td className="num">{it.width ? `${it.width}x${it.height}` : ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {rows.length > 500 && <p className="small muted" style={{ marginTop: 10 }}>Showing the first 500 of {int(rows.length)}. The CSV has every image.</p>}
            </div>
          </div>
        )}
      </section>

      {open && (
        <div className="drawer-back" onClick={() => setOpen(null)}>
          <aside className="drawer" onClick={(e) => e.stopPropagation()} aria-label="Image analysis">
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 18 }}>
              <span className="eyebrow">{open.item.path}</span>
              <button className="btn ghost sm" onClick={() => setOpen(null)}>Close</button>
            </div>
            {open.err ? <div className="error-box">{open.err}</div> : open.a ? <ResultView a={open.a} /> : <div className="skeleton" style={{ height: 420 }} />}
          </aside>
        </div>
      )}
      <div className="spacer" />
    </>
  );
}
