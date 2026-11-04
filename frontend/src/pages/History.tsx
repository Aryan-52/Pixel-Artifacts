import { useState } from "react";
import { Link } from "react-router-dom";
import { clearHistory, readHistory, removeEntry } from "../lib/history";
import { pct } from "../lib/format";

export default function History() {
  const [list, setList] = useState(readHistory);
  const refresh = () => setList(readHistory());
  return (
    <>
      <section className="wrap page-head">
        <span className="eyebrow">History</span>
        <h1>Your recent checks</h1>
        <p className="lede">Stored only in this browser. Nothing here is uploaded or shared. Batch reports stay on the server for a few hours.</p>
      </section>
      <section className="wrap">
        {list.length === 0 ? (
          <div className="card empty">
            <svg width="90" height="70" viewBox="0 0 90 70" aria-hidden><rect x="8" y="10" width="60" height="48" rx="8" fill="#ebe2d2" /><rect x="20" y="4" width="62" height="48" rx="8" fill="none" stroke="#0f5c68" strokeWidth="2" /><path d="M30 40l12-12 9 9 7-6 12 10" stroke="#0f5c68" strokeWidth="2" fill="none" /></svg>
            <h3>Nothing yet</h3>
            <p className="muted">Images you check will appear here.</p>
            <Link className="btn" to="/detect">Check an image</Link>
          </div>
        ) : (
          <>
            <div className="row" style={{ justifyContent: "space-between", marginBottom: 18 }}>
              <span className="muted small">{list.length} entries</span>
              <button className="btn ghost sm" onClick={() => { clearHistory(); refresh(); }}>Clear history</button>
            </div>
            <div className="history">
              {list.map((e) => (
                <div key={e.id} className="h-item">
                  {e.kind === "image"
                    ? (e.thumb ? <img src={e.thumb} alt="" /> : <span className="h-icon">IMG</span>)
                    : <span className="h-icon">{e.images ?? "?"}</span>}
                  <div className="h-meta">
                    <b>{e.name}</b>
                    <span className="small muted">{new Date(e.date).toLocaleString()}</span>
                  </div>
                  {e.kind === "image" && e.verdict && (
                    <span className={`chip ${e.verdict}`}>{e.verdict === "ai" ? "AI" : "Real"} · {pct(e.verdict === "ai" ? e.p_ai : 1 - (e.p_ai ?? 0), 0)}</span>
                  )}
                  {e.kind === "batch" && (
                    <>
                      <span className="chip ai">{e.ai ?? 0} flagged AI</span>
                      <Link className="btn ghost sm" to={`/batch/${e.jobId}`}>Open report</Link>
                    </>
                  )}
                  <button className="link-btn small" onClick={() => { removeEntry(e.id); refresh(); }} aria-label={`Remove ${e.name}`}>Remove</button>
                </div>
              ))}
            </div>
          </>
        )}
      </section>
      <div className="spacer" /><div className="spacer" />
    </>
  );
}
