import { useState } from "react";
import { ALL_TAGS, PAPERS, type Tag } from "../../data/literature";
import { Reveal } from "../../components/Reveal";

export default function Literature() {
  const [tag, setTag] = useState<Tag | "All">("All");
  const [open, setOpen] = useState<number | null>(0);
  const list = PAPERS.map((p, i) => ({ p, i })).filter(({ p }) => tag === "All" || p.tags.includes(tag));
  return (
    <div className="stack" style={{ ["--gap" as string]: "28px" }}>
      <div className="row">
        {(["All", ...ALL_TAGS] as const).map((t) => (
          <button key={t} className={`filter-chip ${tag === t ? "on" : ""}`} onClick={() => setTag(t)}>
            {t}{t !== "All" && <span>{PAPERS.filter((p) => p.tags.includes(t)).length}</span>}
          </button>
        ))}
      </div>
      <div className="papers">
        {list.map(({ p, i }, k) => (
          <Reveal key={p.title} delay={Math.min(k, 6) * 0.04} className={`paper ${open === i ? "open" : ""}`}>
            <button className="paper-head" onClick={() => setOpen(open === i ? null : i)} aria-expanded={open === i}>
              <span className="paper-n">{String(i + 1).padStart(2, "0")}</span>
              <span className="paper-title">
                <b>{p.title}</b>
                <span className="small muted">{p.authors} · {p.venue}</span>
              </span>
              <span className="paper-tags">{p.tags.map((t) => <span key={t} className="chip plum">{t}</span>)}</span>
            </button>
            {open === i && (
              <div className="paper-body">
                <div><h4>Relevance</h4><p>{p.relevance}</p></div>
                <div><h4>Effect on our work</h4><p>{p.effect}</p></div>
                <div><h4>How we differ</h4><p>{p.difference}</p></div>
                <a className="btn ghost sm" href={p.link} target="_blank" rel="noreferrer">Open paper ↗</a>
              </div>
            )}
          </Reveal>
        ))}
      </div>
    </div>
  );
}
