import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api, type Research } from "../../lib/api";

const HEADS: Record<string, { t: string; d: string }> = {
  methodology: { t: "How the detector works", d: "Five independent lenses, a learned gate that weighs them, and a training protocol that never touches the test set." },
  results: { t: "What we measured", d: "Every number here is read from the evaluation files of the trained models. Nothing is retyped by hand." },
  hybrid: { t: "Gated hybrid model", d: "An end-to-end gated hybrid of deep, physics, frequency and CLIP ViT-L/14 branches, trained on ProGAN and evaluated on 13 generators." },
  literature: { t: "Standing on these papers", d: "18 of the 36 papers we reviewed, chosen for how directly they shaped this work: what each contributes, how it affected our design, and how we differ." },
  publications: { t: "Publications & roadmap", d: "The project's written work, the gaps we set out to address, our honest shortcomings, and where this goes next." },
};

export function useResearch() {
  const [r, setR] = useState<Research | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.research().then(setR).catch((e) => setErr(e.message)); }, []);
  return { r, err };
}

export default function ResearchLayout() {
  const seg = useLocation().pathname.split("/")[2] || "methodology";
  const h = HEADS[seg] ?? HEADS.methodology;
  return (
    <>
      <section className="wrap page-head">
        <span className="eyebrow" style={{ color: "var(--plum)" }}>Research & development</span>
        <h1>{h.t}</h1>
        <p className="lede">{h.d}</p>
      </section>
      <section className="wrap">
        <nav className="subnav" aria-label="Research sections">
          <NavLink to="/research/methodology">Methodology</NavLink>
          <NavLink to="/research/results">Results</NavLink>
          <NavLink to="/research/hybrid">Hybrid model</NavLink>
          <NavLink to="/research/literature">Literature</NavLink>
          <NavLink to="/research/publications">Publications & roadmap</NavLink>
        </nav>
        <Outlet />
      </section>
      <div className="spacer" /><div className="spacer" />
    </>
  );
}
