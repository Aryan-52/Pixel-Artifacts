import { useEffect, useRef, useState } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";

export function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden>
      <i /><i /><i /><i />
    </span>
  );
}

const RESEARCH = [
  { to: "/research/methodology", label: "Methodology", hint: "Five lenses, data and training protocol" },
  { to: "/research/results", label: "Results", hint: "Benchmarks, per-GAN, versions, ablation" },
  { to: "/research/hybrid", label: "Hybrid model", hint: "Four-branch gated hybrid and its results" },
  { to: "/research/literature", label: "Literature", hint: "The papers this work builds on" },
  { to: "/research/publications", label: "Publications & roadmap", hint: "Blackbook, paper, gaps, future scope" },
];

function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [drop, setDrop] = useState(false);
  const dropRef = useRef<HTMLDivElement>(null);
  const loc = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  useEffect(() => { setOpen(false); setDrop(false); }, [loc.pathname]);
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (dropRef.current && !dropRef.current.contains(e.target as Node)) setDrop(false);
    };
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);

  const inResearch = loc.pathname.startsWith("/research");
  return (
    <header className={`nav ${scrolled ? "scrolled" : ""} ${open ? "open" : ""}`}>
      <div className="wrap nav-inner">
        <Link to="/" className="brand"><BrandMark /> Pixel-Artifacts</Link>
        <button className="nav-burger" aria-label="Menu" aria-expanded={open} onClick={() => setOpen(!open)}>
          <svg width="20" height="14" viewBox="0 0 20 14"><path d="M0 1h20M0 7h20M0 13h20" stroke="currentColor" strokeWidth="2" /></svg>
        </button>
        <nav className="nav-links" aria-label="Main">
          <NavLink to="/detect">Detect</NavLink>
          <div className={`nav-drop ${drop ? "open" : ""}`} ref={dropRef}>
            <button aria-expanded={drop} onClick={() => setDrop(!drop)} className={inResearch ? "active" : ""}
              style={inResearch ? { background: "var(--teal-soft)", color: "var(--teal-ink)" } : undefined}>
              Research
              <svg width="10" height="6" viewBox="0 0 10 6"><path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.6" fill="none" /></svg>
            </button>
            <div className="nav-menu" role="menu">
              {RESEARCH.map((r) => (
                <NavLink key={r.to} to={r.to} role="menuitem">{r.label}<small>{r.hint}</small></NavLink>
              ))}
            </div>
          </div>
          <NavLink to="/guide">Guide</NavLink>
          <NavLink to="/history">History</NavLink>
          <NavLink to="/team">Team</NavLink>
          <Link to="/detect" className="btn sm nav-cta">Check an image</Link>
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <div className="wrap">
        <div className="footer-grid">
          <div className="stack" style={{ ["--gap" as string]: "14px" }}>
            <Link to="/" className="brand"><BrandMark /> Pixel-Artifacts</Link>
            <p className="small" style={{ maxWidth: 340 }}>
              Checks whether an image is a real photo or AI-generated, and shows the evidence behind every answer.
              Images are analysed on this server and are not stored after a batch expires.
            </p>
          </div>
          <div>
            <h4>Product</h4>
            <Link to="/detect">Check images</Link>
            <Link to="/detect?mode=folder">Folder or ZIP</Link>
            <Link to="/history">History</Link>
            <Link to="/guide">Reading results</Link>
          </div>
          <div>
            <h4>Research</h4>
            {RESEARCH.map((r) => <Link key={r.to} to={r.to}>{r.label}</Link>)}
          </div>
          <div>
            <h4>Project</h4>
            <Link to="/team">Team & mentors</Link>
            <span className="small" style={{ display: "block", paddingTop: 4 }}>MCA Major Project<br />Sardar Patel Institute of Technology</span>
          </div>
        </div>
        <div className="footer-bottom">
          <span>Pixel-Artifacts · model fp-v4.1</span>
          <span>A detector gives evidence, not proof. Use results with care.</span>
        </div>
      </div>
    </footer>
  );
}

export default function Layout() {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return (
    <>
      <Nav />
      <main><Outlet /></main>
      <Footer />
    </>
  );
}
