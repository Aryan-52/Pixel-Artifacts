import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type ModelInfo } from "../lib/api";
import { genName, int, pct } from "../lib/format";
import { CountUp, Reveal } from "../components/Reveal";
import { LensArt, LENS_COPY } from "../components/LensArt";

const PixelGlobe = lazy(() => import("../components/PixelGlobe"));

interface Sample { file: string; generator: string; label: "ai" | "real"; p_ai: number }

function SpotTheAI() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [round, setRound] = useState(0);
  const [picked, setPicked] = useState<number | null>(null);
  const [score, setScore] = useState({ you: 0, model: 0, played: 0 });
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1000));
  useEffect(() => { fetch("/samples/samples.json").then((r) => r.json()).then(setSamples).catch(() => setSamples([])); }, []);

  const rounds = useMemo(() => {
    const gens = Array.from(new Set(samples.map((s) => s.generator)));
    let x = seed;
    const rnd = () => ((x = (x * 9301 + 49297) % 233280) / 233280);
    return gens.sort(() => rnd() - 0.5).slice(0, 5).map((g) => {
      const pair = samples.filter((s) => s.generator === g);
      return rnd() < 0.5 ? pair : [...pair].reverse();
    });
  }, [samples, seed]);

  if (!rounds.length) return null;
  const done = round >= rounds.length;
  const pair = rounds[Math.min(round, rounds.length - 1)];
  const choose = (i: number) => {
    if (picked !== null) return;
    setPicked(i);
    const modelPick = pair[0].p_ai >= pair[1].p_ai ? 0 : 1;
    setScore((s) => ({ you: s.you + (pair[i].label === "ai" ? 1 : 0), model: s.model + (pair[modelPick].label === "ai" ? 1 : 0), played: s.played + 1 }));
  };
  const next = () => { setPicked(null); setRound((r) => r + 1); };
  const restart = () => { setPicked(null); setRound(0); setScore({ you: 0, model: 0, played: 0 }); setSeed((s) => s + 7); };

  return (
    <div className="game card">
      <div className="game-head">
        <div>
          <span className="eyebrow">Round {Math.min(round + 1, rounds.length)} of {rounds.length}</span>
          <h3>{done ? "How did you do?" : "One of these is AI-generated. Which one?"}</h3>
        </div>
        <div className="game-score">
          <div><b>{score.you}</b><span>you</span></div>
          <div><b>{score.model}</b><span>detector</span></div>
        </div>
      </div>
      {done ? (
        <div className="game-end">
          <p className="lede">You spotted <b>{score.you} of {score.played}</b>. The detector spotted <b>{score.model} of {score.played}</b>{" "}
            by picking the image it scored as more AI-like. These images come from a benchmark of real research generators, picked at random.</p>
          <div className="row"><button className="btn" onClick={restart}>Play again</button><Link className="btn ghost" to="/detect">Check your own image</Link></div>
        </div>
      ) : (
        <>
          <div className="game-pair">
            {pair.map((s, i) => {
              const reveal = picked !== null;
              return (
                <button key={s.file} className={`game-img ${reveal ? (s.label === "ai" ? "is-ai" : "is-real") : ""} ${picked === i ? "picked" : ""}`} onClick={() => choose(i)} disabled={reveal}>
                  <img src={`/${s.file}`} alt={`Option ${i + 1}`} />
                  {reveal && (
                    <span className="game-tag">
                      <span className={`chip ${s.label}`}>{s.label === "ai" ? `AI · ${genName(s.generator)}` : "Real photo"}</span>
                      <span className="small">Detector: {pct(s.p_ai, 0)} AI</span>
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          <div className="game-foot">
            <p className="small muted">{picked === null ? "Click the image you think is fake." : pair[picked].label === "ai" ? "Correct! " : "Not this time. "}
              {picked !== null && (pair.find((p) => p.label === "ai")!.p_ai >= 0.5 ? "The detector caught it too." : "The detector missed this one: some generators are much harder than others.")}</p>
            {picked !== null && <button className="btn sm" onClick={next}>{round + 1 < rounds.length ? "Next round" : "See result"}</button>}
          </div>
        </>
      )}
    </div>
  );
}

const STEPS = [
  { n: "01", t: "Drop your image", d: "One photo, a dozen, a whole folder or a ZIP. Nothing to install, no account." },
  { n: "02", t: "Five lenses look at it", d: "Texture, hidden frequency patterns, colour, light and noise, and the overall look are each examined separately." },
  { n: "03", t: "A clear answer, with proof", d: "A verdict in plain words first. Heatmaps, spectra and benchmark statistics are there when you want them." },
];

const FEATURES = [
  { t: "One image or many", d: "Up to 12 images get a full report each; switch between them in one view.", icon: "M6 9h14v12H6zM10 5h14v12" },
  { t: "Folders & ZIP files", d: "Thousands of images at once, with a summary dashboard and a downloadable CSV.", icon: "M4 9a2 2 0 012-2h5l2 2h9a2 2 0 012 2v9a2 2 0 01-2 2H6a2 2 0 01-2-2z" },
  { t: "Grad-CAM heatmaps", d: "See where the detector looked, side by side with the image.", icon: "M5 5h18v18H5zM9 15l3-4 3 3 2-2 3 4" },
  { t: "Forensic views", d: "Frequency spectrum, noise residual and error-level analysis of the same region.", icon: "M4 14h4l2-6 4 12 3-9 2 3h5" },
  { t: "Test on your own data", d: "Label folders real / fake and get accuracy, AUC, a ROC curve and a confusion matrix.", icon: "M5 21V9m6 12V5m6 16v-8m6 8V11" },
  { t: "Metadata check", d: "Flags files whose metadata names an AI tool or stores a generation prompt.", icon: "M7 4h10l4 4v14H7zM11 12h6M11 16h6" },
];

export default function Landing() {
  const [m, setM] = useState<ModelInfo | null>(null);
  useEffect(() => { api.model().then(setM).catch(() => setM(null)); }, []);
  const b = m?.benchmark;
  const best = b ? Object.entries(b.per_generator).filter(([g]) => g !== "progan").sort((x, y) => y[1].accuracy - x[1].accuracy).slice(0, 6) : [];

  return (
    <>
      <section className="hero">
        <div className="wrap hero-grid">
          <div className="hero-copy">
            <Reveal><span className="eyebrow">AI image forensics</span></Reveal>
            <Reveal delay={0.08}><h1>Every pixel <em>keeps</em> a record.</h1></Reveal>
            <Reveal delay={0.16}>
              <p className="lede">Pixel-Artifacts tells you whether an image is a real photo or AI-generated. It reads the faint traces image generators leave behind, and shows you the evidence.</p>
            </Reveal>
            <Reveal delay={0.24} className="row hero-cta">
              <Link to="/detect" className="btn">Check an image <span aria-hidden>→</span></Link>
              <a href="#how" className="btn ghost">See how it works</a>
            </Reveal>
            <Reveal delay={0.32} className="hero-trust">
              <span><b>{b ? pct(b.accuracy) : "83.2%"}</b> accuracy on unseen generators</span>
              <span><b>{b ? b.auc.toFixed(3) : "0.943"}</b> ROC-AUC</span>
              <span><b>No sign-up</b> needed</span>
            </Reveal>
          </div>
          <div className="hero-visual">
            <Suspense fallback={null}><PixelGlobe /></Suspense>
            <div className="float-card fc-1"><span className="dot" style={{ color: "var(--coral)" }} /> Upsampling grid found</div>
            <div className="float-card fc-2"><span className="mono">p(AI)</span> <b>0.97</b></div>
            <div className="float-card fc-3"><span className="dot" style={{ color: "var(--sage)" }} /> Noise pattern: camera-like</div>
          </div>
        </div>
      </section>

      <section className="section-tight">
        <div className="wrap marquee-wrap">
          <span className="small muted">Tested on images from</span>
          <div className="marquee"><div>
            {[...Array(2)].flatMap((_, k) => ["ProGAN", "StyleGAN", "StyleGAN2", "BigGAN", "CycleGAN", "StarGAN", "GauGAN", "CRN", "IMLE", "SAN", "SeeingDark", "Deepfake", "WhichFaceIsReal"].map((g) => <span key={`${k}${g}`}>{g}</span>))}
          </div></div>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal className="section-head">
            <span className="eyebrow">Try it yourself</span>
            <h2>Can you spot the AI?</h2>
            <p className="lede">Most people can't, at least not reliably. Play five rounds against the detector.</p>
          </Reveal>
          <Reveal><SpotTheAI /></Reveal>
        </div>
      </section>

      <section className="section dark" id="how">
        <div className="wrap">
          <Reveal className="section-head">
            <span className="eyebrow">How it works</span>
            <h2>Three steps. No jargon required.</h2>
          </Reveal>
          <div className="grid-3 steps-3">
            {STEPS.map((s, i) => (
              <Reveal key={s.n} delay={i * 0.1} className="step-card">
                <span className="step-n">{s.n}</span>
                <h3>{s.t}</h3>
                <p className="muted">{s.d}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <Reveal className="section-head">
            <span className="eyebrow">The five lenses</span>
            <h2>One image, examined five different ways.</h2>
            <p className="lede">No single clue is reliable on its own. A learned gate weighs all five for every image, so a weak clue never decides alone.</p>
          </Reveal>
          <div className="lenses">
            {LENS_COPY.map((l, i) => (
              <Reveal key={l.key} delay={i * 0.07} className="lens-card">
                <LensArt k={l.key} />
                <h4>{l.name}</h4>
                <p className="small muted">{l.plain}</p>
                <span className="chip">{l.tech}</span>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="section" style={{ paddingTop: 0 }}>
        <div className="wrap">
          <Reveal className="section-head">
            <span className="eyebrow">What you can do</span>
            <h2>Built for one photo or ten thousand.</h2>
          </Reveal>
          <div className="grid-3 features">
            {FEATURES.map((f, i) => (
              <Reveal key={f.t} delay={(i % 3) * 0.08} className="feature">
                <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden><path d={f.icon} fill="none" stroke="var(--teal)" strokeWidth="1.8" strokeLinejoin="round" strokeLinecap="round" /></svg>
                <h4>{f.t}</h4>
                <p className="small muted">{f.d}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="section proof">
        <div className="wrap proof-grid">
          <Reveal className="stack" style={{ ["--gap" as string]: "18px" }}>
            <span className="eyebrow">Measured, not claimed</span>
            <h2>Tested on generators it never saw.</h2>
            <p className="lede">Trained on one GAN (ProGAN), then tested on {b ? int(b.images) : "90,329"} images from {b?.generators ?? 13} different generators. No test image was used to train or tune it.</p>
            <Link to="/research/results" className="link-btn">Explore the full results →</Link>
          </Reveal>
          <div className="stack" style={{ ["--gap" as string]: "28px" }}>
            <div className="grid-3">
              <Reveal className="stat"><b><CountUp value={(b?.accuracy ?? 0.8315) * 100} decimals={1} suffix="%" /></b><span>zero-shot accuracy</span></Reveal>
              <Reveal delay={0.08} className="stat"><b><CountUp value={b?.auc ?? 0.943} decimals={3} /></b><span>ROC-AUC</span></Reveal>
              <Reveal delay={0.16} className="stat"><b><CountUp value={(b?.real_acc ?? 0.97) * 100} decimals={1} suffix="%" /></b><span>real photos kept real</span></Reveal>
            </div>
            <Reveal className="proof-bars">
              {best.map(([g, s]) => (
                <div key={g} className="proof-bar">
                  <span>{genName(g)}</span>
                  <i style={{ width: `${s.accuracy * 100}%` }} />
                  <b>{pct(s.accuracy, 0)}</b>
                </div>
              ))}
            </Reveal>
          </div>
        </div>
      </section>

      <section className="section-tight">
        <div className="wrap">
          <Reveal className="cta-band">
            <div>
              <h2>Have an image you're not sure about?</h2>
              <p className="lede">Free to use and nothing to sign up for. Images are analysed and then discarded.</p>
            </div>
            <Link to="/detect" className="btn light">Check it now →</Link>
          </Reveal>
        </div>
      </section>
    </>
  );
}
