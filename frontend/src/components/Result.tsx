import { Fragment, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Analysis, type ModelInfo } from "../lib/api";
import { bandWord, bytes, download, genName, int, num, pct } from "../lib/format";
import { BarList, Diverging, Gauge, RocChart, ScoreHistogram } from "./Charts";

function useModel() {
  const [m, setM] = useState<ModelInfo | null>(null);
  useEffect(() => { api.model().then(setM).catch(() => setM(null)); }, []);
  return m;
}

/** Original image with the whole-image scan drawn over it: every 224 px window coloured by its
 * score (coral = AI-like, green = real-like), and the most AI-like window outlined. */
function AnalysedImage({ a }: { a: Analysis }) {
  const { width: w, height: h, focus_box: box } = a.image;
  const { windows, size } = a.scan;
  const labels = windows.length <= 16;
  return (
    <div className="analysed">
      <img src={a.visuals?.original} alt={a.filename} />
      {windows.map((t, i) => (
        <span key={i} className="tile" style={{
          left: `${(t.x / w) * 100}%`, top: `${(t.y / h) * 100}%`,
          width: `${(size / w) * 100}%`, height: `${(size / h) * 100}%`,
          background: t.p_ai >= 0.5 ? `rgba(212,81,42,${0.12 + t.p_ai * 0.4})` : `rgba(60,122,89,${0.08 + (1 - t.p_ai) * 0.22})`,
        }}>{labels && <b>{Math.round(t.p_ai * 100)}%</b>}</span>
      ))}
      {box && (
        <span className="crop-box" style={{ left: `${(box[0] / w) * 100}%`, top: `${(box[1] / h) * 100}%`,
          width: `${(box[2] / w) * 100}%`, height: `${(box[3] / h) * 100}%` }}>
          <em>most AI-like</em>
        </span>
      )}
    </div>
  );
}

function ScanNote({ a }: { a: Analysis }) {
  const n = a.scan.windows.length;
  if (!n) return <>The image is smaller than 224 pixels on one side, so it was enlarged and read whole.</>;
  return (
    <>Whole image scanned: {n} region{n > 1 ? "s" : ""} of 224 x 224 px at full resolution, covering every pixel of
      this {a.image.width} x {a.image.height} image. {pct(a.scan.share_ai, 0)} of regions look AI-generated (highest
      {" "}{pct(a.scan.max_p_ai, 0)}); the verdict is the average over all regions.</>
  );
}

function Compare({ before, after, labels }: { before: string; after: string; labels: [string, string] }) {
  const [x, setX] = useState(55);
  return (
    <div className="compare">
      <img src={before} alt={labels[0]} />
      <div className="compare-top" style={{ clipPath: `inset(0 ${100 - x}% 0 0)` }}><img src={after} alt={labels[1]} /></div>
      <span className="compare-line" style={{ left: `${x}%` }} />
      <input type="range" min={0} max={100} value={x} onChange={(e) => setX(+e.target.value)} aria-label="Compare slider" />
      <span className="compare-tag left">{labels[1]}</span>
      <span className="compare-tag right">{labels[0]}</span>
    </div>
  );
}

const VIEWS = [
  { key: "gradcam", title: "Grad-CAM attention", body: "Where the texture lens (fine-tuned EfficientNet-B0) looked when it formed its opinion. Warm areas mattered most. It shows attention, not proof of an edit in that spot." },
  { key: "spectrum", title: "Frequency spectrum", body: "The image's Fourier spectrum. Generator upsampling often leaves bright dots or a grid pattern away from the centre; camera photos usually fade smoothly." },
  { key: "noise", title: "Noise residual", body: "What is left after removing the content. Cameras leave fine, random grain; generators can leave smooth patches or repeating textures." },
  { key: "ela", title: "Error level analysis", body: "How each region reacts to one JPEG re-save. Patches that react very differently from the rest were often processed differently." },
] as const;

function VisualTab({ a }: { a: Analysis }) {
  const v = a.visuals!;
  return (
    <div className="stack" style={{ ["--gap" as string]: "28px" }}>
      <div className="grid-2 align-start">
        <div className="stack">
          <h4>Attention overlay</h4>
          <p className="small muted">The most AI-like 224 px region. Drag the slider to compare it with the Grad-CAM heatmap.</p>
          <Compare before={v.crop} after={v.gradcam} labels={["region", "Grad-CAM"]} />
          {a.spatial_expert_p_ai != null && (
            <p className="small">The texture lens on its own rates this region <b>{pct(a.spatial_expert_p_ai)}</b> AI-generated.</p>
          )}
        </div>
        <div className="stack">
          <h4>Region map</h4>
          <p className="small muted"><ScanNote a={a} /> The outlined region is the one shown on the left and in the views below.</p>
          <AnalysedImage a={a} />
        </div>
      </div>
      <div className="views">
        {VIEWS.map((x) => (
          <figure key={x.key} className="view">
            <img src={v[x.key]} alt={x.title} />
            <figcaption><b>{x.title}</b><span>{x.body}</span></figcaption>
          </figure>
        ))}
      </div>
      {a.histograms && (
        <div className="stack">
          <h4>Colour channel histograms</h4>
          <div className="rgb">
            {(["r", "g", "b"] as const).map((ch) => {
              const vals = a.histograms![ch], m = Math.max(...vals);
              return (
                <svg key={ch} viewBox="0 0 160 60" width="100%" aria-label={`${ch} histogram`}>
                  {vals.map((val, i) => (
                    <rect key={i} x={i * 5} y={60 - (val / m) * 56} width="4" height={(val / m) * 56}
                      fill={ch === "r" ? "#d4512a" : ch === "g" ? "#3c7a59" : "#0f5c68"} opacity=".8" rx="1" />
                  ))}
                </svg>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function InternalsTab({ a, m }: { a: Analysis; m: ModelInfo | null }) {
  return (
    <div className="grid-2 align-start">
      <div className="stack">
        <h4>Fusion weights and influence</h4>
        <p className="small muted">The gate decides how much each lens's evidence is used for this image (weights add up to 100%).
          Influence is how much the score (logit) would change if that lens were removed.</p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Lens</th><th className="num">Weight</th><th className="num">Influence</th></tr></thead>
            <tbody>
              {a.lenses.map((l) => (
                <tr key={l.key}><td>{l.name}<br /><span className="small muted">{l.tech}</span></td>
                  <td className="num">{pct(l.weight)}</td>
                  <td className="num" style={{ color: l.delta_logit >= 0 ? "var(--coral)" : "var(--sage)" }}>{l.delta_logit >= 0 ? "+" : ""}{num(l.delta_logit, 3)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="stack">
        <h4>Raw numbers</h4>
        <dl className="kv">
          <dt>AI probability</dt><dd>{num(a.p_ai, 4)}</dd>
          <dt>Logit</dt><dd>{num(a.logit, 3)}</dd>
          <dt>Decision threshold</dt><dd>{a.threshold} (fixed, never tuned on test data)</dd>
          <dt>Temperature</dt><dd>{m?.temperature ?? 1}</dd>
          <dt>Texture lens alone</dt><dd>{pct(a.spatial_expert_p_ai)}</dd>
          <dt>Model</dt><dd>{a.model_version}{m ? ` on ${m.device.toUpperCase()}` : ""}</dd>
          <dt>Analysis time</dt><dd>{Math.round(a.inference_ms)} ms</dd>
          <dt>Input</dt><dd>{m ? `${m.image_size} x ${m.image_size} centre crop, no resizing` : "224 x 224 centre crop"}</dd>
        </dl>
        <p className="small muted">The fusion weights describe how the model combined its evidence; they are not proof that a specific artifact exists.</p>
      </div>
    </div>
  );
}

function SourceTab({ a }: { a: Analysis }) {
  const s = a.source;
  if (!s) return <p className="muted">Source attribution is not installed on this server.</p>;
  const best = s.ranked[0];
  return (
    <div className="grid-2 align-start">
      <div className="stack">
        <h4>Which generator does it resemble?</h4>
        <div className="source-head">
          <span className="small muted">Most likely source</span>
          <b>{s.top ? genName(s.top) : "Unknown source"}</b>
          <span className="small muted">{s.top ? `${pct(best.p, 0)} likelihood`
            : `No known generator reaches ${pct(s.unknown_below, 0)}; closest is ${genName(best.generator)} (${pct(best.p, 0)})`}</span>
        </div>
        <BarList rows={s.ranked.map((r, i) => ({ label: genName(r.generator), value: r.p,
          color: i === 0 && s.top ? "var(--plum)" : "var(--line-2)" }))} />
      </div>
      <div className="stack">
        <h4>How to read this</h4>
        {a.verdict === "real" && (
          <p className="note">The detector judged this image a real photo. The source guess below only matters if the image is in fact AI-generated.</p>
        )}
        <ul className="ticks small">
          <li>This answers: <i>if</i> the image is AI-generated, which of {s.ranked.length} known generators made it?</li>
          <li>It is a separate model that reads the same five-lens features. It never changes the real-vs-AI verdict.</li>
          <li>Trained on Tiny-GenImage and ProGAN images, not on the zero-shot benchmark. It names the right generator for {pct(s.test_accuracy, 0)} of its held-out test images.</li>
          <li>It only knows these generators. Images from other tools (e.g. DALL·E 3, Flux) are shown as unknown or as the closest match.</li>
        </ul>
      </div>
    </div>
  );
}

function EvidenceTab({ a, m }: { a: Analysis; m: ModelInfo | null }) {
  if (!m) return <div className="skeleton" style={{ height: 260 }} />;
  const b = m.benchmark;
  const gens = Object.entries(b.per_generator).sort((x, y) => y[1].accuracy - x[1].accuracy);
  return (
    <div className="stack" style={{ ["--gap" as string]: "32px" }}>
      <div className="grid-2 align-start">
        <div className="stack">
          <h4>Where this score sits</h4>
          <p className="small muted">Scores the detector gave {int(b.images)} benchmark images from {b.generators} generators it never saw in training.</p>
          <ScoreHistogram real={b.histogram.real} ai={b.histogram.ai} marker={a.p_ai} />
          <p className="small">
            {a.evidence.bin_share_ai != null && <>Of the {int(a.evidence.bin_images)} benchmark images that scored between {pct(a.evidence.bin[0], 0)} and {pct(a.evidence.bin[1], 0)},
            <b> {pct(a.evidence.bin_share_ai)}</b> were actually AI-generated. </>}
            This score is higher than {pct(a.evidence.real_scored_lower, 0)} of real images and {pct(a.evidence.ai_scored_lower, 0)} of AI images in the benchmark.
          </p>
        </div>
        <div className="stack">
          <h4>Overall reliability (ROC)</h4>
          <RocChart fpr={b.roc.fpr} tpr={b.roc.tpr} auc={b.roc.auc} point={b.operating_point} />
          <div className="mini-stats">
            <div><b>{pct(b.accuracy)}</b><span>accuracy</span></div>
            <div><b>{num(b.auc)}</b><span>ROC-AUC</span></div>
            <div><b>{num(b.ap)}</b><span>avg. precision</span></div>
            <div><b>{pct(b.real_acc)}</b><span>real kept real</span></div>
            <div><b>{pct(b.fake_acc)}</b><span>AI caught</span></div>
          </div>
        </div>
      </div>
      <div className="stack">
        <h4>Accuracy by image generator</h4>
        <p className="small muted">Generators differ a lot. If you know where an image came from, this tells you how far to trust the result.</p>
        <BarList rows={gens.map(([g, s]) => ({ label: genName(g), value: s.accuracy,
          color: s.accuracy >= 0.85 ? "var(--teal)" : s.accuracy >= 0.7 ? "var(--saffron)" : "var(--coral)" }))} />
        <p className="small"><Link to="/research/results">Full benchmark results →</Link></p>
      </div>
    </div>
  );
}

function MetaTab({ a }: { a: Analysis }) {
  const md = a.metadata;
  return (
    <div className="grid-2 align-start">
      <div className="stack">
        <h4>Metadata clues</h4>
        <div className={md.ai_markers.length ? "note warn" : "note"}>{md.note}</div>
        {md.ai_markers.length > 0 && <div className="row">{md.ai_markers.map((x) => <span className="chip ai" key={x}>{x}</span>)}</div>}
        {Object.keys(md.exif).length > 0 && (
          <dl className="kv">{Object.entries(md.exif).map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v}</dd></Fragment>)}</dl>
        )}
        {Object.keys(md.text_chunks).length > 0 && (
          <details><summary className="small">Embedded text ({Object.keys(md.text_chunks).length})</summary>
            <dl className="kv">{Object.entries(md.text_chunks).map(([k, v]) => <Fragment key={k}><dt>{k}</dt><dd>{v}</dd></Fragment>)}</dl>
          </details>
        )}
      </div>
      <div className="stack">
        <h4>File</h4>
        <dl className="kv">
          <dt>Name</dt><dd>{a.filename}</dd>
          <dt>Format</dt><dd>{md.format ?? "unknown"} · {md.mode}</dd>
          <dt>Size</dt><dd>{bytes(a.bytes)}</dd>
          <dt>Dimensions</dt><dd>{a.image.width} x {a.image.height} px</dd>
          <dt>Regions scanned</dt><dd>{a.image.windows} x (224 x 224 px){a.image.focus_box ? ` · most AI-like at x ${a.image.focus_box[0]}, y ${a.image.focus_box[1]}` : " · whole image (enlarged)"}</dd>
          <dt>SHA-256</dt><dd className="mono" style={{ wordBreak: "break-all" }}>{a.sha256}</dd>
        </dl>
      </div>
    </div>
  );
}

const TABS = [["visual", "Visual forensics"], ["source", "Likely source"], ["internals", "Model internals"], ["evidence", "Benchmark evidence"], ["meta", "Metadata & file"]] as const;

export function ResultView({ a }: { a: Analysis }) {
  const m = useModel();
  const [adv, setAdv] = useState(false);
  const [tab, setTab] = useState<(typeof TABS)[number][0]>("visual");
  useEffect(() => { setAdv(false); setTab("visual"); }, [a.sha256]);

  if (a.error) return <div className="error-box">{a.filename}: {a.error}</div>;
  const ai = a.verdict === "ai";
  const lenses = [...a.lenses].sort((x, y) => Math.abs(y.delta_logit) - Math.abs(x.delta_logit));
  return (
    <article className="result">
      <div className={`result-hero card ${ai ? "is-ai" : "is-real"}`}>
        <div className="result-media">
          <AnalysedImage a={a} />
          <p className="small muted" style={{ marginTop: 10 }}><ScanNote a={a} /></p>
        </div>
        <div className="result-verdict">
          <div className="row">
            <span className={`chip ${ai ? "ai" : "real"}`}><span className="dot" />{ai ? "AI-generated" : "Real photo"}</span>
            <span className={`chip ${a.band === "low" ? "low" : "teal"}`}>{bandWord[a.band]}</span>
          </div>
          <h2>{a.summary.headline}</h2>
          <div className="gauge-row">
            <Gauge value={a.p_ai} size={220} />
            <div className="prob-pair">
              <div><b style={{ color: "var(--coral)" }}>{pct(a.p_ai, 1)}</b><span>chance AI-generated</span></div>
              <div><b style={{ color: "var(--sage)" }}>{pct(a.p_real, 1)}</b><span>chance real</span></div>
            </div>
          </div>
          <p>{a.summary.lead} {a.summary.why}</p>
          <p className={a.band === "low" ? "note" : "small muted"}>{a.summary.caution}</p>
          {a.metadata.ai_markers.length > 0 && (
            <p className="note warn">The file's metadata also names an AI tool: {a.metadata.ai_markers.join(", ")}.</p>
          )}
        </div>
      </div>

      <div className="grid-2 align-stretch">
        <div className="card flat">
          <h3>What the five lenses saw</h3>
          <p className="small muted" style={{ margin: "6px 0 18px" }}>Each lens looks at the image differently. Bars show which way each one pushed the decision.</p>
          <Diverging rows={lenses.map((l) => ({ label: <span title={l.plain}>{l.name}</span>, value: l.delta_logit }))} />
        </div>
        <div className="card flat">
          <h3>How much to trust this</h3>
          <div className="trust">
            {a.evidence.bin_share_ai != null && (
              <p><span className="trust-big">{pct(ai ? a.evidence.bin_share_ai : 1 - a.evidence.bin_share_ai, 0)}</span>
                of {int(a.evidence.bin_images)} test images with a similar score really were {ai ? "AI-generated" : "real photos"}.</p>
            )}
            <p className="small muted">Measured on {m ? int(m.benchmark.images) : "90,329"} images from {m?.benchmark.generators ?? 13} generators the detector never trained on.
              Overall accuracy {m ? pct(m.benchmark.accuracy) : "…"}, ROC-AUC {m ? num(m.benchmark.auc) : "…"}.</p>
            <p className="small muted">It is strongest on GAN images like those it was built for. Images from newer diffusion tools, heavy compression or small crops can fool it.</p>
          </div>
        </div>
      </div>

      <div className="adv-toggle">
        <button className="btn ghost" aria-expanded={adv} onClick={() => setAdv(!adv)}>
          {adv ? "Hide advanced analysis" : "Show advanced analysis"}
          <svg width="12" height="8" viewBox="0 0 12 8" style={{ transform: adv ? "rotate(180deg)" : undefined, transition: ".3s" }}>
            <path d="M1 1l5 5 5-5" stroke="currentColor" strokeWidth="1.8" fill="none" />
          </svg>
        </button>
        <button className="btn ghost" onClick={() => {
          const { visuals: _v, ...rest } = a;
          download(`${a.filename}.pixel-artifacts.json`, JSON.stringify(rest, null, 1));
        }}>Download report (JSON)</button>
        <button className="btn ghost" onClick={() => { setAdv(true); setTimeout(() => window.print(), 300); }}>Print / save PDF</button>
      </div>

      {adv && (
        <section className="card advanced">
          <div className="tabs" role="tablist">
            {TABS.map(([k, label]) => (
              <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{label}</button>
            ))}
          </div>
          <div className="adv-body">
            {tab === "visual" && a.visuals && <VisualTab a={a} />}
            {tab === "source" && <SourceTab a={a} />}
            {tab === "internals" && <InternalsTab a={a} m={m} />}
            {tab === "evidence" && <EvidenceTab a={a} m={m} />}
            {tab === "meta" && <MetaTab a={a} />}
          </div>
        </section>
      )}
    </article>
  );
}
