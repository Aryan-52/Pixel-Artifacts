import { useEffect, useMemo, useState } from "react";
import { api, type AblationRow, type ModelInfo, type ProtocolStats, type Research } from "../../lib/api";
import { genName, int, num, pct } from "../../lib/format";
import { BarList, Confusion, RocChart, ScoreHistogram } from "../../components/Charts";
import { Reveal } from "../../components/Reveal";
import { useResearch } from "./ResearchLayout";

const TABS = [["overview", "Benchmark"], ["generators", "Per generator"], ["versions", "Version history"], ["ablation", "Ablation (31 runs)"], ["source", "Source attribution"]] as const;
type Tab = (typeof TABS)[number][0];

function Overview({ m, r }: { m: ModelInfo; r: Research }) {
  const b = m.benchmark;
  const [[tn, fp], [fn, tp]] = b.confusion;
  return (
    <div className="stack" style={{ ["--gap" as string]: "28px" }}>
      <div className="metric-strip">
        {[["Accuracy", pct(b.accuracy)], ["ROC-AUC", num(b.auc)], ["Average precision", num(b.ap)], ["Precision", pct(b.precision)],
          ["Recall (AI caught)", pct(b.recall)], ["Real kept real", pct(b.real_acc)], ["F1", num(b.f1)], ["Validation accuracy", pct(b.validation.accuracy, 2)]]
          .map(([k, v]) => <div key={k}><b>{v}</b><span>{k}</span></div>)}
      </div>
      <p className="note">Zero-shot: trained on ProGAN only, tested on {int(b.images)} images from {b.generators} generators, scored the way the app scores ({b.protocol}). Validation ({int(b.validation.images)} ProGAN images) is in-distribution and only used to pick the best epoch.</p>
      {r.protocols && <Protocols p={r.protocols} />}
      <div className="grid-2 align-start">
        <div className="card flat"><h4>ROC curve</h4><RocChart fpr={b.roc.fpr} tpr={b.roc.tpr} auc={b.roc.auc} point={b.operating_point} /></div>
        <div className="card flat stack"><h4>Confusion matrix (threshold 0.5)</h4><Confusion tn={tn} fp={fp} fn={fn} tp={tp} />
          <p className="small muted">The detector is cautious: it rarely calls a real photo fake ({pct(1 - b.real_acc)} false alarms) but misses {pct(1 - b.fake_acc)} of AI images from unfamiliar generators.</p></div>
      </div>
      <div className="card flat"><h4>Score distribution</h4><ScoreHistogram real={b.histogram.real} ai={b.histogram.ai} /></div>
    </div>
  );
}

function Protocols({ p }: { p: NonNullable<Research["protocols"]> }) {
  const rows: [string, (s: ProtocolStats) => string][] = [["Accuracy", (s) => pct(s.accuracy)], ["ROC-AUC", (s) => num(s.roc_auc)],
    ["Average precision", (s) => num(s.average_precision)], ["Real kept real", (s) => pct(s.real_acc)], ["AI caught", (s) => pct(s.fake_acc)]];
  return (
    <div className="card flat stack">
      <h4>One centre crop vs the whole image</h4>
      <p className="small muted">The model was trained and first evaluated on one 224 px centre crop per image. The app instead scores every
        224 px window and averages them ({int(p.windows_total)} windows for the test set). The averaging rule was fixed before this test was run.</p>
      <div className="table-wrap">
        <table>
          <thead><tr><th></th><th className="num">Centre crop</th><th className="num">Whole-image scan (app)</th><th className="num">Change</th></tr></thead>
          <tbody>
            {rows.map(([k, f]) => {
              const key = { "Accuracy": "accuracy", "ROC-AUC": "roc_auc", "Average precision": "average_precision",
                "Real kept real": "real_acc", "AI caught": "fake_acc" }[k] as keyof ProtocolStats;
              const d = p.fullscan[key] - p.centre[key];
              return <tr key={k}><td>{k}</td><td className="num">{f(p.centre)}</td><td className="num"><b>{f(p.fullscan)}</b></td>
                <td className="num" style={{ color: d >= 0 ? "var(--sage)" : "var(--coral)" }}>{d >= 0 ? "+" : ""}{key === "roc_auc" || key === "average_precision" ? d.toFixed(3) : (d * 100).toFixed(1) + " pt"}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Generators({ r }: { r: Research }) {
  const [metric, setMetric] = useState<"accuracy" | "auc">("accuracy");
  const vs = r.versions.filter((v) => Object.keys(v.per_generator).length);
  const latest = vs[vs.length - 1];
  const scan = metric === "accuracy" ? r.protocols?.per_generator : undefined;
  return (
    <div className="stack" style={{ ["--gap" as string]: "24px" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <p className="muted small" style={{ maxWidth: 640 }}>Results for all 13 generators. ProGAN is the training generator (its test images are still unseen); the other 12 are zero-shot.</p>
        <div className="tabs"><button aria-selected={metric === "accuracy"} onClick={() => setMetric("accuracy")}>Accuracy</button><button aria-selected={metric === "auc"} onClick={() => setMetric("auc")}>ROC-AUC</button></div>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Generator</th>{vs.map((v) => <th key={v.id} className="num">{v.id}</th>)}{scan && <th className="num">fp-v4.1 whole image</th>}<th className="num">Change v4.0 → v4.1</th><th>AI caught (v4.1)</th></tr></thead>
          <tbody>
            {r.generators.map((g) => {
              const a = vs.find((v) => v.id === "fp-v4.0")?.per_generator[g]?.[metric];
              const c = latest.per_generator[g]?.[metric];
              const d = a != null && c != null ? c - a : null;
              return (
                <tr key={g}><td><b>{genName(g)}</b>{g === "progan" && <span className="chip" style={{ marginLeft: 8 }}>seen</span>}</td>
                  {vs.map((v) => <td key={v.id} className="num">{v.per_generator[g] ? (metric === "accuracy" ? pct(v.per_generator[g][metric]) : num(v.per_generator[g][metric])) : "–"}</td>)}
                  {scan && <td className="num"><b>{pct(scan[g]?.fullscan)}</b></td>}
                  <td className="num" style={{ color: d == null ? undefined : d >= 0 ? "var(--sage)" : "var(--coral)" }}>{d == null ? "–" : `${d >= 0 ? "+" : ""}${metric === "accuracy" ? (d * 100).toFixed(1) + " pt" : d.toFixed(3)}`}</td>
                  <td style={{ minWidth: 160 }}><div className="inline-bar"><span className="ib-track"><i style={{ width: `${(latest.per_generator[g]?.fake_acc ?? 0) * 100}%` }} /></span><span>{pct(latest.per_generator[g]?.fake_acc, 0)}</span></div></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted"><b>Version columns</b> = accuracy on that generator's test images (real and AI together, threshold 0.5), one centre crop per image.
        {r.protocols && <> <b>Whole image</b> = the same fp-v4.1 model scoring every 224 px window, as the app does.</>}
        <b> AI caught</b> = share of that generator's AI images flagged as AI (recall). False positive rate = 100% minus real photos kept real
        (overall {pct(1 - (latest.real_acc ?? 0))}).</p>
      <p className="small muted">fp-v3.0 was tested on a balanced 8,480-image sample of 12 generators (no ProGAN), so its column is indicative only. fp-v4.0 and fp-v4.1 use all 90,329 test images.</p>
    </div>
  );
}

function Versions({ r }: { r: Research }) {
  const max = 1;
  return (
    <div className="stack" style={{ ["--gap" as string]: "28px" }}>
      <div className="timeline">
        {r.versions.map((v, i) => (
          <Reveal key={v.id} delay={i * 0.08} className={`tl-item ${i === r.versions.length - 1 ? "current" : ""}`}>
            <span className="tl-dot" />
            <div className="tl-head"><h3>{v.id}</h3>{i === r.versions.length - 1 && <span className="chip teal">in this app</span>}</div>
            <p className="small">{v.notes}</p>
            <div className="tl-stats">
              <div><b>{pct(v.accuracy)}</b><span>accuracy</span></div>
              <div><b>{num(v.auc)}</b><span>AUC</span></div>
              <div><b>{num(v.ap)}</b><span>AP</span></div>
              <div><b>{int(v.test_images)}</b><span>test images</span></div>
            </div>
            {v.source && <p className="small muted">Source: {v.source}</p>}
          </Reveal>
        ))}
      </div>
      <div className="card flat">
        <h4 style={{ marginBottom: 14 }}>Zero-shot accuracy by version</h4>
        <BarList max={max} rows={r.versions.map((v, i) => ({ label: v.id, value: v.accuracy, color: i === r.versions.length - 1 ? "var(--teal)" : "var(--line-2)" }))} />
      </div>
    </div>
  );
}

const COLS: { k: keyof AblationRow; label: string; fmt: (v: number) => string }[] = [
  { k: "val_acc", label: "Val acc", fmt: (v) => pct(v) },
  { k: "test_acc", label: "Test acc", fmt: (v) => pct(v) },
  { k: "test_auc", label: "Test AUC", fmt: (v) => num(v) },
  { k: "test_ap", label: "Test AP", fmt: (v) => num(v) },
  { k: "real_acc", label: "Real acc", fmt: (v) => pct(v) },
  { k: "fake_acc", label: "Fake acc", fmt: (v) => pct(v) },
  { k: "mean_gen_acc", label: "Mean gen acc", fmt: (v) => pct(v) },
  { k: "worst_gen_acc", label: "Worst gen", fmt: (v) => pct(v) },
];
const LENS_KEYS = ["spatial", "frequency", "color", "physical", "semantic"] as const;
const LENS_SHORT = { spatial: "Tex", frequency: "Freq", color: "Col", physical: "Phy", semantic: "Sem" };

function Ablation({ r }: { r: Research }) {
  const [sort, setSort] = useState<keyof AblationRow>("test_acc");
  const [size, setSize] = useState(0);
  const full = r.ablation.find((x) => x.pillars === 5)!;
  const rows = useMemo(() => r.ablation.filter((x) => !size || x.pillars === size)
    .sort((a, b) => (b[sort] as number) - (a[sort] as number)), [r, sort, size]);
  const singles = r.ablation.filter((x) => x.pillars === 1).sort((a, b) => b.test_acc - a.test_acc);
  return (
    <div className="stack" style={{ ["--gap" as string]: "28px" }}>
      <p className="muted">Every one of the 31 possible lens combinations was trained from scratch with the same protocol (fp-v4.0 features) and tested on the full zero-shot set.</p>
      <div className="grid-2 align-start">
        <div className="card flat stack">
          <h4>Each lens on its own</h4>
          <BarList rows={singles.map((s) => ({ label: s.subset[0].toUpperCase() + s.subset.slice(1), value: s.test_acc,
            color: s.test_acc > 0.75 ? "var(--teal)" : s.test_acc > 0.65 ? "var(--saffron)" : "var(--coral)" }))} />
        </div>
        <div className="card flat stack">
          <h4>What the ablation tells us</h4>
          <ul className="ticks small">
            <li>The semantic lens is the strongest alone ({pct(singles[0].test_acc)}); removing it costs the most.</li>
            <li>Weak lenses still help: physical alone is {pct(r.ablation.find((x) => x.subset === "physical")!.test_acc)}, but it lifts several combinations.</li>
            <li>The full five-lens model has the best AUC ({num(full.test_auc)}) and AP ({num(full.test_ap)}) of all 31 runs.</li>
            <li>The best accuracy is within about 0.3 points of the full model, so the gate is not hurting.</li>
          </ul>
        </div>
      </div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div className="tabs">{[0, 1, 2, 3, 4, 5].map((n) => <button key={n} aria-selected={size === n} onClick={() => setSize(n)}>{n ? `${n} lens${n > 1 ? "es" : ""}` : "All"}</button>)}</div>
        <span className="small muted">Click a column to sort</span>
      </div>
      <div className="table-wrap" style={{ maxHeight: 640 }}>
        <table>
          <thead><tr><th>#</th>{LENS_KEYS.map((k) => <th key={k} title={k}>{LENS_SHORT[k]}</th>)}
            {COLS.map((c) => <th key={c.k} className={`num sortable`} onClick={() => setSort(c.k)}>{c.label}{sort === c.k ? " ↓" : ""}</th>)}<th className="num">Δ vs full</th></tr></thead>
          <tbody>
            {rows.map((x) => (
              <tr key={x.subset} className={x.pillars === 5 ? "highlight" : ""}>
                <td className="num">{x.pillars}</td>
                {LENS_KEYS.map((k) => <td key={k}>{x[k] ? <span className="on-dot" /> : <span className="off-dot" />}</td>)}
                {COLS.map((c) => <td key={c.k} className="num">{c.fmt(x[c.k] as number)}</td>)}
                <td className="num" style={{ color: x.test_acc >= full.test_acc ? "var(--sage)" : "var(--coral)" }}>{((x.test_acc - full.test_acc) * 100).toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Attribution({ r }: { r: Research }) {
  const a = r.attribution;
  if (!a) return <p className="muted">The source-attribution model has not been trained yet.</p>;
  const t = a.test, d = a.detector_zero_shot_genimage;
  const split = (s: "train" | "val" | "test") => Object.values(a.images_per_split[s]).reduce((x, y) => x + y, 0);
  return (
    <div className="stack" style={{ ["--gap" as string]: "28px" }}>
      <p className="muted" style={{ maxWidth: 820 }}>A separate model that names which generator most likely made an AI image. It reads the
        detector's five-lens features and is trained on {a.dataset}, never on the zero-shot benchmark, so the detector's results above are unchanged.
        Training: batch {a.batch_size}, max {a.max_epochs} epochs, early stopping (stopped after {a.epochs_run}, best epoch {a.best_epoch}).
        Images: {int(split("train"))} train, {int(split("val"))} validation, {int(t.images)} held-out test.</p>
      <div className="metric-strip">
        {[["Test accuracy", pct(t.accuracy)], ["Top-2 accuracy", pct(t.top2_accuracy)], ["Macro F1", num(t.macro_f1)],
          [`Accuracy when answered (${pct(t.answered_share, 0)} not unknown)`, pct(t.accuracy_when_answered)]]
          .map(([k, v]) => <div key={k}><b>{v}</b><span>{k}</span></div>)}
      </div>
      <div className="grid-2 align-start">
        <div className="card flat stack">
          <h4>Correctly named, per generator</h4>
          <BarList rows={a.classes.map((c) => ({ label: genName(c), value: t.per_class[c].recall,
            color: t.per_class[c].recall >= 0.85 ? "var(--plum)" : t.per_class[c].recall >= 0.6 ? "var(--saffron)" : "var(--coral)" }))} />
        </div>
        <div className="card flat stack">
          <h4>Our detector on these images (zero-shot)</h4>
          <p className="small muted">The real-vs-AI detector was never trained on Tiny-GenImage, so this is a fresh zero-shot test on mostly diffusion images ({int(d.images)} images).</p>
          <div className="mini-stats">
            <div><b>{pct(d.accuracy)}</b><span>accuracy</span></div><div><b>{num(d.auc)}</b><span>ROC-AUC</span></div>
            <div><b>{pct(d.real_acc)}</b><span>real kept real</span></div><div><b>{pct(d.fake_acc)}</b><span>AI caught</span></div>
          </div>
          <BarList rows={Object.entries(d.ai_caught).sort((x, y) => y[1] - x[1]).map(([g, v]) => ({ label: genName(g), value: v,
            color: v >= 0.7 ? "var(--teal)" : v >= 0.4 ? "var(--saffron)" : "var(--coral)" }))} />
        </div>
      </div>
      <div className="card flat stack">
        <h4>Confusion matrix (held-out test)</h4>
        <div className="table-wrap">
          <table>
            <thead><tr><th>True \ predicted</th>{a.classes.map((c) => <th key={c} className="num" title={genName(c)}>{c.replace(/^SD(\d)(\d)$/, "SD $1.$2")}</th>)}</tr></thead>
            <tbody>
              {t.confusion.map((row, i) => (
                <tr key={a.classes[i]}><td><b>{genName(a.classes[i])}</b></td>
                  {row.map((v, j) => <td key={j} className="num" style={i === j ? { color: "var(--plum)", fontWeight: 700 } : undefined}>{v}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small muted">Limits: it only knows these {a.classes.length} generators, and image size differs by generator in this dataset (e.g. BigGAN 128 px), which can act as a shortcut.</p>
      </div>
    </div>
  );
}

export default function Results() {
  const { r, err } = useResearch();
  const [m, setM] = useState<ModelInfo | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  useEffect(() => { api.model().then(setM).catch(() => setM(null)); }, []);
  if (err) return <div className="error-box">{err}</div>;
  return (
    <div className="stack" style={{ ["--gap" as string]: "32px" }}>
      <div className="tabs" role="tablist">{TABS.map(([k, l]) => <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{l}</button>)}</div>
      {!r || !m ? <div className="skeleton" style={{ height: 420 }} /> : (
        <>
          {tab === "overview" && <Overview m={m} r={r} />}
          {tab === "generators" && <Generators r={r} />}
          {tab === "versions" && <Versions r={r} />}
          {tab === "ablation" && <Ablation r={r} />}
          {tab === "source" && <Attribution r={r} />}
        </>
      )}
    </div>
  );
}
