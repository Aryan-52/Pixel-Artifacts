import { BarList } from "../../components/Charts";
import { Pipeline } from "../../components/Pipeline";
import { int } from "../../lib/format";
import {
  ablation, branches, caveats, comparison, comparisonMean, comparisonMethods, headline as h,
  perGenerator, perGeneratorMean, training,
} from "../../data/sahilHybrid";

const p = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;

export default function Hybrid() {
  const best = (row: number[]) => Math.max(...row);
  return (
    <div className="stack" style={{ ["--gap" as string]: "32px" }}>
      <p className="note">Evaluation results of the gated hybrid model. The live detector in this app runs fp-v4.1.</p>

      <div className="metric-strip">
        {[["Accuracy", p(h.accuracy)], ["ROC-AUC", h.auc.toFixed(3)], ["Average precision", h.ap.toFixed(3)], ["Real kept real", p(h.realKept)],
          ["AI caught (recall)", p(h.aiCaught)], [`Mean accuracy, ${h.generators} generators`, p(h.meanGenAcc)],
          ["Fold B ROC-AUC", h.foldB.auc.toFixed(3)], ["Fold B accuracy", p(h.foldB.accuracy)]]
          .map(([k, v]) => <div key={k}><b>{v}</b><span>{k}</span></div>)}
      </div>
      <p className="small muted">Full test split: {int(h.images)} images, {h.generators} generators, threshold 0.5. Fold B ({h.foldB.generators};
        {" "}{int(h.foldB.images)} images) is untouched by every selection rule; calibration error there is {h.foldB.ece}.</p>

      <Pipeline
        input={{ title: "Image", note: "any size" }}
        window={{ title: "224 × 224 crop", note: "no resampling" }}
        branches={branches.map((b) => ({ name: b.name, dims: b.dim === "d_f" ? "frequency stats" : `${b.dim} features` }))}
        fusion={{ title: "Gated fusion", note: "per-image gate, dropout 0.15" }}
        verdict={{ title: "Verdict", note: "MLP → softmax, AI if p ≥ 0.5" }} />

      <div className="card flat stack">
        <h4>Four branches, one gate</h4>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Branch</th><th>What it looks at</th><th>Why a fake may be caught</th><th className="num">Dim.</th></tr></thead>
            <tbody>{branches.map((b) => <tr key={b.name}><td><b>{b.name}</b></td><td>{b.looks}</td><td>{b.why}</td><td className="num">{b.dim}</td></tr>)}</tbody>
          </table>
        </div>
        <p className="small muted">x = [d; p; f; s] → x′ = [γ₁d; γ₂p; γ₃f; γ₄s] → MLP → softmax. The gate computes γ for each image.</p>
      </div>

      <div className="card flat stack">
        <h4>Training setup</h4>
        <div className="table-wrap">
          <table><tbody>{training.map(([k, v]) => <tr key={k}><td><b>{k}</b></td><td>{v}</td></tr>)}</tbody></table>
        </div>
      </div>

      <div className="card flat stack">
        <h4>Results per generator (threshold 0.5)</h4>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Generator</th><th className="num">Accuracy %</th><th className="num">Recall %</th><th className="num">Specificity %</th><th className="num">AP %</th></tr></thead>
            <tbody>
              {perGenerator.map(([g, ...v]) => <tr key={g}><td><b>{g}</b></td>{v.map((x, i) => <td key={i} className="num">{x.toFixed(1)}</td>)}</tr>)}
              <tr><td><b>Mean (13)</b></td>{perGeneratorMean.map((x, i) => <td key={i} className="num"><b>{x.toFixed(1)}</b></td>)}</tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid-2 align-start">
        <div className="card flat stack">
          <h4>Mean accuracy over 13 generators</h4>
          <BarList max={100} format={(v) => `${v.toFixed(1)}%`} rows={comparisonMethods.map((m, i) => ({
            label: m, value: comparisonMean[i], color: i === 0 ? "var(--teal)" : "var(--line)" }))} />
          <p className="small muted">All methods retrained on the same ProGAN data and scored on the same test images; backbone, augmentation and checkpoint rule differ.</p>
        </div>
        <div className="card flat stack">
          <h4>Where each method wins</h4>
          <p className="small"><b>Hybrid leads</b> on CRN (95.4 vs 55.9 for UnivFD), IMLE (95.3 vs 68.6), DeepFake (90.7 vs 62.5) and StarGAN (99.8).</p>
          <p className="small"><b>UnivFD leads</b> on BigGAN, CycleGAN and GauGAN, where CLIP content cues are enough.</p>
          <p className="small"><b>Hard for everyone:</b> SAN (hybrid 61.9%, others 50–56%).</p>
        </div>
      </div>

      <div className="card flat stack">
        <h4>Comparison per generator (accuracy %)</h4>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Generator</th>{comparisonMethods.map((m) => <th key={m} className="num">{m}</th>)}</tr></thead>
            <tbody>
              {comparison.map(([g, ...v]) => <tr key={g}><td><b>{g}</b></td>
                {v.map((x, i) => <td key={i} className="num" style={x === best(v) ? { fontWeight: 700, color: "var(--teal-ink)" } : undefined}>{x.toFixed(2)}</td>)}</tr>)}
              <tr><td><b>Mean</b></td>{comparisonMean.map((x, i) => <td key={i} className="num"><b>{x.toFixed(2)}</b></td>)}</tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="card flat stack">
        <h4>Ablation: all 15 branch combinations</h4>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Combination</th><th className="num">Acc. %</th><th className="num">Spec. %</th><th className="num">Recall %</th><th className="num">AUC</th><th className="num">AP</th><th className="num">Mean-gen acc. %</th></tr></thead>
            <tbody>
              {ablation.map(([c, acc, sp, rc, auc, ap, mg], i) => <tr key={c}><td>{i === 0 ? <b>{c}</b> : c}</td>
                <td className="num">{acc.toFixed(1)}</td><td className="num">{sp.toFixed(1)}</td><td className="num">{rc.toFixed(1)}</td>
                <td className="num">{auc.toFixed(3)}</td><td className="num">{ap.toFixed(3)}</td><td className="num">{mg.toFixed(1)}</td></tr>)}
            </tbody>
          </table>
        </div>
        <p className="small muted">Leave one out from the four-branch mean (84.7%): without CLIP 80.7, without deep 75.8, without physics 84.9, without frequency 85.6.</p>
      </div>

      <div className="note warn stack" style={{ ["--gap" as string]: "6px" }}>
        <b>Evaluation notes</b>
        <ul style={{ margin: 0, paddingLeft: 18 }}>{caveats.map((c) => <li key={c}>{c}</li>)}</ul>
      </div>
    </div>
  );
}
