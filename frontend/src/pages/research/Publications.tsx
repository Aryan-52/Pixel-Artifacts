import { Reveal } from "../../components/Reveal";

const DOCS = [
  { kind: "Blackbook", title: "Project report (SPIT MCA template)", body: "Introduction, literature survey, analysis, design and methodology, results, conclusion and future work.", status: "In preparation" },
  { kind: "Research paper", title: "Revised research paper", body: "Five-lens fusion, full per-generator tables, individual and combined ablation, comparison with published detectors.", status: "In revision" },
];

const GAPS = [
  { t: "Generalisation to unseen generators", d: "Detectors trained on one GAN often fail on others. Most papers report only a few test generators." },
  { t: "Single-cue detectors", d: "Most detectors rely on one kind of evidence (CNN texture, spectrum or CLIP). When that cue fails, the detector fails." },
  { t: "Opaque outputs", d: "Few systems tell a non-expert why an image was flagged or how much to trust the score." },
  { t: "Leaky evaluation", d: "Thresholds or calibration are often tuned on the test generators, which inflates zero-shot results." },
];

const SHORTCOMINGS = [
  { t: "Misses many AI images from unfamiliar generators", d: "Real photos are kept real 97% of the time, but only 69% of AI images are caught. SAN (11%), Deepfake (16%) and WhichFaceIsReal (37%) are the hardest." },
  { t: "Weak on diffusion images", d: "The detector was trained on GAN images only. On Tiny-GenImage (zero-shot, mostly diffusion) it catches only 23% of AI images (Midjourney 7%), while keeping 97% of real photos real." },
  { t: "Trained on 224 x 224 crops", d: "The model was trained on 224 px centre crops. The app averages 224 px windows over the whole image; a plain average can dilute a small AI-edited region." },
  { t: "Training features of the fine-tuned lens", d: "In fp-v4.1 the texture lens is fine-tuned on the same training images the gate later sees, which can make the gate trust it slightly too much." },
];

const FUTURE = [
  { t: "Larger CLIP model (ViT-L/14)", d: "Swap the semantic lens for a larger CLIP model, which published work shows transfers better to unseen generators.", gain: "+2 to 4 pt accuracy", effort: "~6 h compute" },
  { t: "Wider training augmentation", d: "Add resizing, noise and stronger JPEG to the training copies so fewer AI images slip through.", gain: "+1 to 3 pt", effort: "~5 h compute" },
  { t: "Diffusion-model coverage", d: "Add diffusion images (Stable Diffusion, Midjourney and others) to training and test on held-out diffusion generators.", gain: "Needed for real-world use", effort: "~1 to 2 weeks" },
  { t: "Learned region weighting", d: "Train how to combine window scores (instead of a plain average) so small edited regions are not diluted.", gain: "Better on partly edited images", effort: "~1 week" },
  { t: "Content credentials (C2PA)", d: "Verify signed content credentials in addition to the forensic analysis.", gain: "Certain answer when present", effort: "~1 week" },
];

export default function Publications() {
  return (
    <div className="stack" style={{ ["--gap" as string]: "80px" }}>
      <div className="stack" style={{ ["--gap" as string]: "20px" }}>
        <h2>Documents</h2>
        <div className="grid-2">
          {DOCS.map((d, i) => (
            <Reveal key={d.kind} delay={i * 0.08} className="doc-card">
              <div className="doc-cover"><span>{d.kind}</span><i /><i /><i /></div>
              <div className="stack" style={{ ["--gap" as string]: "10px" }}>
                <span className="chip low">{d.status}</span>
                <h3>{d.title}</h3>
                <p className="small muted">{d.body}</p>
                <button className="btn ghost sm" disabled>PDF available soon</button>
              </div>
            </Reveal>
          ))}
        </div>
      </div>

      <div className="grid-2 align-start">
        <div className="stack" style={{ ["--gap" as string]: "18px" }}>
          <h2>Research gaps we address</h2>
          {GAPS.map((g, i) => (
            <Reveal key={g.t} delay={i * 0.05} className="gap-item"><span className="gap-n">{i + 1}</span><div><h4>{g.t}</h4><p className="small muted">{g.d}</p></div></Reveal>
          ))}
        </div>
        <div className="stack" style={{ ["--gap" as string]: "18px" }}>
          <h2>Honest shortcomings</h2>
          {SHORTCOMINGS.map((g, i) => (
            <Reveal key={g.t} delay={i * 0.05} className="gap-item warn"><span className="gap-n">!</span><div><h4>{g.t}</h4><p className="small muted">{g.d}</p></div></Reveal>
          ))}
        </div>
      </div>

      <div className="stack" style={{ ["--gap" as string]: "20px" }}>
        <h2>Future scope</h2>
        <p className="muted">Expected gains are estimates from published results on similar setups, not measurements on this model.</p>
        <div className="roadmap">
          {FUTURE.map((f, i) => (
            <Reveal key={f.t} delay={i * 0.06} className="road-item">
              <span className="road-n">{String(i + 1).padStart(2, "0")}</span>
              <h4>{f.t}</h4>
              <p className="small muted">{f.d}</p>
              <div className="row"><span className="chip teal">{f.gain}</span><span className="chip">{f.effort}</span></div>
            </Reveal>
          ))}
        </div>
      </div>
    </div>
  );
}
