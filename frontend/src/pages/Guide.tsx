import { useState } from "react";
import { Link } from "react-router-dom";
import { Gauge } from "../components/Charts";
import { Reveal } from "../components/Reveal";

const TERMS = [
  { t: "Verdict", d: "The short answer: AI-generated or real photo. It is AI-generated when the AI probability is 50% or more." },
  { t: "AI probability", d: "How strongly the detector leans towards 'AI-generated', from 0% to 100%. It is a model estimate, not a guarantee." },
  { t: "Confidence", d: "High when the probability is below 10% or above 90%; moderate between 10–30% or 70–90%; uncertain in the middle. Treat uncertain results as 'we don't know'." },
  { t: "Lenses", d: "Five independent ways of examining the image: texture, hidden frequency patterns, colour, light and noise, and overall look." },
  { t: "Influence", d: "How much a lens pushed the decision. Coral bars pushed towards AI, green bars towards real." },
  { t: "Grad-CAM", d: "A heatmap of where the texture lens paid attention. Warm colours mattered more. It shows attention, not the exact location of an edit." },
  { t: "Frequency spectrum", d: "The image broken into fine and coarse patterns. A grid of bright dots away from the centre is a classic sign of AI upsampling." },
  { t: "ROC-AUC", d: "A score from 0.5 (guessing) to 1.0 (perfect) for how well the detector separates real from AI images across all thresholds. Ours is 0.943 on unseen generators." },
  { t: "Zero-shot", d: "Tested on images from generators the detector never saw during training. This is the honest way to measure a detector." },
];

const FAQ = [
  { q: "Is the result proof?", a: "No. It is strong evidence for the generators we tested, but no detector is perfect. For anything important, combine it with other checks: the source of the image, reverse image search and content credentials." },
  { q: "Are my images stored?", a: "Single images are analysed and discarded. Batches (folders and ZIPs) are kept on the server for a few hours so you can open the report and download the CSV, then deleted. History lives only in your own browser." },
  { q: "Why does it say 'real' for an image I know is AI?", a: "The detector was trained on GAN images. Newer diffusion tools (Stable Diffusion, Midjourney, DALL-E) can leave different traces, and heavy compression or resizing removes evidence. It catches about 69% of AI images from unfamiliar generators while rarely flagging real photos." },
  { q: "Does it check the whole image?", a: "Yes. Generation traces live at the pixel level and resizing destroys them, so the detector never shrinks your image. It slides a 224 x 224 pixel window over every part of it at full resolution, scores each region, and averages the scores. The region map shows which parts looked AI-like. Large photos take a few seconds." },
  { q: "How do I test it on my own labelled images?", a: "Put your images in folders named real and fake (or 0_real and 1_fake), then upload the parent folder or a ZIP of it. The batch report will show accuracy, AUC, a ROC curve and a confusion matrix for your data." },
];

export default function Guide() {
  const [p, setP] = useState(0.82);
  const band = Math.abs(p - 0.5) >= 0.4 ? "High confidence" : Math.abs(p - 0.5) >= 0.2 ? "Moderate confidence" : "Uncertain";
  return (
    <>
      <section className="wrap page-head">
        <span className="eyebrow">Guide</span>
        <h1>Reading your results</h1>
        <p className="lede">Everything on a result page, explained without jargon. Start with the verdict; go deeper only if you want to.</p>
      </section>
      <section className="wrap stack" style={{ ["--gap" as string]: "72px" }}>
        <Reveal className="card guide-gauge">
          <div>
            <h3>Play with the gauge</h3>
            <p className="muted small" style={{ margin: "8px 0 18px" }}>Move the slider to see how probability turns into a verdict and a confidence level.</p>
            <input type="range" min={0} max={100} value={Math.round(p * 100)} onChange={(e) => setP(+e.target.value / 100)} aria-label="AI probability" style={{ width: "100%" }} />
            <div className="row" style={{ marginTop: 16 }}>
              <span className={`chip ${p >= 0.5 ? "ai" : "real"}`}><span className="dot" />{p >= 0.5 ? "AI-generated" : "Real photo"}</span>
              <span className={`chip ${band === "Uncertain" ? "low" : "teal"}`}>{band}</span>
            </div>
          </div>
          <Gauge key={Math.round(p * 20)} value={p} size={260} />
        </Reveal>

        <div className="stack" style={{ ["--gap" as string]: "20px" }}>
          <h2>Terms on the result page</h2>
          <div className="terms">
            {TERMS.map((t, i) => (
              <Reveal key={t.t} delay={(i % 3) * 0.06} className="term"><h4>{t.t}</h4><p className="small muted">{t.d}</p></Reveal>
            ))}
          </div>
        </div>

        <div className="stack" style={{ ["--gap" as string]: "14px" }}>
          <h2>Questions</h2>
          {FAQ.map((f) => (
            <details key={f.q} className="faq"><summary>{f.q}</summary><p className="muted">{f.a}</p></details>
          ))}
        </div>

        <div className="cta-band">
          <div><h2>Ready to try it?</h2><p className="lede">Upload an image and open the advanced analysis to see all of this in action.</p></div>
          <Link to="/detect" className="btn light">Check an image →</Link>
        </div>
      </section>
      <div className="spacer" /><div className="spacer" />
    </>
  );
}
