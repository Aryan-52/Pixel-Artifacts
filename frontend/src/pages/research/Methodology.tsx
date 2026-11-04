import { LensArt, LENS_COPY } from "../../components/LensArt";
import { LineChart } from "../../components/Charts";
import { Reveal } from "../../components/Reveal";
import { int } from "../../lib/format";
import { useResearch } from "./ResearchLayout";

const DIMS: Record<string, string> = { spatial: "1,952", frequency: "408", color: "133", physical: "89", semantic: "512" };
const HOW: Record<string, string> = {
  spatial: "Mean and standard deviation of EfficientNet-B0 stages 2, 3, 4 and 6 plus the 1,280-d pooled vector. In fp-v4.1 the network is first fine-tuned as a real/fake classifier.",
  frequency: "Hann-windowed 2-D FFT of the grey image and of its high-pass residual, averaged over 64 radial bands, plus spectral statistics.",
  color: "Channel means, spreads, skew, kurtosis and cross-channel correlations in RGB, YCbCr and opponent colour space.",
  physical: "Gradient, shading and local-noise consistency statistics from Sobel and high-pass filters. Image-derived cues, not camera PRNU.",
  semantic: "OpenCLIP ViT-B/32 (OpenAI weights) image embedding, L2-normalised. Frozen.",
};

function Pipeline() {
  return (
    <div className="pipeline">
      <div className="pipe-node"><b>Image</b><span>any size</span></div>
      <div className="pipe-arrow" />
      <div className="pipe-node"><b>224 px windows</b><span>cover the whole image, no resizing</span></div>
      <div className="pipe-arrow" />
      <div className="pipe-fan">
        {LENS_COPY.map((l) => <div key={l.key} className="pipe-lens"><b>{l.name}</b><span>{DIMS[l.key]} features</span></div>)}
      </div>
      <div className="pipe-arrow" />
      <div className="pipe-node accent"><b>Projection + gate</b><span>256-d per lens, softmax weights</span></div>
      <div className="pipe-arrow" />
      <div className="pipe-node dark-node"><b>Verdict</b><span>mean p(AI) over windows, threshold 0.5</span></div>
    </div>
  );
}

export default function Methodology() {
  const { r } = useResearch();
  const t = r?.training;
  return (
    <div className="stack" style={{ ["--gap" as string]: "72px" }}>
      <Reveal className="stack" style={{ ["--gap" as string]: "20px" }}>
        <h2>The pipeline</h2>
        <p className="lede">Each image goes through the same steps used in training, so the app's scores match the evaluated model exactly (checked by an automated parity test).</p>
        <Pipeline />
      </Reveal>

      <div className="stack" style={{ ["--gap" as string]: "20px" }}>
        <h2>The five lenses in detail</h2>
        <div className="lens-table">
          {LENS_COPY.map((l, i) => (
            <Reveal key={l.key} delay={i * 0.05} className="lens-row">
              <LensArt k={l.key} />
              <div><h4>{l.name}</h4><span className="chip">{l.tech}</span></div>
              <p className="small">{HOW[l.key]}</p>
              <div className="lens-dim"><b>{DIMS[l.key]}</b><span>features</span></div>
            </Reveal>
          ))}
        </div>
        <p className="small muted">Each lens is projected to 256 dimensions. A gate looks at all five projections and gives each a weight per image (softmax).
          During training a whole lens is randomly dropped 15% of the time, so the model never depends on one lens alone.</p>
      </div>

      <div className="grid-2 align-start">
        <Reveal className="card flat stack">
          <h3>Data</h3>
          <dl className="kv">
            <dt>Training generator</dt><dd>ProGAN, 20 LSUN categories</dd>
            <dt>Training images</dt><dd>{t ? int(Number(t.data.train_images)) : "720,030"} (+ one blur/JPEG copy each)</dd>
            <dt>Validation</dt><dd>{t ? int(Number(t.data.val_images)) : "7,569"} ProGAN images (model selection only)</dd>
            <dt>Test</dt><dd>{t ? int(Number(t.data.test_images)) : "90,329"} images, 13 generators (CNN_synth_testset)</dd>
            <dt>Leakage check</dt><dd>89 train and 431 val images also present in other splits removed</dd>
          </dl>
        </Reveal>
        <Reveal delay={0.08} className="card flat stack">
          <h3>Rules we kept</h3>
          <ul className="ticks">
            <li>The test set was never used for training, model selection, threshold or calibration.</li>
            <li>Threshold fixed at 0.5 and temperature at 1.0 before testing.</li>
            <li>Model chosen only by clean validation loss, with early stopping.</li>
            <li>Batch size 64 and at most 50 epochs, with early stopping.</li>
            <li>Feature statistics computed on the training split only.</li>
          </ul>
        </Reveal>
      </div>

      {t && (
        <div className="stack" style={{ ["--gap" as string]: "24px" }}>
          <h2>Training, in two stages</h2>
          <div className="grid-2 align-start">
            {t.stages.map((s, i) => (
              <Reveal key={i} delay={i * 0.08} className="card flat stack">
                <h3>{String(s.name)}</h3>
                <dl className="kv">
                  <dt>Batch size</dt><dd>{s.batch_size}</dd>
                  <dt>Max epochs</dt><dd>{s.max_epochs} (stopped after {s.epochs_run}, best epoch {s.best_epoch})</dd>
                  <dt>Epoch</dt><dd>{s.epoch_size}</dd>
                  <dt>Optimiser</dt><dd>{s.optimizer}</dd>
                  <dt>Early stopping</dt><dd>{s.early_stopping}</dd>
                  <dt>Regularisation</dt><dd>{s.augmentation}</dd>
                </dl>
              </Reveal>
            ))}
          </div>
          <div className="grid-2 align-start">
            <div className="card flat">
              <h4>Stage 1: loss per epoch</h4>
              <LineChart xLabel="epoch" yLabel="BCE loss" series={[
                { name: "train loss", color: "#0f5c68", points: t.spatial_history.map((h) => [h.epoch, h.train_loss]) },
                { name: "validation loss", color: "#d4512a", points: t.spatial_history.map((h) => [h.epoch, h.val_loss]), dashed: true },
              ]} />
            </div>
            <div className="card flat">
              <h4>Stage 2: validation loss and gate weights</h4>
              <LineChart xLabel="epoch" yLabel="value" series={[
                { name: "val loss x 10", color: "#d4512a", points: t.head_history.map((h) => [h.epoch, h.val_loss_clean * 10]), dashed: true },
                { name: "semantic weight", color: "#6b4e71", points: t.head_history.map((h) => [h.epoch, h.gate_semantic]) },
                { name: "spatial weight", color: "#0f5c68", points: t.head_history.map((h) => [h.epoch, h.gate_spatial]) },
              ]} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
