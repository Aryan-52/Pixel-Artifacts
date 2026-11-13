// Reported results of Sahil Sawant's four-branch gated hybrid (v3).
// Transcribed from his results deck ("Beyond Pixel Artifacts", 18 slides). These are his
// evaluation numbers; the app's live detector (fp-v4.1) does not run this model.

export const headline = {
  images: 90245,
  generators: 13,
  accuracy: 0.915,
  auc: 0.965,
  ap: 0.972,
  realKept: 0.956,
  aiCaught: 0.874,
  meanGenAcc: 0.896,
  foldB: { auc: 0.995, accuracy: 0.95, ece: 0.053, generators: "CRN, IMLE, SeeingDark, WhichFaceIsReal", images: 27880 },
};

export const branches = [
  { name: "Deep", looks: "EfficientNet-B0 pooled features, all layers fine-tuned end to end", why: "Learns up-sampling traces in local texture", dim: "1,280" },
  { name: "Physics", looks: "22 scene statistics: lighting, shadows, spectrum slope, noise", why: "Photos are consistent in light and sensor; fakes may not be", dim: "22" },
  { name: "Frequency", looks: "Frequency-domain statistics", why: "Up-convolution leaves spectral peaks", dim: "d_f" },
  { name: "Semantic", looks: "Frozen CLIP ViT-L/14 embedding", why: "Never trained on fakes, so ProGAN artefacts do not shape it", dim: "768" },
];

export const training = [
  ["Training data", "200,000 images: fixed random subset (seed 42) of 721,869. Fakes: ProGAN, 20 LSUN classes. Real: LSUN + FFHQ."],
  ["Validation", "8,000 ProGAN rows (cached-feature runs)"],
  ["Test", "90,245 images, 13 generators, 42 source folders"],
  ["Input", "224 × 224 × 3 crop, no resampling"],
  ["Augmentation", "Blur p 0.5 (σ up to 3), JPEG p 0.8 (quality ≥ 30), down-scaling p 0.3 (factor 0.5–0.9)"],
  ["Hybrid training", "End to end on raw images: 4 epochs, batch 32, lr 1e-4, seed 42"],
  ["Fusion", "Concatenate [deep; physics; frequency; semantic], a gate scales each branch per image, branch dropout 0.15, MLP to 2 logits"],
  ["Hardware", "One Tesla T4 (Kaggle)"],
] as const;

// generator, accuracy, recall, specificity, AP (all %)
export const perGenerator: [string, number, number, number, number][] = [
  ["ProGAN (seen)", 100.0, 100.0, 100.0, 100.0],
  ["StyleGAN2", 83.9, 67.8, 100.0, 99.9],
  ["StyleGAN", 83.8, 67.6, 100.0, 97.6],
  ["BigGAN", 89.9, 86.1, 93.7, 97.3],
  ["CycleGAN", 95.4, 94.8, 96.0, 99.2],
  ["StarGAN", 99.8, 99.6, 99.9, 100.0],
  ["GauGAN", 94.0, 98.0, 90.1, 98.7],
  ["CRN", 95.4, 99.9, 90.9, 99.8],
  ["IMLE", 95.3, 99.7, 90.9, 99.7],
  ["SeeingDark", 81.8, 76.0, 87.7, 91.2],
  ["SAN", 61.9, 25.2, 98.6, 82.8],
  ["DeepFake", 90.7, 84.5, 96.9, 96.6],
  ["WhichFaceIsReal", 92.8, 91.1, 94.6, 98.3],
];
export const perGeneratorMean: [number, number, number, number] = [89.6, 83.9, 95.3, 97.0];

// generator, hybrid, UnivFD, DIRE-recipe, Wang et al. (accuracy %)
export const comparison: [string, number, number, number, number][] = [
  ["ProGAN (seen)", 100.0, 99.69, 99.33, 100.0],
  ["StyleGAN", 83.78, 83.55, 61.12, 90.2],
  ["BigGAN", 89.94, 95.55, 55.98, 71.2],
  ["CycleGAN", 95.44, 97.01, 78.9, 87.6],
  ["StarGAN", 99.8, 95.17, 78.4, 94.6],
  ["GauGAN", 94.05, 99.11, 69.11, 81.4],
  ["CRN", 95.41, 55.85, 73.54, 86.3],
  ["IMLE", 95.28, 68.6, 85.58, 86.3],
  ["SeeingDark", 81.84, 58.33, 54.75, 98.1],
  ["SAN", 61.93, 55.94, 50.69, 50.0],
  ["DeepFake", 90.69, 62.46, 56.02, 50.7],
  ["StyleGAN2", 83.89, 75.58, 57.27, 86.9],
  ["WhichFaceIsReal", 92.84, 82.55, 50.4, 91.6],
];
export const comparisonMean: [number, number, number, number] = [89.61, 79.18, 67.01, 82.68];
export const comparisonMethods = ["Gated hybrid", "UnivFD", "DIRE-recipe", "Wang et al."] as const;

// combination, accuracy, specificity, recall, AUC, AP, mean per-generator accuracy
export const ablation: [string, number, number, number, number, number, number][] = [
  ["Full hybrid (gated, joint)", 91.5, 95.6, 87.4, 0.965, 0.972, 89.6],
  ["CLIP", 77.7, 99.7, 55.7, 0.921, 0.935, 77.0],
  ["Deep", 83.2, 80.6, 85.8, 0.927, 0.932, 82.6],
  ["Physics", 60.2, 48.2, 72.1, 0.626, 0.584, 57.3],
  ["Frequency", 58.9, 53.8, 64.0, 0.601, 0.563, 55.2],
  ["CLIP + Deep", 85.4, 99.7, 71.2, 0.965, 0.971, 84.7],
  ["CLIP + Physics", 77.9, 99.7, 56.1, 0.864, 0.89, 77.2],
  ["CLIP + Frequency", 77.7, 99.7, 55.7, 0.836, 0.874, 77.0],
  ["Physics + Deep", 82.7, 79.6, 85.7, 0.859, 0.802, 82.2],
  ["Frequency + Deep", 82.5, 79.6, 85.3, 0.846, 0.804, 82.0],
  ["Physics + Frequency", 61.8, 54.0, 69.6, 0.643, 0.592, 58.4],
  ["CLIP + Physics + Deep", 85.6, 89.1, 82.2, 0.936, 0.939, 84.1],
  ["CLIP + Frequency + Deep", 84.9, 89.0, 80.9, 0.921, 0.933, 82.9],
  ["CLIP + Physics + Frequency", 75.8, 95.8, 55.8, 0.859, 0.886, 74.7],
  ["Physics + Frequency + Deep", 80.7, 77.4, 83.9, 0.841, 0.782, 79.9],
  ["CLIP + Physics + Frequency + Deep", 84.7, 87.1, 82.3, 0.921, 0.928, 83.3],
];

export const caveats = [
  "Single seed.",
  "Fold A was used for epoch selection and calibration, so only fold B (CRN, IMLE, SeeingDark, WhichFaceIsReal) is fully held out.",
  "Single branches and their combinations are equal-weight means of separately trained branches; only the full hybrid is gated and trained jointly.",
  "Gate weights were not recorded, so which branch the gate favours is not known.",
  "Wang et al. column as supplied in the source results.",
  "Only GAN-era generators were tested; no diffusion models.",
];

// Four branches in the shape the landing and methodology pages use (key = LensArt illustration).
export const lenses = [
  { key: "spatial", name: "Texture & detail", tech: "EfficientNet-B0 (fine-tuned)", dims: "1,280",
    plain: "Fine textures, edges and tiny glitches that generators get subtly wrong.",
    how: "Pooled EfficientNet-B0 features; all layers are fine-tuned end to end with the rest of the model." },
  { key: "physical", name: "Light & noise", tech: "22 physics cues", dims: "22",
    plain: "Whether lighting, shadows and sensor noise behave consistently across the image.",
    how: "22 scene statistics: illumination, shadows, spectrum slope, noise and edges. Image-derived cues, not camera PRNU." },
  { key: "frequency", name: "Hidden patterns", tech: "Frequency statistics", dims: "d_f",
    plain: "Repeating pixel patterns left by the upsampling layers inside generators.",
    how: "Frequency-domain statistics of the image; up-convolution leaves spectral peaks." },
  { key: "semantic", name: "Overall look", tech: "CLIP ViT-L/14", dims: "768",
    plain: "The image as a whole, the way a person would take it in at a glance.",
    how: "Frozen CLIP ViT-L/14 image embedding. Never trained on fakes, so ProGAN artefacts do not shape it." },
];
