export const pct = (x: number | null | undefined, digits = 1) =>
  x == null || Number.isNaN(x) ? "–" : `${(x * 100).toFixed(digits)}%`;

export const num = (x: number | null | undefined, digits = 3) =>
  x == null || Number.isNaN(x) ? "–" : x.toFixed(digits);

export const int = (x: number) => x.toLocaleString("en-IN");

export const bytes = (n: number) =>
  n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;

export const duration = (s: number | null | undefined) => {
  if (s == null) return "–";
  if (s < 60) return `${Math.max(1, Math.round(s))} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${Math.round(s - m * 60)} s`;
};

export const GEN_NAMES: Record<string, string> = {
  progan: "ProGAN", biggan: "BigGAN", crn: "CRN", cyclegan: "CycleGAN", deepfake: "Deepfake", gaugan: "GauGAN",
  imle: "IMLE", san: "SAN", seeingdark: "SeeingDark", stargan: "StarGAN", stylegan: "StyleGAN",
  stylegan2: "StyleGAN2", whichfaceisreal: "WhichFaceIsReal",
  // source-attribution classes (Tiny-GenImage + ProGAN)
  ProGAN: "ProGAN", BigGAN: "BigGAN", ADM: "ADM (diffusion)", GLIDE: "GLIDE (diffusion)", Midjourney: "Midjourney",
  SD14: "Stable Diffusion 1.4", SD15: "Stable Diffusion 1.5", VQDM: "VQDM (diffusion)", Wukong: "Wukong (diffusion)",
};

export const genName = (g: string) => GEN_NAMES[g] ?? g;

export const verdictWord = (v: "ai" | "real") => (v === "ai" ? "AI-generated" : "Real photo");

export const bandWord = { high: "High confidence", moderate: "Moderate confidence", low: "Uncertain" } as const;

export const IMAGE_EXT = /\.(jpe?g|png|webp|bmp|tiff?)$/i;

export function download(name: string, text: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
