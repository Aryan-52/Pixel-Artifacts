// Typed client for the FastAPI backend (backend/app/main.py).

export type Verdict = "ai" | "real";
export type Band = "high" | "moderate" | "low";

export interface Lens {
  key: string;
  name: string;
  tech: string;
  plain: string;
  weight: number;
  delta_logit: number;
}

export interface Roc { fpr: number[]; tpr: number[]; auc: number }

/** Source attribution: which generator the image most resembles (null top = unknown source). */
export interface Source {
  top: string | null; unknown_below: number; test_accuracy: number; ranked: { generator: string; p: number }[];
}

export interface AttributionEval {
  images: number; accuracy: number; top2_accuracy: number; macro_f1: number; unknown_below: number;
  answered_share: number; accuracy_when_answered: number; confusion: number[][];
  per_class: Record<string, { images: number; recall: number; precision: number }>;
}

export interface Analysis {
  filename: string;
  error?: string;
  sha256: string;
  bytes: number;
  verdict: Verdict;
  p_ai: number;
  p_real: number;
  confidence: number;
  band: Band;
  logit: number;
  threshold: number;
  summary: { headline: string; lead: string; why: string; caution: string };
  lenses: Lens[];
  /** focus_box = the 224 px window that looked most AI-generated (null: small image read whole). */
  image: { width: number; height: number; upscaled: boolean; windows: number; focus_box: number[] | null };
  /** Whole-image scan: every 224 px window's score; the verdict is their mean. */
  scan: { windows: { x: number; y: number; p_ai: number }[]; size: number; max_p_ai: number; share_ai: number };
  metadata: {
    format: string | null; mode: string; exif: Record<string, string>;
    text_chunks: Record<string, string>; ai_markers: string[]; camera_info: boolean; note: string;
  };
  evidence: { bin: [number, number]; bin_images: number; bin_share_ai: number | null;
    real_scored_lower: number; ai_scored_lower: number };
  visuals?: Record<"original" | "crop" | "gradcam" | "spectrum" | "noise" | "ela", string>;
  spatial_expert_p_ai?: number;
  histograms?: Record<"r" | "g" | "b", number[]>;
  source: Source | null;
  model_version: string;
  inference_ms: number;
}

export interface GenStats { accuracy: number; auc: number; n: number; real_acc: number; fake_acc: number }

export interface Benchmark {
  dataset: string; images: number; generators: number; zero_shot: boolean; protocol: string;
  accuracy: number; auc: number; ap: number; precision: number; recall: number; f1: number;
  real_acc: number; fake_acc: number; confusion: number[][]; roc: Roc;
  validation: { accuracy: number; auc: number; images: number };
  per_generator: Record<string, GenStats>;
  histogram: { edges: number[]; real: number[]; ai: number[] };
  operating_point: { threshold: number; fpr: number; tpr: number };
}

export interface ModelInfo {
  model_version: string; device: string; pillars: string[]; threshold: number; temperature: number;
  image_size: number; crop_policy: string; parameters_head: number; benchmark: Benchmark;
}

export interface BinaryMetrics {
  n: number; accuracy: number; precision: number; recall: number; specificity: number; f1: number;
  confusion: { tn: number; fp: number; fn: number; tp: number }; roc?: Roc; auc?: number;
}

export interface JobItem {
  index: number; path: string; filename: string; bytes: number; verdict?: Verdict; p_ai?: number;
  band?: Band; width?: number; height?: number; format?: string; ground_truth: 0 | 1 | null;
  correct?: boolean; error?: string; ai_markers?: string[]; gates?: Record<string, number>;
  windows?: number; share_ai_windows?: number;
}

export interface JobSummary {
  images: number; scored: number; errors: number; skipped: number; ai: number; real: number;
  ai_share: number; mean_p_ai: number | null; bands: Record<Band, number>; histogram: number[];
  with_ai_markers: number; folders: { folder: string; images: number; ai: number; mean_p_ai: number }[];
  mean_gates: Record<string, number>;
  labelled?: BinaryMetrics & { images: number; real: number; ai: number };
}

export interface Job {
  id: string; source: "zip" | "folder"; name: string; status: "queued" | "running" | "done" | "failed";
  total: number; done: number; created: string; seconds?: number; eta_seconds: number | null;
  skipped: { path: string; reason: string }[]; summary?: JobSummary; items: JobItem[];
}

export interface VersionRow {
  id: string; notes: string; test_images: number; generators: number; accuracy: number; auc: number;
  ap: number; f1?: number; real_acc?: number; fake_acc?: number; val_accuracy?: number; source?: string;
  per_generator: Record<string, { accuracy: number; auc: number; fake_acc: number }>;
}

export interface AblationRow {
  pillars: number; subset: string; spatial: number; frequency: number; color: number; physical: number;
  semantic: number; best_epoch: number; val_acc: number; test_acc: number; test_auc: number; test_ap: number;
  test_f1: number; real_acc: number; fake_acc: number; mean_gen_acc: number; worst_gen_acc: number;
  mean_gen_auc: number;
}

export interface ProtocolStats {
  accuracy: number; roc_auc: number; average_precision: number; real_acc: number; fake_acc: number; n: number;
}

export interface Research {
  generators: string[];
  versions: VersionRow[];
  ablation: AblationRow[];
  training: {
    data: Record<string, string | number>;
    stages: Record<string, string | number>[];
    spatial_history: Record<string, number>[];
    head_history: Record<string, number>[];
  };
  protocols: {
    windows_total: number;
    centre: ProtocolStats; fullscan: ProtocolStats;
    per_generator: Record<string, { centre: number; fullscan: number }>;
  } | null;
  attribution: {
    dataset: string; classes: string[]; images_per_split: Record<"train" | "val" | "test", Record<string, number>>;
    batch_size: number; max_epochs: number; epochs_run: number; best_epoch: number; test: AttributionEval;
    detector_zero_shot_genimage: { images: number; accuracy: number; auc: number; real_acc: number; fake_acc: number;
      ai_caught: Record<string, number> };
  } | null;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new Error("Cannot reach the Pixel-Artifacts server. Is it running?");
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

let modelCache: Promise<ModelInfo> | null = null;
let researchCache: Promise<Research> | null = null;

export const api = {
  model: () => (modelCache ??= request<ModelInfo>("/api/model").catch((e) => { modelCache = null; throw e; })),
  research: () => (researchCache ??= request<Research>("/api/research").catch((e) => { researchCache = null; throw e; })),
  analyze: (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    return request<{ results: Analysis[] }>("/api/analyze", { method: "POST", body: fd });
  },
  createJob: (files: File[], paths: string[], name: string) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f, f.name));
    fd.append("paths", JSON.stringify(paths));
    fd.append("name", name);
    return request<{ job_id: string }>("/api/jobs", { method: "POST", body: fd });
  },
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  jobItem: (id: string, index: number) => request<Analysis>(`/api/jobs/${id}/items/${index}`),
  thumbUrl: (id: string, index: number) => `/api/jobs/${id}/items/${index}/thumb`,
  csvUrl: (id: string) => `/api/jobs/${id}/report.csv`,
};
