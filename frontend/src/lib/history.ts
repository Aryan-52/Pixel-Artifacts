// Recent analyses, kept only in this browser (no accounts, nothing sent anywhere).
import type { Analysis, Job } from "./api";

export interface HistoryEntry {
  id: string;
  kind: "image" | "batch";
  name: string;
  date: string;
  verdict?: "ai" | "real";
  p_ai?: number;
  thumb?: string;          // small data URL for images
  jobId?: string;
  images?: number;
  ai?: number;
}

const KEY = "pixel-artifacts.history.v1";
const MAX = 60;

export function readHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

function write(list: HistoryEntry[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX)));
  } catch { /* storage full or blocked: history is a convenience only */ }
}

async function tinyThumb(src: string, side = 96): Promise<string | undefined> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const s = side / Math.max(img.width, img.height);
      const c = Object.assign(document.createElement("canvas"), {
        width: Math.round(img.width * s), height: Math.round(img.height * s),
      });
      c.getContext("2d")?.drawImage(img, 0, 0, c.width, c.height);
      resolve(c.toDataURL("image/jpeg", 0.7));
    };
    img.onerror = () => resolve(undefined);
    img.src = src;
  });
}

export async function rememberImage(a: Analysis) {
  const thumb = a.visuals ? await tinyThumb(a.visuals.original) : undefined;
  const entry: HistoryEntry = {
    id: `${a.sha256}-${Date.now()}`, kind: "image", name: a.filename, date: new Date().toISOString(),
    verdict: a.verdict, p_ai: a.p_ai, thumb,
  };
  write([entry, ...readHistory().filter((e) => !(e.kind === "image" && e.id.startsWith(a.sha256)))]);
}

export function rememberJob(job: Job) {
  if (readHistory().some((e) => e.jobId === job.id)) return;
  write([{ id: `job-${job.id}`, kind: "batch", name: job.name, date: new Date().toISOString(), jobId: job.id,
           images: job.summary?.scored, ai: job.summary?.ai }, ...readHistory()]);
}

export function clearHistory() {
  write([]);
}

export function removeEntry(id: string) {
  write(readHistory().filter((e) => e.id !== id));
}
