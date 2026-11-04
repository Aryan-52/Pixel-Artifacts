import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, type Analysis } from "../lib/api";
import { IMAGE_EXT, pct } from "../lib/format";
import { rememberImage } from "../lib/history";
import { ResultView } from "../components/Result";

type Mode = "images" | "folder" | "zip";
const MAX_DIRECT = 12;

interface Sample { file: string; generator: string; label: "ai" | "real"; p_ai: number }

const MODES: { key: Mode; title: string; hint: string }[] = [
  { key: "images", title: "Images", hint: `1 to ${MAX_DIRECT} images, full analysis each` },
  { key: "folder", title: "Folder", hint: "A whole folder, with a CSV report" },
  { key: "zip", title: "ZIP archive", hint: "A .zip of images, with a CSV report" },
];

const STEPS = ["Reading pixels", "Texture & detail lens", "Frequency lens", "Colour and light lenses", "Overall-look lens", "Fusing the evidence", "Drawing the heatmaps"];

/** Walk a dropped folder (webkitGetAsEntry) and return files with their relative paths. */
async function walk(entry: FileSystemEntry, prefix = ""): Promise<{ file: File; path: string }[]> {
  if (entry.isFile) {
    const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
    return [{ file, path: prefix + entry.name }];
  }
  const reader = (entry as FileSystemDirectoryEntry).createReader();
  const all: FileSystemEntry[] = [];
  for (;;) {
    const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
    if (!batch.length) break;
    all.push(...batch);
  }
  const nested = await Promise.all(all.map((e) => walk(e, `${prefix}${entry.name}/`)));
  return nested.flat();
}

function Scanner({ preview, count }: { preview?: string; count: number }) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 650);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="scanner card">
      <div className="scanner-media">
        {preview ? <img src={preview} alt="" /> : <div className="scanner-blank" />}
        <span className="scan-line" />
        <span className="scan-grid" />
      </div>
      <div className="stack">
        <span className="eyebrow">Analysing {count > 1 ? `${count} images` : "image"}</span>
        <h3>Looking through five lenses…</h3>
        <ol className="steps">
          {STEPS.map((s, i) => <li key={s} className={i < step ? "done" : i === step ? "now" : ""}>{s}</li>)}
        </ol>
      </div>
    </div>
  );
}

export default function Detect() {
  const [params, setParams] = useSearchParams();
  const mode = (params.get("mode") as Mode) || "images";
  const setMode = (m: Mode) => setParams(m === "images" ? {} : { mode: m }, { replace: true });
  const nav = useNavigate();
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState<{ preview?: string; count: number; batch?: boolean } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<Analysis[]>([]);
  const [sel, setSel] = useState(0);
  const [samples, setSamples] = useState<Sample[]>([]);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/samples/samples.json").then((r) => r.json()).then(setSamples).catch(() => setSamples([]));
  }, []);

  async function runImages(files: File[]) {
    setError(null);
    if (files.length > MAX_DIRECT) return runBatch(files.map((f) => ({ file: f, path: f.name })), `${files.length} images`);
    setBusy({ preview: URL.createObjectURL(files[0]), count: files.length });
    try {
      const { results } = await api.analyze(files);
      setResults(results);
      setSel(0);
      results.filter((r) => !r.error).forEach((r) => rememberImage(r));
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function runBatch(entries: { file: File; path: string }[], name: string) {
    setError(null);
    const imgs = entries.filter((e) => IMAGE_EXT.test(e.path) || /\.zip$/i.test(e.path));
    if (!imgs.length) { setError("No supported images found (JPG, PNG, WEBP, BMP or TIFF)."); return; }
    setBusy({ count: imgs.length, batch: true });
    try {
      const { job_id } = await api.createJob(imgs.map((e) => e.file), imgs.map((e) => e.path), name);
      nav(`/batch/${job_id}`);
    } catch (e) {
      setError((e as Error).message);
      setBusy(null);
    }
  }

  function onPick(list: FileList | null) {
    if (!list?.length) return;
    const files = Array.from(list);
    if (mode === "zip") return runBatch([{ file: files[0], path: files[0].name }], files[0].name);
    if (mode === "folder") {
      const root = (files[0] as File & { webkitRelativePath: string }).webkitRelativePath.split("/")[0] || "Folder";
      return runBatch(files.map((f) => ({ file: f, path: (f as File & { webkitRelativePath: string }).webkitRelativePath || f.name })), root);
    }
    runImages(files.filter((f) => IMAGE_EXT.test(f.name)));
  }

  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDrag(false);
    const items = Array.from(e.dataTransfer.items || []);
    const entries = items.map((i) => i.webkitGetAsEntry?.()).filter(Boolean) as FileSystemEntry[];
    if (entries.some((x) => x.isDirectory)) {
      const all = (await Promise.all(entries.map((x) => walk(x)))).flat();
      return runBatch(all, entries.find((x) => x.isDirectory)!.name);
    }
    const files = Array.from(e.dataTransfer.files);
    const zip = files.find((f) => /\.zip$/i.test(f.name));
    if (zip) return runBatch([{ file: zip, path: zip.name }], zip.name);
    runImages(files.filter((f) => IMAGE_EXT.test(f.name)));
  }

  async function trySample(s: Sample) {
    const blob = await (await fetch(`/${s.file}`)).blob();
    runImages([new File([blob], s.file.split("/").pop()!, { type: blob.type })]);
  }

  const accept = mode === "zip" ? ".zip,application/zip" : "image/*";
  const current = results[sel];
  return (
    <>
      <section className="wrap page-head detect-head">
        <span className="eyebrow">Detector</span>
        <h1>Is it real, or made by AI?</h1>
        <p className="lede">Drop an image, a handful of images, a folder or a ZIP. You get a clear answer first; the evidence is one click away.</p>
      </section>

      <section className="wrap">
        <div className="mode-pick" role="radiogroup" aria-label="What are you checking?">
          {MODES.map((m) => (
            <button key={m.key} role="radio" aria-checked={mode === m.key} className={mode === m.key ? "on" : ""} onClick={() => setMode(m.key)}>
              <ModeIcon mode={m.key} />
              <span><b>{m.title}</b><small>{m.hint}</small></span>
            </button>
          ))}
        </div>

        {busy ? (
          busy.batch ? (
            <div className="scanner card"><div className="stack"><span className="eyebrow">Uploading</span><h3>Sending {busy.count} files to the detector…</h3><div className="bar-indef" /></div></div>
          ) : <Scanner preview={busy.preview} count={busy.count} />
        ) : (
          <div className={`drop ${drag ? "drag" : ""}`} onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)} onDrop={onDrop} onClick={() => input.current?.click()}
            role="button" tabIndex={0} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}>
            <input ref={input} type="file" hidden accept={accept} multiple={mode === "images"}
              {...(mode === "folder" ? { webkitdirectory: "", directory: "" } : {})}
              onChange={(e) => { onPick(e.target.files); e.target.value = ""; }} />
            <DropArt />
            <h3>{mode === "images" ? "Drop images here" : mode === "folder" ? "Drop a folder here" : "Drop a ZIP here"}</h3>
            <p className="muted">or <span className="link-btn">browse your computer</span></p>
            <p className="small muted">
              {mode === "images" ? `JPG, PNG, WEBP, BMP or TIFF · up to 25 MB each · more than ${MAX_DIRECT} images become a batch`
                : "Sub-folders named real / fake (or 0_real / 1_fake) are treated as labels, and you also get accuracy, AUC and a ROC curve"}
            </p>
          </div>
        )}
        {error && <div className="error-box" style={{ marginTop: 18 }}>{error}</div>}

        {!busy && !results.length && samples.length > 0 && (
          <div className="samples">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h4>No image handy? Try one from the benchmark</h4>
              <span className="small muted">Labels are hidden until you check</span>
            </div>
            <div className="sample-strip">
              {samples.filter((_, i) => i % 3 !== 2).slice(0, 10).map((s) => (
                <button key={s.file} className="sample" onClick={() => trySample(s)} title="Analyse this sample">
                  <img src={`/${s.file}`} alt="benchmark sample" loading="lazy" />
                </button>
              ))}
            </div>
          </div>
        )}
      </section>

      {results.length > 0 && !busy && (
        <section className="wrap section-tight" ref={resultsRef}>
          {results.length > 1 && (
            <div className="result-rail">
              {results.map((r, i) => (
                <button key={i} className={`rail-item ${i === sel ? "on" : ""}`} onClick={() => setSel(i)}>
                  {r.visuals ? <img src={r.visuals.original} alt="" /> : <span className="rail-err">!</span>}
                  <span className="rail-meta">
                    <span className="rail-name">{r.filename}</span>
                    {r.error ? <span className="chip low">unreadable</span>
                      : <span className={`chip ${r.verdict}`}>{r.verdict === "ai" ? "AI" : "Real"} · {pct(r.verdict === "ai" ? r.p_ai : r.p_real, 0)}</span>}
                  </span>
                </button>
              ))}
            </div>
          )}
          {current && <ResultView a={current} />}
          <div className="row" style={{ justifyContent: "center", marginTop: 36 }}>
            <button className="btn" onClick={() => { setResults([]); window.scrollTo({ top: 0, behavior: "smooth" }); }}>Check another image</button>
          </div>
        </section>
      )}
    </>
  );
}

function ModeIcon({ mode }: { mode: Mode }) {
  const s = { stroke: "currentColor", strokeWidth: 1.8, fill: "none", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden>
      {mode === "images" && (<><rect x="5" y="8" width="20" height="18" rx="3" {...s} /><rect x="9" y="4" width="20" height="18" rx="3" {...s} /><path d="M12 18l4-4 5 5 3-2 4 4" {...s} /></>)}
      {mode === "folder" && (<><path d="M4 10a3 3 0 013-3h6l3 3h11a3 3 0 013 3v11a3 3 0 01-3 3H7a3 3 0 01-3-3z" {...s} /><path d="M10 19h14" {...s} /></>)}
      {mode === "zip" && (<><rect x="7" y="4" width="20" height="26" rx="3" {...s} /><path d="M17 4v4m0 3v3m0 3v3" {...s} /><rect x="14" y="20" width="6" height="5" rx="1.5" {...s} /></>)}
    </svg>
  );
}

function DropArt() {
  return (
    <svg className="drop-art" width="150" height="110" viewBox="0 0 150 110" aria-hidden>
      <rect x="30" y="18" width="78" height="66" rx="10" fill="#ebe2d2" transform="rotate(-8 69 51)" />
      <rect x="40" y="14" width="78" height="66" rx="10" fill="#fbf8f2" stroke="#0f5c68" strokeWidth="2" />
      <circle cx="62" cy="36" r="7" fill="#e3a01c" />
      <path d="M46 72l20-20 14 14 10-8 22 14" fill="none" stroke="#0f5c68" strokeWidth="2.4" strokeLinejoin="round" />
      {[0, 1, 2, 3].map((i) => <rect key={i} x={96 + (i % 2) * 9} y={56 + Math.floor(i / 2) * 9} width="7" height="7" rx="1.5" fill={i === 3 ? "#d4512a" : "#0f5c68"} opacity={i === 3 ? 1 : 0.5} />)}
      <circle cx="118" cy="82" r="18" fill="#0f5c68" />
      <path d="M118 90V74m-7 7l7-7 7 7" stroke="#f4eee3" strokeWidth="2.6" fill="none" strokeLinecap="round" />
    </svg>
  );
}
