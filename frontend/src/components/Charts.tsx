// Small hand-made SVG charts. Colours come from the CSS palette so they match the app.
import { useId, type ReactNode } from "react";

const C = { ink: "#13262b", muted: "#6c797a", line: "#ddd3c1", teal: "#0f5c68", coral: "#d4512a", sage: "#3c7a59", saffron: "#e3a01c", plum: "#6b4e71" };

/** Semicircle gauge for the AI probability. */
export function Gauge({ value, size = 240 }: { value: number; size?: number }) {
  const r = 92, cx = 120, cy = 118;
  const arc = (a0: number, a1: number) => {
    const p = (a: number) => [cx + r * Math.cos(Math.PI * (1 - a)), cy - r * Math.sin(Math.PI * (1 - a))];
    const [x0, y0] = p(a0), [x1, y1] = p(a1);
    return `M${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1}`;
  };
  const v = Math.min(1, Math.max(0, value));
  const ang = Math.PI * (1 - v);
  const nx = cx + (r - 18) * Math.cos(ang), ny = cy - (r - 18) * Math.sin(ang);
  return (
    <svg viewBox="0 0 240 150" width={size} role="img" aria-label={`AI probability ${(v * 100).toFixed(0)} percent`}>
      <path d={arc(0, 0.3)} stroke={C.sage} strokeWidth="16" fill="none" strokeLinecap="round" />
      <path d={arc(0.3, 0.7)} stroke={C.saffron} strokeWidth="16" fill="none" opacity=".75" />
      <path d={arc(0.7, 1)} stroke={C.coral} strokeWidth="16" fill="none" strokeLinecap="round" />
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={C.ink} strokeWidth="4" strokeLinecap="round"
        style={{ transformOrigin: `${cx}px ${cy}px`, animation: "needle 1.1s cubic-bezier(.2,.7,.2,1)" }} />
      <circle cx={cx} cy={cy} r="9" fill={C.ink} />
      <text x="22" y="146" fontSize="11" fill={C.muted} fontFamily="Manrope">Real</text>
      <text x="196" y="146" fontSize="11" fill={C.muted} fontFamily="Manrope">AI</text>
      <style>{`@keyframes needle { from { transform: rotate(${-(Math.PI - ang) * 180 / Math.PI}deg); } }`}</style>
    </svg>
  );
}

/** ROC curve with an optional operating point. */
export function RocChart({ fpr, tpr, auc, point, color = C.teal, label }: {
  fpr: number[]; tpr: number[]; auc: number; point?: { fpr: number; tpr: number }; color?: string; label?: string;
}) {
  const W = 320, H = 300, P = 40;
  const x = (v: number) => P + v * (W - P - 12), y = (v: number) => H - P - v * (H - P - 12);
  const d = fpr.map((f, i) => `${i ? "L" : "M"}${x(f).toFixed(1)},${y(tpr[i]).toFixed(1)}`).join("");
  const id = useId();
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 420 }} role="img" aria-label={`ROC curve, AUC ${auc.toFixed(3)}`}>
      <defs>
        <clipPath id={id}><rect x={P} y={12} width={W - P - 12} height={H - P - 12} /></clipPath>
      </defs>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={x(g)} x2={x(g)} y1={y(0)} y2={y(1)} stroke={C.line} />
          <line x1={x(0)} x2={x(1)} y1={y(g)} y2={y(g)} stroke={C.line} />
          <text x={x(g)} y={H - P + 16} fontSize="10" textAnchor="middle" fill={C.muted}>{g}</text>
          <text x={P - 8} y={y(g) + 3} fontSize="10" textAnchor="end" fill={C.muted}>{g}</text>
        </g>
      ))}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={C.muted} strokeDasharray="4 4" />
      <path d={`${d}L${x(1)},${y(0)}L${x(0)},${y(0)}Z`} fill={color} opacity=".1" clipPath={`url(#${id})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="2.5" />
      {point && (
        <g>
          <circle cx={x(point.fpr)} cy={y(point.tpr)} r="6" fill={C.coral} stroke="#fff" strokeWidth="2" />
          <text x={x(point.fpr) + 10} y={y(point.tpr) + 16} fontSize="10.5" fill={C.ink}>threshold 0.5</text>
        </g>
      )}
      <text x={W / 2 + 10} y={H - 4} fontSize="11" textAnchor="middle" fill={C.ink}>False positive rate (real flagged as AI)</text>
      <text transform={`translate(11 ${H / 2 - 10}) rotate(-90)`} fontSize="11" textAnchor="middle" fill={C.ink}>True positive rate (AI caught)</text>
      <text x={x(0.55)} y={y(0.2)} fontSize="15" fontFamily="Fraunces" fill={C.ink}>{label ?? "AUC"} {auc.toFixed(3)}</text>
    </svg>
  );
}

/** Two-class score histogram (benchmark real vs AI), with an optional marker for one score. */
export function ScoreHistogram({ real, ai, marker }: { real: number[]; ai: number[]; marker?: number }) {
  const W = 520, H = 210, P = 30, n = real.length;
  const max = Math.max(...real, ...ai);
  const bw = (W - P * 2) / n;
  const h = (v: number) => (Math.sqrt(v / max)) * (H - P - 26);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Benchmark score distribution">
      {real.map((v, i) => (
        <rect key={`r${i}`} x={P + i * bw + 1} y={H - P - h(v)} width={bw / 2 - 1} height={h(v)} fill={C.sage} rx="2" />
      ))}
      {ai.map((v, i) => (
        <rect key={`a${i}`} x={P + i * bw + bw / 2} y={H - P - h(v)} width={bw / 2 - 1} height={h(v)} fill={C.coral} rx="2" />
      ))}
      <line x1={P} x2={W - P} y1={H - P} y2={H - P} stroke={C.ink} />
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <text key={g} x={P + g * (W - 2 * P)} y={H - P + 16} fontSize="10" textAnchor="middle" fill={C.muted}>{g}</text>
      ))}
      <line x1={P + 0.5 * (W - 2 * P)} x2={P + 0.5 * (W - 2 * P)} y1={14} y2={H - P} stroke={C.muted} strokeDasharray="3 4" />
      {marker != null && (
        <g>
          <line x1={P + marker * (W - 2 * P)} x2={P + marker * (W - 2 * P)} y1={10} y2={H - P} stroke={C.ink} strokeWidth="2.5" />
          <rect x={P + marker * (W - 2 * P) - 38} y={0} width={76} height={18} rx="9" fill={C.ink} />
          <text x={P + marker * (W - 2 * P)} y={13} fontSize="10.5" textAnchor="middle" fill="#fff">this image</text>
        </g>
      )}
      <g fontSize="11" fill={C.ink}>
        <rect x={W - 150} y={22} width="10" height="10" fill={C.sage} rx="2" /><text x={W - 135} y={31}>real images</text>
        <rect x={W - 150} y={40} width="10" height="10" fill={C.coral} rx="2" /><text x={W - 135} y={49}>AI images</text>
      </g>
      <text x={W / 2} y={H - 2} fontSize="10.5" textAnchor="middle" fill={C.muted}>AI probability given by the detector (bar height: square-root scale)</text>
    </svg>
  );
}

/** Horizontal bars, values 0..1 by default. */
export function BarList({ rows, max = 1, format = (v) => `${(v * 100).toFixed(1)}%`, color = C.teal }: {
  rows: { label: ReactNode; value: number; color?: string; note?: ReactNode }[];
  max?: number; format?: (v: number) => string; color?: string;
}) {
  return (
    <div className="barlist">
      {rows.map((r, i) => (
        <div className="barlist-row" key={i}>
          <div className="barlist-label">{r.label}</div>
          <div className="barlist-track">
            <div className="barlist-fill" style={{ width: `${Math.max(0, Math.min(1, r.value / max)) * 100}%`, background: r.color ?? color, ["--i" as string]: i }} />
          </div>
          <div className="barlist-value">{format(r.value)}{r.note}</div>
        </div>
      ))}
    </div>
  );
}

/** Diverging bars around zero (lens influence). Positive = toward AI. */
export function Diverging({ rows }: { rows: { label: ReactNode; value: number }[] }) {
  const m = Math.max(0.5, ...rows.map((r) => Math.abs(r.value)));
  return (
    <div className="diverging">
      {rows.map((r, i) => {
        const w = (Math.abs(r.value) / m) * 50;
        return (
          <div className="div-row" key={i}>
            <div className="div-label">{r.label}</div>
            <div className="div-track">
              <span className="div-axis" />
              <span className="div-fill" style={{
                left: r.value >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`,
                background: r.value >= 0 ? C.coral : C.sage,
              }} />
            </div>
            <div className="div-value mono">{r.value >= 0 ? "+" : ""}{r.value.toFixed(2)}</div>
          </div>
        );
      })}
      <div className="div-legend small muted"><span>← toward real</span><span>toward AI →</span></div>
    </div>
  );
}

/** Simple multi-series line chart for training curves. */
export function LineChart({ series, xLabel, yLabel, height = 220 }: {
  series: { name: string; color: string; points: [number, number][]; dashed?: boolean }[];
  xLabel: string; yLabel: string; height?: number;
}) {
  const W = 520, H = height, P = 44;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = 0, y1 = Math.max(...ys) * 1.1 || 1;
  const X = (v: number) => P + ((v - x0) / Math.max(x1 - x0, 1e-9)) * (W - P - 14);
  const Y = (v: number) => H - P + 8 - ((v - y0) / (y1 - y0)) * (H - P - 14);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={yLabel}>
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line x1={P} x2={W - 14} y1={Y(y0 + g * (y1 - y0))} y2={Y(y0 + g * (y1 - y0))} stroke={C.line} />
          <text x={P - 6} y={Y(y0 + g * (y1 - y0)) + 3} fontSize="10" textAnchor="end" fill={C.muted}>{(y0 + g * (y1 - y0)).toPrecision(2)}</text>
        </g>
      ))}
      {Array.from(new Set(xs)).map((v) => (
        <text key={v} x={X(v)} y={H - P + 24} fontSize="10" textAnchor="middle" fill={C.muted}>{v}</text>
      ))}
      {series.map((s) => (
        <g key={s.name}>
          <path d={s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0])},${Y(p[1])}`).join("")} fill="none"
            stroke={s.color} strokeWidth="2.4" strokeDasharray={s.dashed ? "5 5" : undefined} />
          {s.points.map((p) => <circle key={p[0]} cx={X(p[0])} cy={Y(p[1])} r="3" fill={s.color} />)}
        </g>
      ))}
      <g fontSize="11" fill={C.ink}>
        {series.map((s, i) => (
          <g key={s.name} transform={`translate(${W - 170} ${14 + i * 16})`}>
            <line x1="0" x2="16" y1="0" y2="0" stroke={s.color} strokeWidth="2.4" strokeDasharray={s.dashed ? "4 3" : undefined} />
            <text x="22" y="4">{s.name}</text>
          </g>
        ))}
      </g>
      <text x={W / 2} y={H - 4} fontSize="11" textAnchor="middle" fill={C.ink}>{xLabel}</text>
      <text transform={`translate(11 ${H / 2 - 12}) rotate(-90)`} fontSize="11" textAnchor="middle" fill={C.ink}>{yLabel}</text>
    </svg>
  );
}

/** Donut of AI vs real counts. */
export function Donut({ ai, real, size = 180 }: { ai: number; real: number; size?: number }) {
  const total = Math.max(ai + real, 1), r = 70, L = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 180 180" width={size} role="img" aria-label={`${ai} AI, ${real} real`}>
      <circle cx="90" cy="90" r={r} fill="none" stroke={C.sage} strokeWidth="22" />
      <circle cx="90" cy="90" r={r} fill="none" stroke={C.coral} strokeWidth="22"
        strokeDasharray={`${(ai / total) * L} ${L}`} transform="rotate(-90 90 90)" />
      <text x="90" y="88" textAnchor="middle" fontSize="30" fontFamily="Fraunces" fill={C.ink}>{Math.round((ai / total) * 100)}%</text>
      <text x="90" y="108" textAnchor="middle" fontSize="11" fill={C.muted}>flagged as AI</text>
    </svg>
  );
}

/** 2x2 confusion matrix. */
export function Confusion({ tn, fp, fn, tp }: { tn: number; fp: number; fn: number; tp: number }) {
  const cell = (v: number, good: boolean, label: string) => (
    <div className={`cm-cell ${good ? "good" : "bad"}`}><b>{v.toLocaleString("en-IN")}</b><span>{label}</span></div>
  );
  return (
    <div className="cm">
      <div />
      <div className="cm-h">Said real</div>
      <div className="cm-h">Said AI</div>
      <div className="cm-h side">Actually real</div>
      {cell(tn, true, "correct")}
      {cell(fp, false, "false alarm")}
      <div className="cm-h side">Actually AI</div>
      {cell(fn, false, "missed")}
      {cell(tp, true, "caught")}
    </div>
  );
}
