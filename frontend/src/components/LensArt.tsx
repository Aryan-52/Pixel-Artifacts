// Small illustrations for the five lenses (inline SVG, palette colours only).

export const LENS_COPY = [
  { key: "spatial", name: "Texture & detail", tech: "EfficientNet-B0 (fine-tuned)", plain: "Fine textures, edges and tiny glitches that generators get subtly wrong." },
  { key: "frequency", name: "Hidden patterns", tech: "Fourier spectrum", plain: "Repeating pixel patterns left by the upsampling layers inside generators." },
  { key: "color", name: "Colour balance", tech: "RGB · YCbCr · opponent", plain: "Whether colours relate to each other the way a camera sensor records them." },
  { key: "physical", name: "Light & noise", tech: "Shading & noise cues", plain: "Whether lighting, shading and grain behave consistently across the image." },
  { key: "semantic", name: "Overall look", tech: "CLIP ViT-B/32", plain: "The image as a whole, the way a person would take it in at a glance." },
] as const;

export function LensArt({ k }: { k: string }) {
  const T = "#0f5c68", C = "#d4512a", S = "#e3a01c", P = "#ebe2d2", I = "#0a3a42";
  return (
    <svg className="lens-art" viewBox="0 0 120 90" aria-hidden>
      <rect x="0" y="0" width="120" height="90" rx="14" fill={P} />
      {k === "spatial" && (
        <g>
          {Array.from({ length: 30 }).map((_, i) => (
            <rect key={i} x={14 + (i % 6) * 16} y={10 + Math.floor(i / 6) * 14} width="13" height="11" rx="2"
              fill={i === 14 || i === 21 ? C : T} opacity={i === 14 || i === 21 ? 1 : 0.18 + ((i * 37) % 10) / 16} />
          ))}
        </g>
      )}
      {k === "frequency" && (
        <g>
          <circle cx="60" cy="45" r="30" fill={I} />
          <circle cx="60" cy="45" r="5" fill={S} />
          {[[-16, 0], [16, 0], [0, -16], [0, 16]].map(([dx, dy], i) => <circle key={i} cx={60 + dx} cy={45 + dy} r="2.6" fill={C} />)}
          <circle cx="60" cy="45" r="16" fill="none" stroke={S} strokeWidth="1" strokeDasharray="2 3" opacity=".7" />
        </g>
      )}
      {k === "color" && (
        <g>
          <circle cx="48" cy="40" r="20" fill={C} opacity=".7" />
          <circle cx="72" cy="40" r="20" fill={T} opacity=".7" />
          <circle cx="60" cy="58" r="20" fill={S} opacity=".7" />
        </g>
      )}
      {k === "physical" && (
        <g>
          <circle cx="92" cy="20" r="8" fill={S} />
          <path d="M92 20L50 52" stroke={S} strokeWidth="1.5" strokeDasharray="3 3" />
          <ellipse cx="50" cy="70" rx="26" ry="5" fill={I} opacity=".25" />
          <circle cx="50" cy="54" r="15" fill={T} />
          <path d="M40 46a15 15 0 0 1 14-6" stroke={P} strokeWidth="3" fill="none" strokeLinecap="round" opacity=".8" />
        </g>
      )}
      {k === "semantic" && (
        <g>
          <path d="M20 66l22-26 14 16 10-10 24 20z" fill={T} />
          <circle cx="82" cy="28" r="9" fill={S} />
          <rect x="12" y="12" width="96" height="66" rx="8" fill="none" stroke={I} strokeWidth="2" strokeDasharray="6 4" />
        </g>
      )}
    </svg>
  );
}
