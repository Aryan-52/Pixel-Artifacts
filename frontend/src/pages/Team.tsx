import { Reveal } from "../components/Reveal";

// Photos: put files in frontend/public/team/ and set `photo` (e.g. "/team/aryan.jpg").
interface Person { name: string; role: string; photo?: string; tone: string }

const TEAM: Person[] = [
  { name: "Aryan Dandekar", role: "Team member · MCA", tone: "#0f5c68" },
  { name: "Gyana Karn", role: "Team member · MCA", tone: "#6b4e71" },
  { name: "Sahil Sawant", role: "Team member · MCA", tone: "#d4512a" },
];

const MENTORS: Person[] = [
  { name: "Prof. Sakina Salmani", role: "Project guide", tone: "#3c7a59" },
  { name: "Prof. Harshil Kanakia", role: "Project guide", tone: "#e3a01c" },
];

const initials = (n: string) => n.replace(/^Prof\.\s*/, "").split(" ").map((w) => w[0]).slice(0, 2).join("");

function Avatar({ p, big }: { p: Person; big?: boolean }) {
  return (
    <div className={`avatar ${big ? "big" : ""}`} style={{ ["--tone" as string]: p.tone }}>
      {p.photo ? <img src={p.photo} alt={p.name} /> : (
        <svg viewBox="0 0 100 100" aria-hidden>
          {Array.from({ length: 25 }).map((_, i) => (
            <rect key={i} x={(i % 5) * 20} y={Math.floor(i / 5) * 20} width="19" height="19" fill="var(--tone)"
              opacity={((i * 7 + p.name.length * 3) % 10) / 22 + 0.08} />
          ))}
          <text x="50" y="61" textAnchor="middle" fontSize="30" fontFamily="Fraunces" fill="#fbf8f2">{initials(p.name)}</text>
        </svg>
      )}
    </div>
  );
}

export default function Team() {
  return (
    <>
      <section className="wrap page-head">
        <span className="eyebrow">The people</span>
        <h1>Team & mentors</h1>
        <p className="lede">Pixel-Artifacts is the MCA major project of three students at Sardar Patel Institute of Technology, Mumbai, guided by two faculty mentors.</p>
      </section>
      <section className="wrap">
        <div className="grid-3">
          {TEAM.map((p, i) => (
            <Reveal key={p.name} delay={i * 0.08} className="person">
              <Avatar p={p} big />
              <h3>{p.name}</h3>
              <p className="small muted">{p.role}</p>
            </Reveal>
          ))}
        </div>
        <div className="section-tight">
          <span className="eyebrow">Guided by</span>
          <div className="grid-2" style={{ marginTop: 24 }}>
            {MENTORS.map((p, i) => (
              <Reveal key={p.name} delay={i * 0.08} className="mentor">
                <Avatar p={p} />
                <div><h3>{p.name}</h3><p className="small muted">{p.role} · Sardar Patel Institute of Technology</p></div>
              </Reveal>
            ))}
          </div>
        </div>
        <Reveal className="card flat thanks">
          <h3>Acknowledgements</h3>
          <p className="muted">We thank our guides for their review of every version of this work, and the authors of the CNNDetection benchmark (Wang et al., CVPR 2020) for making their dataset public.</p>
        </Reveal>
      </section>
      <div className="spacer" /><div className="spacer" />
    </>
  );
}
