type Step = { title: string; note: string };

/** Left-to-right model diagram: input → window → feature branches → fusion → verdict. */
export function Pipeline({ input, window, branches, fusion, verdict }: {
  input: Step; window: Step; branches: { name: string; dims: string }[]; fusion: Step; verdict: Step;
}) {
  return (
    <div className="pipeline">
      <div className="pipe-node"><b>{input.title}</b><span>{input.note}</span></div>
      <div className="pipe-arrow" />
      <div className="pipe-node"><b>{window.title}</b><span>{window.note}</span></div>
      <div className="pipe-arrow" />
      <div className="pipe-fan">
        {branches.map((l) => <div key={l.name} className="pipe-lens"><b>{l.name}</b><span>{l.dims}</span></div>)}
      </div>
      <div className="pipe-arrow" />
      <div className="pipe-node accent"><b>{fusion.title}</b><span>{fusion.note}</span></div>
      <div className="pipe-arrow" />
      <div className="pipe-node dark-node"><b>{verdict.title}</b><span>{verdict.note}</span></div>
    </div>
  );
}
