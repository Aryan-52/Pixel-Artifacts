import { useEffect, useRef, useState, type ReactNode, type CSSProperties, type ElementType } from "react";

/** Fades/slides its children in the first time they scroll into view. */
export function Reveal({ children, delay = 0, as: Tag = "div", className = "", style }:
  { children: ReactNode; delay?: number; as?: ElementType; className?: string; style?: CSSProperties }) {
  const ref = useRef<HTMLElement>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setInView(true); io.disconnect(); }
    }, { rootMargin: "0px 0px -8% 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <Tag ref={ref} className={`reveal ${inView ? "in" : ""} ${className}`}
      style={{ ...style, ["--d" as string]: `${delay}s` }}>
      {children}
    </Tag>
  );
}

/** Counts up to `value` once visible. */
export function CountUp({ value, decimals = 0, suffix = "", duration = 1400 }:
  { value: number; decimals?: number; suffix?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    const io = new IntersectionObserver(([e]) => {
      if (!e.isIntersecting) return;
      io.disconnect();
      const t0 = performance.now();
      const tick = (t: number) => {
        const k = Math.min(1, (t - t0) / duration);
        setShown(value * (1 - Math.pow(1 - k, 3)));
        if (k < 1) raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
    });
    io.observe(el);
    return () => { io.disconnect(); cancelAnimationFrame(raf); };
  }, [value, duration]);
  return <span ref={ref}>{shown.toLocaleString("en-IN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}{suffix}</span>;
}
