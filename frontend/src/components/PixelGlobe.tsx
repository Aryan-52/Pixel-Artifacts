import { useEffect, useRef } from "react";
import * as THREE from "three";

/**
 * Hero scene: a globe of pixel cubes. A scan line sweeps top to bottom; the few "artifact" pixels it
 * passes are lifted out and turn coral, the way the detector picks out generator traces.
 * Pauses when off screen; renders one still frame if the user prefers reduced motion.
 */
export default function PixelGlobe({ className = "" }: { className?: string }) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 50);
    camera.position.set(0, 0, 7.2);

    scene.add(new THREE.HemisphereLight(0xfff6e8, 0x0a3a42, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(3, 4, 5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0xe3a01c, 1.1);
    rim.position.set(-4, -2, -3);
    scene.add(rim);

    const world = new THREE.Group();
    scene.add(world);

    // Fibonacci sphere of cubes
    const N = 1500, R = 1.75;
    const geo = new THREE.BoxGeometry(0.075, 0.075, 0.075);
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 });
    const mesh = new THREE.InstancedMesh(geo, mat, N);
    const base: THREE.Vector3[] = [];
    const baseColor: THREE.Color[] = [];
    const artifact: boolean[] = [];
    const lift = new Float32Array(N);
    const palette = [new THREE.Color("#0f5c68"), new THREE.Color("#0a3a42"), new THREE.Color("#ebe2d2"),
                     new THREE.Color("#5f8f88")];
    const coral = new THREE.Color("#d4512a"), saffron = new THREE.Color("#e3a01c");
    let seed = 11;
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < N; i++) {
      const y = 1 - (i / (N - 1)) * 2;
      const r = Math.sqrt(1 - y * y);
      const th = golden * i;
      base.push(new THREE.Vector3(Math.cos(th) * r, y, Math.sin(th) * r).multiplyScalar(R));
      const u = rand();
      baseColor.push(palette[u < 0.46 ? 0 : u < 0.7 ? 1 : u < 0.88 ? 2 : 3]);
      artifact.push(rand() < 0.07);
    }
    world.add(mesh);

    // Orbit rings ("lenses")
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x0a3a42, transparent: true, opacity: 0.35 });
    const ring1 = new THREE.Mesh(new THREE.TorusGeometry(2.45, 0.006, 8, 200), ringMat);
    ring1.rotation.set(1.2, 0.2, 0);
    const ring2 = new THREE.Mesh(new THREE.TorusGeometry(2.75, 0.005, 8, 200), ringMat.clone());
    (ring2.material as THREE.MeshBasicMaterial).opacity = 0.2;
    ring2.rotation.set(1.9, -0.5, 0.3);
    scene.add(ring1, ring2);
    const probe = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 16), new THREE.MeshBasicMaterial({ color: 0xd4512a }));
    scene.add(probe);

    // Scan plane: a faint disc that moves with the scan line
    const scanDisc = new THREE.Mesh(new THREE.RingGeometry(0.2, 2.15, 64),
      new THREE.MeshBasicMaterial({ color: 0xe3a01c, transparent: true, opacity: 0.08, side: THREE.DoubleSide }));
    scanDisc.rotation.x = Math.PI / 2;
    world.add(scanDisc);

    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(1, 1, 1);
    const p = new THREE.Vector3(), up = new THREE.Vector3(0, 0, 1), dir = new THREE.Vector3(), c = new THREE.Color();
    const pointer = { x: 0, y: 0 };
    const onMove = (e: PointerEvent) => {
      const b = el.getBoundingClientRect();
      pointer.x = ((e.clientX - b.left) / b.width - 0.5) * 2;
      pointer.y = ((e.clientY - b.top) / b.height - 0.5) * 2;
    };
    window.addEventListener("pointermove", onMove, { passive: true });

    const resize = () => {
      const { width, height } = el.getBoundingClientRect();
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(height, 1);
      camera.position.z = camera.aspect < 0.9 ? 10 : 8.6;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    const PERIOD = 6.5;
    let raf = 0, visible = true, t0 = performance.now();
    const frame = (now: number) => {
      const t = (now - t0) / 1000;
      const phase = (t % PERIOD) / PERIOD;
      const scanY = R * 1.15 - phase * R * 2.3;
      scanDisc.position.y = scanY;
      const discScale = Math.sqrt(Math.max(0.02, 1 - (scanY / R) ** 2));
      scanDisc.scale.setScalar(discScale);
      for (let i = 0; i < N; i++) {
        const b = base[i];
        const d = Math.abs(b.y - scanY);
        const inBand = d < 0.16;
        const passed = b.y > scanY;
        let target = 0;
        if (artifact[i] && passed) target = 0.38;
        else if (inBand) target = 0.12 * (1 - d / 0.16);
        lift[i] += (target - lift[i]) * 0.12;
        dir.copy(b).normalize();
        p.copy(b).addScaledVector(dir, lift[i]);
        q.setFromUnitVectors(up, dir);
        s.setScalar(artifact[i] && passed ? 1.35 : 1);
        m4.compose(p, q, s);
        mesh.setMatrixAt(i, m4);
        if (artifact[i] && passed) c.copy(coral);
        else if (inBand) c.copy(baseColor[i]).lerp(saffron, 0.75 * (1 - d / 0.16));
        else c.copy(baseColor[i]);
        mesh.setColorAt(i, c);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      world.rotation.y = t * 0.16 + pointer.x * 0.35;
      world.rotation.x = 0.18 + pointer.y * 0.18;
      ring1.rotation.z = t * 0.22;
      ring2.rotation.z = -t * 0.14;
      const a = t * 0.9;
      probe.position.set(Math.cos(a) * 2.45, Math.sin(a) * 2.45 * Math.cos(1.2), Math.sin(a) * 2.45 * Math.sin(1.2));
      renderer.render(scene, camera);
      if (!reduce && visible) raf = requestAnimationFrame(frame);
    };
    const io = new IntersectionObserver(([e]) => {
      const was = visible;
      visible = e.isIntersecting;
      if (visible && !was && !reduce) raf = requestAnimationFrame(frame);
    });
    io.observe(el);
    if (reduce) { t0 = performance.now() - 3800; frame(performance.now()); } else raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
      ro.disconnect();
      window.removeEventListener("pointermove", onMove);
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        m.geometry?.dispose?.();
        const mm = m.material as THREE.Material | undefined;
        mm?.dispose?.();
      });
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={host} className={className} aria-hidden style={{ width: "100%", height: "100%" }} />;
}
