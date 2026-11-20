// Réglage discret du raisonnement : « Effort (Medium) » et une rangée de points, le niveau choisi en gros point.
// On peut cliquer sur un point, faire glisser le point (souris ou doigt) ou utiliser les flèches du clavier.
import { useRef } from "react";

export default function EffortBar({ levels, value, onChange, hint }) {
  const track = useRef(null);
  const dragging = useRef(false);
  const pick = (clientX) => {
    // point le plus proche de la position du pointeur
    const dots = [...track.current.querySelectorAll(".dot")];
    let best = 0;
    dots.forEach((d, i) => {
      const r = d.getBoundingClientRect();
      const b = dots[best].getBoundingClientRect();
      if (Math.abs(clientX - (r.left + r.width / 2)) < Math.abs(clientX - (b.left + b.width / 2))) best = i;
    });
    if (levels.levels[best] !== value) onChange(levels.levels[best]);
  };
  const index = Math.max(0, levels.levels.indexOf(value));
  const move = (delta) => {
    const next = Math.min(levels.levels.length - 1, Math.max(0, index + delta));
    if (next !== index) onChange(levels.levels[next]);
  };
  return (
    <div className="effort" title={hint}>
      <span className="effort-label">
        Effort <span className="effort-value">({levels.labels[value]})</span>
      </span>
      <div
        ref={track}
        className="effort-dots"
        onPointerDown={(e) => {
          dragging.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          pick(e.clientX);
        }}
        onPointerMove={(e) => dragging.current && pick(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
        role="slider"
        tabIndex={0}
        aria-label="Effort de raisonnement"
        aria-valuemin={0}
        aria-valuemax={levels.levels.length - 1}
        aria-valuenow={index}
        aria-valuetext={levels.labels[value]}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft" || e.key === "ArrowDown") (e.preventDefault(), move(-1));
          if (e.key === "ArrowRight" || e.key === "ArrowUp") (e.preventDefault(), move(1));
        }}
      >
        {levels.levels.map((level, i) => (
          <button
            key={level}
            type="button"
            tabIndex={-1}
            className={i === index ? "dot active" : "dot"}
            aria-label={levels.labels[level]}
          />
        ))}
      </div>
    </div>
  );
}
