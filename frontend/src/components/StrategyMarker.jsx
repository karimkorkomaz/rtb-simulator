// Shape-coded markers so strategies stay distinguishable without colour
// (greyscale printing / projection) -- used by both charts and as the
// legend swatch in the table.
export function MarkerShape({ shape, color, size = 5, x, y }) {
  const s = size;
  if (shape === "square") {
    return <rect x={x - s} y={y - s} width={s * 2} height={s * 2} fill={color} stroke="#fff" strokeWidth={1} />;
  }
  if (shape === "diamond") {
    const points = [
      [x, y - s * 1.25],
      [x + s * 1.25, y],
      [x, y + s * 1.25],
      [x - s * 1.25, y],
    ]
      .map((p) => p.join(","))
      .join(" ");
    return <polygon points={points} fill={color} stroke="#fff" strokeWidth={1} />;
  }
  // default: circle
  return <circle cx={x} cy={y} r={s} fill={color} stroke="#fff" strokeWidth={1} />;
}

/** Small inline legend swatch (line + marker) for table headers / row
 * labels, matching the exact style used on the charts. */
export function LegendSwatch({ meta }) {
  return (
    <svg width="28" height="14" viewBox="0 0 28 14" aria-hidden="true" className="inline-block align-middle">
      <line
        x1="1"
        y1="7"
        x2="27"
        y2="7"
        stroke={meta.color}
        strokeWidth="2.5"
        strokeDasharray={meta.dash}
        strokeLinecap="round"
      />
      <MarkerShape shape={meta.marker} color={meta.color} size={4} x={14} y={7} />
    </svg>
  );
}
