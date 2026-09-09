import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { STRATEGY_ORDER, STRATEGY_META, formatSweepParam } from "../lib/strategyStyle";
import { MarkerShape } from "./StrategyMarker";
import { formatInt } from "../lib/format";

/**
 * The thesis-question chart: the full parameter sweep for the selected
 * (advertiser, budget) cell, clicks won against the strategy's own
 * sweep parameter, with the best-of-sweep point marked as an
 * unambiguous peak. Each strategy gets its own panel (own x-axis) because
 * the three parameters are not the same quantity or unit (a flat fen
 * bid, a fen range pair, and a CTR multiplier) -- plotting them on one
 * shared numeric axis would misrepresent them as comparable when they
 * are not; `meta.sweep_parameters` names, units, and orders each axis
 * explicitly rather than this component guessing or hardcoding text.
 * All three panels share one y-axis domain (max clicks in this cell) so
 * the *heights* of the peaks remain directly, visually comparable across
 * strategies -- the actual point of this chart.
 */
export default function SweepChart({ cell, meta }) {
  const maxClicks = Math.max(
    ...STRATEGY_ORDER.flatMap((strategy) => cell.strategies[strategy].map((p) => p.clicks_won))
  );
  const yDomain = [0, Math.ceil(maxClicks * 1.15)];

  return (
    <section aria-label="Budget sweep: full parameter curve" className="flex flex-col gap-2">
      <div>
        <h2 className="text-base font-semibold text-slate-900">
          Clicks won across each strategy's own parameter sweep
        </h2>
        <p className="text-xs text-slate-500">
          One panel per strategy, own x-axis (parameters are different quantities — not comparable on one shared
          axis). The filled, ringed marker is the best-of-sweep point used in the table and spend chart above.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {STRATEGY_ORDER.map((strategy) => (
          <SweepPanel
            key={strategy}
            strategy={strategy}
            points={cell.strategies[strategy]}
            paramMeta={meta.sweep_parameters[strategy]}
            yDomain={yDomain}
          />
        ))}
      </div>
    </section>
  );
}

function SweepPanel({ strategy, points, paramMeta, yDomain }) {
  const styleMeta = STRATEGY_META[strategy];
  const labelField = paramMeta.label_field;
  const data = points.map((p) => ({
    label: p.selected_param[labelField],
    clicks_won: p.clicks_won,
    is_best_by_clicks: p.is_best_by_clicks,
    selected_param: p.selected_param,
  }));

  return (
    <div className="rounded border border-slate-200 bg-white p-2">
      <div className="mb-1 flex items-center gap-1.5 px-1 text-sm font-medium text-slate-800">
        <svg width="20" height="12" aria-hidden="true">
          <MarkerShape shape={styleMeta.marker} color={styleMeta.color} size={4} x={10} y={6} />
        </svg>
        {styleMeta.label}
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 18, right: 12, bottom: 24, left: 8 }}>
            <CartesianGrid stroke="#e2e8f0" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10 }}
              interval={0}
              label={{
                value: `${paramMeta.parameter_name} (${paramMeta.parameter_units})`,
                position: "insideBottom",
                offset: -18,
                fontSize: 10,
              }}
            />
            <YAxis
              domain={yDomain}
              width={36}
              tick={{ fontSize: 10 }}
              tickFormatter={formatInt}
              label={{ value: "Clicks won", angle: -90, position: "insideLeft", fontSize: 10 }}
            />
            <Tooltip content={<SweepTooltip strategy={strategy} />} />
            <Line
              dataKey="clicks_won"
              stroke={styleMeta.color}
              strokeWidth={2}
              strokeDasharray={styleMeta.dash}
              isAnimationActive={false}
              dot={(props) => <SweepDot {...props} color={styleMeta.color} shape={styleMeta.marker} />}
              activeDot={{ r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SweepDot({ cx, cy, payload, color, shape, key }) {
  if (cx === undefined || cy === undefined) return null;
  if (!payload.is_best_by_clicks) {
    return <MarkerShape key={key} shape={shape} color={color} size={3.5} x={cx} y={cy} />;
  }
  // The peak: a visibly larger, ringed marker plus a text label, so it
  // reads as "the peak" at a glance rather than blending into the line.
  return (
    <g key={key}>
      <circle cx={cx} cy={cy} r={9} fill="none" stroke={color} strokeWidth={1.5} opacity={0.55} />
      <MarkerShape shape={shape} color={color} size={5.5} x={cx} y={cy} />
      <text x={cx} y={cy - 13} textAnchor="middle" fontSize={9} fontWeight={700} fill={color}>
        peak
      </text>
    </g>
  );
}

function SweepTooltip({ active, payload, strategy }) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0].payload;
  return (
    <div className="rounded border border-slate-300 bg-white px-3 py-2 text-xs shadow-sm">
      <div className="font-semibold text-slate-800">{formatSweepParam(strategy, point.selected_param)}</div>
      <div className="mt-0.5 text-slate-600">
        Clicks won: <span className="font-medium tabular-nums text-slate-900">{point.clicks_won}</span>
      </div>
      {point.is_best_by_clicks && <div className="mt-0.5 font-medium text-amber-700">Best-of-sweep peak</div>}
    </div>
  );
}
