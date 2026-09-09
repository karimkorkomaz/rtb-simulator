import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { STRATEGY_ORDER, STRATEGY_META } from "../lib/strategyStyle";
import { MarkerShape } from "./StrategyMarker";
import { formatFen, formatFenShort, formatHour } from "../lib/format";

/** Cumulative spend over the 24 hourly checkpoints, one line per
 * strategy, for the currently selected cell's `is_best_by_clicks` run.
 * The budget ceiling is drawn as a reference line; a strategy that
 * exhausts its budget shows an explicit marker at the exhaustion hour,
 * and its line is flat (by construction of the source data) from that
 * point on -- both facts are called out in the caption rather than left
 * for the reader to notice unaided. */
export default function SpendTrajectoryChart({ cell }) {
  const bestPoints = Object.fromEntries(
    STRATEGY_ORDER.map((strategy) => [strategy, cell.strategies[strategy].find((p) => p.is_best_by_clicks)])
  );

  const data = Array.from({ length: 24 }, (_, hour) => {
    const row = { hour };
    for (const strategy of STRATEGY_ORDER) {
      row[strategy] = bestPoints[strategy].spend_trajectory_fen[hour];
    }
    return row;
  });

  const exhaustionDots = STRATEGY_ORDER.filter((s) => bestPoints[s].exhaustion.exhausted).map((strategy) => {
    const ts = bestPoints[strategy].exhaustion.timestamp;
    const hour = Number(/T(\d{2}):/.exec(ts)[1]);
    return { strategy, hour, timestamp: ts };
  });

  return (
    <section aria-label="Spend trajectory" className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold text-slate-800">
        Cumulative spend over the test day — best-of-sweep run per strategy
      </h2>
      <div className="h-80 w-full rounded border border-slate-200 bg-white p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
            <CartesianGrid stroke="#e2e8f0" vertical={false} />
            <XAxis
              dataKey="hour"
              tickFormatter={formatHour}
              type="number"
              domain={[0, 23]}
              ticks={[0, 4, 8, 12, 16, 20, 23]}
              label={{ value: "Hour of day (local, 2013-06-12)", position: "insideBottom", offset: -6, fontSize: 12 }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              tickFormatter={formatFenShort}
              width={70}
              tick={{ fontSize: 11 }}
              label={{ value: "Cumulative spend (fen)", angle: -90, position: "insideLeft", fontSize: 12 }}
            />
            <Tooltip content={<SpendTooltip exhaustionByStrategy={bestPoints} />} />
            <Legend
              verticalAlign="top"
              height={28}
              formatter={(value) => STRATEGY_META[value]?.label ?? value}
              wrapperStyle={{ fontSize: 12 }}
            />
            <ReferenceLine
              y={cell.budget_fen}
              stroke="#64748b"
              strokeDasharray="4 4"
              label={{ value: "Budget ceiling", position: "insideTopRight", fontSize: 11, fill: "#475569" }}
            />
            {STRATEGY_ORDER.map((strategy) => {
              const meta = STRATEGY_META[strategy];
              return (
                <Line
                  key={strategy}
                  dataKey={strategy}
                  name={strategy}
                  stroke={meta.color}
                  strokeWidth={2}
                  strokeDasharray={meta.dash}
                  dot={false}
                  isAnimationActive={false}
                  activeDot={{ r: 4 }}
                />
              );
            })}
            {exhaustionDots.map(({ strategy, hour }) => (
              <ReferenceDot
                key={strategy}
                x={hour}
                y={cell.budget_fen}
                r={6}
                fill={STRATEGY_META[strategy].color}
                stroke="#fff"
                strokeWidth={1.5}
                isFront
                ifOverflow="extendDomain"
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-slate-500">
        A line goes flat once that strategy exhausts its budget (marked with a filled dot on the budget ceiling, at
        the exact exhaustion hour) — spend cannot exceed the budget by construction. Lines that never reach the
        ceiling did not exhaust their budget by day end.
      </p>
    </section>
  );
}

function SpendTooltip({ active, payload, label, exhaustionByStrategy }) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded border border-slate-300 bg-white px-3 py-2 text-xs shadow-sm">
      <div className="mb-1 font-semibold text-slate-700">{formatHour(label)}</div>
      {payload.map((entry) => {
        const meta = STRATEGY_META[entry.dataKey];
        const exhausted = exhaustionByStrategy[entry.dataKey]?.exhaustion?.exhausted;
        return (
          <div key={entry.dataKey} className="flex items-center gap-1.5 py-0.5">
            <svg width="10" height="10" aria-hidden="true">
              <MarkerShape shape={meta.marker} color={meta.color} size={4} x={5} y={5} />
            </svg>
            <span className="text-slate-600">{meta.label}:</span>
            <span className="font-medium tabular-nums text-slate-900">{formatFen(entry.value)}</span>
            {exhausted && <span className="text-amber-700">(exhausted)</span>}
          </div>
        );
      })}
    </div>
  );
}
