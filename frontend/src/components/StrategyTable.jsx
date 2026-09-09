import { STRATEGY_ORDER, STRATEGY_META, formatSweepParam } from "../lib/strategyStyle";
import { formatFen, formatPercent, formatExhaustion, formatEffectiveCpc } from "../lib/format";
import { LegendSwatch } from "./StrategyMarker";

/** One row per strategy, the `is_best_by_clicks` point of the currently
 * selected (advertiser, budget) cell -- clicks won, win rate (pool),
 * spend, effective CPC, exhaustion. The post-hoc-selection caveat is
 * shown directly above the table (not a footer) because it governs the
 * two columns an examiner is most likely to read first. */
export default function StrategyTable({ cell, meta }) {
  return (
    <section aria-label="Strategy comparison" className="flex flex-col gap-3">
      <div className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
        <strong className="font-semibold">Upper bound, not a forecast.</strong>{" "}
        {meta.caveat_post_hoc_selection}
      </div>

      <div className="overflow-x-auto rounded border border-slate-200">
        <table className="w-full min-w-[720px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-100 text-left text-xs uppercase tracking-wide text-slate-500">
              <th scope="col" className="px-3 py-2 font-semibold">
                Strategy
              </th>
              <th scope="col" className="px-3 py-2 font-semibold">
                Best-of-sweep parameter
              </th>
              <th scope="col" className="px-3 py-2 text-right font-semibold">
                Clicks won <sup>†</sup>
              </th>
              <th scope="col" className="px-3 py-2 text-right font-semibold">
                Win rate (pool)
              </th>
              <th scope="col" className="px-3 py-2 text-right font-semibold">
                Spend (fen)
              </th>
              <th scope="col" className="px-3 py-2 text-right font-semibold">
                Effective CPC (fen) <sup>†</sup>
              </th>
              <th scope="col" className="px-3 py-2 font-semibold">
                Exhaustion
              </th>
            </tr>
          </thead>
          <tbody>
            {STRATEGY_ORDER.map((strategy) => {
              const styleMeta = STRATEGY_META[strategy];
              const points = cell.strategies[strategy];
              const point = points.find((p) => p.is_best_by_clicks);
              return (
                <tr key={strategy} className="border-b border-slate-100 last:border-0 even:bg-slate-50/60">
                  <th scope="row" className="whitespace-nowrap px-3 py-2 text-left font-medium text-slate-800">
                    <span className="mr-2">
                      <LegendSwatch meta={styleMeta} />
                    </span>
                    {styleMeta.label}
                  </th>
                  <td className="whitespace-nowrap px-3 py-2 text-slate-600">
                    {formatSweepParam(strategy, point.selected_param)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums font-semibold text-slate-900">
                    {point.clicks_won}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-slate-700">
                    {formatPercent(point.win_rate_pool)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-slate-700">
                    {formatFen(point.spend_fen)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-slate-700">
                    {formatEffectiveCpc(point.effective_cpc_fen)}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2 text-slate-700">
                    {formatExhaustion(point.exhaustion)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-slate-500">
        <sup>†</sup> Selected post hoc on this same test day (see note above) — read as an upper bound, not a
        forecast of live performance.
      </p>
    </section>
  );
}
