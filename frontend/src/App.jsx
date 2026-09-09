import { useMemo, useState } from "react";
import simulationData from "./simulationData.json";
import Controls from "./components/Controls";
import StrategyTable from "./components/StrategyTable";
import SpendTrajectoryChart from "./components/SpendTrajectoryChart";
import SweepChart from "./components/SweepChart";

const { meta, cells } = simulationData;

// Default view: advertiser 1458 at budget 1/8 -- the same
// (advertiser, budget) worked example used in
// docs/analysis/simulation-results.md S4.1, so the first screen an
// examiner sees matches a number they can cross-check directly against
// the thesis document.
const DEFAULT_ADVERTISER = meta.advertisers[0];
const DEFAULT_BUDGET_FRACTION = meta.budget_fractions.find((f) => Math.abs(f - 0.125) < 1e-9) ?? meta.budget_fractions[0];

export default function App() {
  const cellsByKey = useMemo(() => {
    const map = new Map();
    for (const cell of cells) {
      map.set(`${cell.advertiser}|${cell.budget_fraction}`, cell);
    }
    return map;
  }, []);

  const [advertiser, setAdvertiser] = useState(DEFAULT_ADVERTISER);
  const [budgetFraction, setBudgetFraction] = useState(DEFAULT_BUDGET_FRACTION);

  const cell = cellsByKey.get(`${advertiser}|${budgetFraction}`);

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-8 px-4 py-8 sm:px-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-bold text-slate-900">Real-time bidding: three baseline strategies, replayed</h1>
        <p className="max-w-3xl text-sm text-slate-600">
          Second-price auction replay against the iPinYou season-2 test day ({meta.date_range?.[0]}), one advertiser
          at a time, under a hard budget constraint. Every strategy competes over the exact same won-auction pool —
          more wins never means more inventory, only a different split of it.
        </p>
      </header>

      <Controls
        meta={meta}
        cellsByKey={cellsByKey}
        advertiser={advertiser}
        budgetFraction={budgetFraction}
        onAdvertiserChange={setAdvertiser}
        onBudgetChange={setBudgetFraction}
      />

      {cell ? (
        <>
          <StrategyTable cell={cell} meta={meta} />
          <SpendTrajectoryChart cell={cell} />
          <SweepChart cell={cell} meta={meta} />
        </>
      ) : (
        <p className="text-sm text-red-700">
          No data for advertiser {advertiser} at budget {budgetFraction} — this combination is missing from the
          export.
        </p>
      )}

      <footer className="border-t border-slate-200 pt-4 text-xs text-slate-500">
        Units: spend, budget, and effective CPC are all RMB fen (1/100 CNY yuan) — the raw iPinYou collection unit;
        no currency conversion is applied. Win rate uses the full-pool denominator (comparable across strategies).
        Source: {meta.n_raw_sweep_runs} simulation runs across {meta.n_advertisers} advertisers ×{" "}
        {meta.n_budgets} budget levels × {meta.n_strategies} strategies.
      </footer>
    </div>
  );
}
