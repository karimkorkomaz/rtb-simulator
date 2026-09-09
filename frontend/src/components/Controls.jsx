import { formatFen } from "../lib/format";

/** Advertiser + budget-level selectors. These two values drive every
 * other region on the page -- nothing else holds independent state. */
export default function Controls({
  meta,
  cellsByKey,
  advertiser,
  budgetFraction,
  onAdvertiserChange,
  onBudgetChange,
}) {
  return (
    <section aria-label="Controls" className="flex flex-col gap-4 sm:flex-row sm:items-start sm:gap-10">
      <fieldset className="flex flex-col gap-2">
        <legend className="text-xs font-semibold text-slate-500">Advertiser</legend>
        <div className="flex flex-wrap gap-1.5" role="group">
          {meta.advertisers.map((adv) => {
            const active = adv === advertiser;
            return (
              <button
                key={adv}
                type="button"
                aria-pressed={active}
                onClick={() => onAdvertiserChange(adv)}
                className={
                  "rounded border px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 " +
                  (active
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-300 bg-white text-slate-700 hover:border-slate-400")
                }
              >
                {adv}
              </button>
            );
          })}
        </div>
      </fieldset>

      <fieldset className="flex flex-col gap-2">
        <legend className="text-xs font-semibold text-slate-500">
          Budget level (fraction of this advertiser's own historical spend)
        </legend>
        <div className="flex flex-wrap gap-1.5" role="group">
          {meta.budget_fractions.map((frac) => {
            const active = Math.abs(frac - budgetFraction) < 1e-9;
            const cell = cellsByKey.get(`${advertiser}|${frac}`);
            const label = meta.budget_labels[String(frac)] ?? String(frac);
            return (
              <button
                key={frac}
                type="button"
                aria-pressed={active}
                onClick={() => onBudgetChange(frac)}
                className={
                  "flex flex-col items-start rounded border px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 " +
                  (active
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-300 bg-white text-slate-700 hover:border-slate-400")
                }
              >
                <span>{label}</span>
                <span className={"text-[11px] font-normal " + (active ? "text-slate-300" : "text-slate-500")}>
                  {cell ? formatFen(cell.budget_fen) : "—"}
                </span>
              </button>
            );
          })}
        </div>
      </fieldset>
    </section>
  );
}
