// Single source of truth for how a strategy is drawn/labelled, reused by
// the table, the spend-trajectory chart, and the sweep chart so a
// strategy carries the same colour, line style, and marker shape
// everywhere on the page (task requirement: "consistent colour per
// strategy across BOTH charts and the table"). Colour is never the only
// distinguishing channel -- each strategy also gets its own dash pattern
// and marker shape, so the page still reads correctly in greyscale.
import { formatFenShort } from "./format";

export const STRATEGY_ORDER = ["constant", "random", "linear_ctr"];

export const STRATEGY_META = {
  constant: {
    label: "Constant",
    shortLabel: "Constant",
    color: "#0f172a", // slate-900 -- darkest, solid line
    dash: undefined, // solid
    marker: "circle",
  },
  random: {
    label: "Random",
    shortLabel: "Random",
    color: "#b45309", // amber-700
    dash: "8 5", // dashed
    marker: "square",
  },
  linear_ctr: {
    label: "Linear-in-CTR",
    shortLabel: "Linear-CTR",
    color: "#1d4ed8", // blue-700
    dash: "2 5", // dotted
    marker: "diamond",
  },
};

/** Human-readable description of a sweep point's parameter, specific to
 * the strategy's own parameter shape (flat amount / range pair /
 * CTR-multiplier base) -- units always stated, per the task's data
 * rules. */
export function formatSweepParam(strategy, selectedParam) {
  if (!selectedParam) return "—";
  if (strategy === "constant") {
    return `${selectedParam.quantile_label} · flat bid ${formatFenShort(selectedParam.amount)} fen`;
  }
  if (strategy === "linear_ctr") {
    return `${selectedParam.quantile_label} · base ${formatFenShort(selectedParam.base)} fen per unit CTR`;
  }
  if (strategy === "random") {
    return `${selectedParam.range_label} · [${formatFenShort(selectedParam.low)}, ${formatFenShort(
      selectedParam.high
    )}) fen`;
  }
  return JSON.stringify(selectedParam);
}
