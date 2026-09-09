// Formatting helpers shared by the table and both charts. Kept tiny and
// dependency-free -- this is presentation-only string formatting, no
// value transformation (no rounding decisions beyond display digits;
// all real rounding already happened in the export script).

const FEN_FORMATTER = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/** Whole-number RMB fen with thousands separators, unit suffixed. */
export function formatFen(value) {
  if (value === null || value === undefined) return "—";
  return `${FEN_FORMATTER.format(Math.round(value))} fen`;
}

/** Same as formatFen but without the unit suffix, for axis ticks where
 * the unit is already in the axis title. */
export function formatFenShort(value) {
  if (value === null || value === undefined) return "—";
  return FEN_FORMATTER.format(Math.round(value));
}

/** Plain integer with thousands separators, no unit -- for counts (e.g.
 * clicks won) where formatFenShort's currency semantics would be
 * misleading even though the numeric formatting is identical. */
export function formatInt(value) {
  if (value === null || value === undefined) return "—";
  return FEN_FORMATTER.format(Math.round(value));
}

export function formatPercent(fraction, digits = 1) {
  if (fraction === null || fraction === undefined) return "—";
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** iPinYou timestamps are naive local (Beijing) time, e.g.
 * "2013-06-12T14:32:07.029000" -- no timezone suffix. Extracting the
 * HH:MM:SS substring directly (not via `Date`) avoids the browser
 * silently reinterpreting it in the viewer's own timezone. */
export function formatTimeOfDay(isoLikeString) {
  if (!isoLikeString) return null;
  const match = /T(\d{2}:\d{2}:\d{2})/.exec(isoLikeString);
  return match ? match[1] : isoLikeString;
}

export function formatHour(hour) {
  return `${String(hour).padStart(2, "0")}:00`;
}

/** Renders an exhaustion record explicitly -- never a bare blank/"null"
 * for the "never exhausted" case (see simulationData.json meta). */
export function formatExhaustion(exhaustion) {
  if (!exhaustion) return "—";
  if (!exhaustion.exhausted) return "Not exhausted (budget unspent at day end)";
  const time = formatTimeOfDay(exhaustion.timestamp);
  return `Exhausted at ${time}`;
}

export function formatEffectiveCpc(value) {
  if (value === null || value === undefined) return "N/A (0 clicks)";
  return formatFen(value);
}
