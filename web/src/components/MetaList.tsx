/**
 * A label/value description list, written once.
 *
 * Five panels hand-rolled the same `<dl><div><dt/><dd/></div></dl>` shape: the
 * wall tile's frame counts, the viewer's metadata, the smoke card's metrics,
 * free-play's state readout and the scoreboard's provisional counts.
 *
 * The class name stays a prop rather than being folded into one canonical
 * class. That is deliberate: the five rules differ in ways that are real design
 * decisions, not accidents — one column versus two, a hairline above each row
 * or none, right-aligned figures or not — and `styles.contrast.test.ts` pins
 * four of those `dd` rules individually to keep `var(--font-mono)` and
 * `tabular-nums` in the same block. Collapsing the CSS would mean weakening
 * that guarantee to save four lines, which is the wrong trade.
 *
 * What was actually duplicated is this markup, and now it is not.
 */
import type { ReactNode } from "react";

export type MetaItem = {
  label: string;
  value: ReactNode;
  /** Applied to the `<dd>`, for the panels whose figures are tabular. */
  valueClassName?: string;
};

export type MetaListProps = {
  className: string;
  items: MetaItem[];
};

export function MetaList({ className, items }: MetaListProps) {
  return (
    <dl className={className}>
      {items.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd className={item.valueClassName}>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}
