import { useCallback, useRef } from "react";
import { cn } from "../lib/utils";

export type TabItem<T extends string = string> = {
  id: T;
  label: string;
  /** Optional short line under the label, for a nav that has to explain itself. */
  hint?: string;
};

/**
 * Mirage's two-tab control, generalised and given the keyboard behaviour a
 * tablist owes its users.
 *
 * Theirs was a pair of plain buttons with a class name: no role, no
 * `aria-selected`, and every tab in the tab order. Here the widget is one tab
 * stop and the arrow keys move between tabs, which is the WAI-ARIA authoring
 * practice for a tablist and is also simply faster to drive on stage.
 *
 * The selected tab is marked three ways — `aria-selected`, a brighter ink and
 * the accent underline — so selection never rests on the 2 px line alone.
 */
export function TabSelector<T extends string>({
  tabs,
  value,
  onChange,
  label,
  idPrefix,
  centered,
  display,
  className,
}: {
  tabs: TabItem<T>[];
  value: T;
  onChange: (next: T) => void;
  /** Accessible name for the tablist itself. */
  label: string;
  /**
   * Prefix for the generated `id`/`aria-controls` pair. The caller renders the
   * matching panel with `id={`${idPrefix}-panel-${value}`}`.
   */
  idPrefix: string;
  centered?: boolean;
  /** Use the display face, as Mirage's landing nav does. */
  display?: boolean;
  className?: string;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  const focusTab = useCallback((index: number) => {
    const buttons = containerRef.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]');
    buttons?.[index]?.focus();
  }, []);

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const current = tabs.findIndex((tab) => tab.id === value);
    if (current < 0) return;
    let next: number | undefined;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (current + 1) % tabs.length;
    else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (current - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    onChange(tabs[next].id);
    focusTab(next);
  };

  return (
    <div
      ref={containerRef}
      role="tablist"
      aria-label={label}
      className={cn("tablist", centered && "tablist-center", className)}
      onKeyDown={onKeyDown}
    >
      {tabs.map((tab) => {
        const selected = tab.id === value;
        return (
          <button
            key={tab.id}
            type="button"
            role="tab"
            id={`${idPrefix}-tab-${tab.id}`}
            aria-selected={selected}
            // Only the selected tab has a panel in the DOM, and `aria-controls`
            // must reference an element that exists.
            aria-controls={selected ? `${idPrefix}-panel-${tab.id}` : undefined}
            // One tab stop for the whole widget; arrow keys move inside it.
            tabIndex={selected ? 0 : -1}
            className={cn("tab", display && "tab-display")}
            onClick={() => onChange(tab.id)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
