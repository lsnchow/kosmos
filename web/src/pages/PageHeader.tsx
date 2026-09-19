/**
 * The one `h1` a page is allowed.
 *
 * The single-scroll console had two — the wordmark and the masthead — which is
 * an outline error a screen reader surfaces as two document titles. Splitting
 * into pages fixes it structurally: each page states what it is, once.
 */
import type { ReactNode } from "react";

export type PageHeaderProps = {
  eyebrow?: ReactNode;
  title: string;
  lede?: ReactNode;
  actions?: ReactNode;
};

export function PageHeader({ eyebrow, title, lede, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-header-main">
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1 className="text-balance">{title}</h1>
        {lede && <p className="page-lede text-pretty">{lede}</p>}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </header>
  );
}
