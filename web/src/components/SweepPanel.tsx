import { SlidersHorizontal, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import type { SweepPoint, SweepResponse } from "../lib/api";
import {
  formatCount,
  formatGpuSeconds,
  formatRateAsPercent,
  formatUsd,
  pickString,
} from "../lib/format";
import { cn } from "../lib/utils";
import { EmptyState, Note, Panel, SourceChip, StatusPill } from "./Primitives";

export function pointLabel(point: SweepPoint | undefined, index: number): string {
  return (
    pickString(point?.operating_point_id, point?.label, point?.id) ?? `Operating point ${index + 1}`
  );
}

function toleranceState(point: SweepPoint | undefined) {
  if (point?.tolerances_met === true) return { status: "pass", label: "tolerances met" } as const;
  if (point?.tolerances_met === false) return { status: "fail", label: "tolerances not met" } as const;
  return { status: "unknown", label: "tolerance result not reported" } as const;
}

/**
 * Cost–fidelity slider.
 *
 * It reads precomputed sweep points and issues no request on drag — dragging is
 * pure state, so the ranking cannot appear to change because a fresh generation
 * happened to come out differently. The fixed task horizon and coverage sit
 * beside every point, because a shorter task is not a cheaper equivalent
 * evaluation and must not be able to masquerade as a saving.
 */
export function SweepPanel({ sweeps }: { sweeps: SweepResponse }) {
  const points = sweeps.points ?? [];
  const [selected, setSelected] = useState(0);
  useEffect(() => {
    setSelected((current) => Math.min(current, Math.max(points.length - 1, 0)));
  }, [points.length]);

  const point = points[selected];
  const tolerance = toleranceState(point);
  const firstFailure = points.findIndex((candidate) => candidate.tolerances_met === false);

  return (
    <Panel
      title="Cost–fidelity operating point"
      className="sweep-panel"
      action={<SourceChip>{pickString(sweeps.status) ?? "status not reported"}</SourceChip>}
    >
      {points.length === 0 ? (
        <EmptyState icon={<SlidersHorizontal aria-hidden="true" className="size-5" />}>
          {pickString(sweeps.reason) ??
            "No persisted sweep points are available."}{" "}
          The slider will not generate, estimate, or imply a cheaper setting that was never measured.
        </EmptyState>
      ) : (
        <>
          <div className="range-row">
            <label htmlFor="cost-sweep">
              Operating point{" "}
              <span className="tabular-nums">
                {selected + 1} of {points.length}
              </span>
            </label>
            <input
              id="cost-sweep"
              type="range"
              min={0}
              max={points.length - 1}
              step={1}
              value={selected}
              onChange={(event) => setSelected(Number(event.target.value))}
              list="cost-sweep-ticks"
            />
            <datalist id="cost-sweep-ticks">
              {points.map((candidate, index) => (
                <option key={pointLabel(candidate, index)} value={index} label={String(index + 1)} />
              ))}
            </datalist>
          </div>

          {/* A tick per point: filled where tolerances held, hatched where they broke. */}
          <ol className="sweep-track" aria-hidden="true">
            {points.map((candidate, index) => (
              <li
                key={pointLabel(candidate, index)}
                className={cn(
                  "sweep-tick",
                  candidate.tolerances_met === false && "sweep-tick-fail",
                  candidate.tolerances_met === true && "sweep-tick-pass",
                  index === selected && "sweep-tick-selected",
                )}
              />
            ))}
          </ol>
          {firstFailure >= 0 && (
            <p className="sweep-break" role="note">
              <TriangleAlert aria-hidden="true" className="size-4" />
              Preregistered tolerances first fail at point {firstFailure + 1} of {points.length} (
              {pointLabel(points[firstFailure], firstFailure)}). Cheaper than that point, the ranking is not
              supported.
            </p>
          )}

          <div className="sweep-readout">
            <div>
              <span>Operating point</span>
              <strong>{pointLabel(point, selected)}</strong>
            </div>
            <div>
              <span>Estimated cost</span>
              <strong className="tabular-nums">{formatUsd(point?.estimated_usd)}</strong>
            </div>
            <div>
              <span>GPU-seconds</span>
              <strong className="tabular-nums">{formatGpuSeconds(point?.gpu_seconds)}</strong>
            </div>
            <div>
              <span>Fixed task horizon</span>
              <strong className="tabular-nums">
                {point?.fixed_task_horizon === undefined || point?.fixed_task_horizon === null
                  ? "not reported"
                  : `${point.fixed_task_horizon} control ticks`}
              </strong>
            </div>
            <div>
              <span>Coverage</span>
              <strong className="tabular-nums">{formatRateAsPercent(point?.coverage)}</strong>
            </div>
            <div>
              <span>Pairwise order agreement</span>
              <strong className="tabular-nums">
                {formatRateAsPercent(point?.pairwise_order_agreement)}
              </strong>
            </div>
            <div>
              <span>Resolution</span>
              <strong>{point?.resolution === undefined ? "not reported" : String(point.resolution)}</strong>
            </div>
            <div>
              <span>Denoise steps</span>
              <strong className="tabular-nums">{formatCount(point?.denoise_steps, "not reported")}</strong>
            </div>
            <div>
              <span>Chunk partition</span>
              <strong>
                {point?.chunk_partition === undefined ? "not reported" : String(point.chunk_partition)}
              </strong>
            </div>
            <div>
              <span>Batch size</span>
              <strong className="tabular-nums">{formatCount(point?.batch_size, "not reported")}</strong>
            </div>
            <div>
              <span>Tolerances</span>
              <strong>
                <StatusPill status={tolerance.status}>{tolerance.label}</StatusPill>
              </strong>
            </div>
            <div>
              <span>Qualification</span>
              <strong>
                <StatusPill status={pickString(point?.qualification) ?? "unknown"}>
                  {pickString(point?.qualification) ?? "not reported"}
                </StatusPill>
              </strong>
            </div>
          </div>

          {point?.tolerances_met === false && (
            <p className="inline-error" role="alert">
              This point failed its preregistered per-cell error, coverage or pairwise-order tolerances. It is
              shown because failed settings are kept, not because it is selectable for a qualified claim.
            </p>
          )}
        </>
      )}
      <Note summary="How cost per point is measured">
        {points.length > 0 && pickString(sweeps.reason) ? `${sweeps.reason} ` : ""}
        Every point keeps its fixed task horizon beside its cost, so a shorter task cannot read as a saving.
        Cost is an allocation-ledger estimate until billing reconciliation. Only a held-out-confirmed
        configuration supports a qualified claim; the rest are retained evidence.
      </Note>
    </Panel>
  );
}
