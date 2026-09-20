import { isRecord, type DemoJudgment } from "./api";

/** Presentation-only completion estimate. Never changes the persisted strict verdict. */
export function demoHeuristic(judgment?: DemoJudgment | null) {
  if (judgment?.status !== "abstained") return undefined;
  const samples = judgment.result?.report?.raw_judge_samples;
  if (!Array.isArray(samples)) return undefined;
  let met = 0;
  let notMet = 0;
  for (const sample of samples.filter(isRecord)) {
    const attempts = Array.isArray(sample.attempts) ? sample.attempts.filter(isRecord) : [];
    const final = attempts.at(-1);
    if (!final || final.failure_reason || !isRecord(final.parsed)) continue;
    const vote = final.parsed;
    if (vote.completion_evidence === "met" && vote.progress === 5) met++;
    if (vote.completion_evidence === "not_met" && typeof vote.progress === "number" && Number.isInteger(vote.progress) && vote.progress >= 0 && vote.progress < 5) notMet++;
  }
  // Fixed five-sample protocol: never turn missing or malformed output into a vote.
  if (samples.length !== 5) return undefined;
  const label = met >= 3 ? "Likely completed" : notMet >= 3 ? "Likely not completed" : "Insufficient agreement";
  return { label, votes: Math.max(met, notMet), total: samples.length };
}
