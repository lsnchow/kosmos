from plumb.policies.judge import (
    JudgeAttempt, JudgeSampleReport, JudgeSamplingConfig, RubricSample,
    _aggregate_sample_reports,
)


def test_even_consensus_keeps_integer_observable_milestone():
    reports = []
    for i, progress in enumerate([1, 2, 3, 4]):
        sample = RubricSample("intact", "none_visible", progress, "not_met", (15,), "Incomplete at end.")
        attempt = JudgeAttempt(i, i, 0, None, sample, None, None, None)
        reports.append(JudgeSampleReport(i, i, (attempt,)))
    sample = RubricSample("uncertain", "uncertain", None, "uncertain", (15,), "Occluded.")
    reports.append(JudgeSampleReport(4, 4, (JudgeAttempt(4, 4, 0, None, sample, None, None, None),)))
    binary, progress, status, reason, votes = _aggregate_sample_reports(reports, JudgeSamplingConfig())
    assert (binary, progress, status, reason, votes) == (False, 2, "evaluable", None, 4)
    assert isinstance(progress, int)
