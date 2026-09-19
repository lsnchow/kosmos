"""Regression for nonzero-mean action normalization; not an identity fixture."""
from plumb.adapters.bridge import ActionNormalizer, NormalizationMethod


def test_meanstd_uses_std_not_std_minus_mean():
    normalizer = ActionNormalizer(
        revision="test-only", method=NormalizationMethod.MEANSTD,
        low=(2.0, -4.0), high=(3.0, 2.0))
    assert normalizer.normalize((5.0, -2.0)) == (1.0, 1.0)
    assert normalizer.denormalize(normalizer.normalize((5.0, -2.0))) == (5.0, -2.0)
