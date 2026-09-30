import pytest

from biosppy.signals import ecg


def test_compare_segmentation_counts_default_false_positives():
    result = ecg.compare_segmentation(
        reference=[100, 300, 500],
        test=[100, 200, 300, 400, 500],
        sampling_rate=1000,
        tol=0.05,
    )

    assert result["TP"] == 3
    assert result["FP"] == 2
    assert result["performance"] == 1.0
    assert result["acc"] == pytest.approx(3 / 5)
    assert result["err"] == pytest.approx(2 / 5)


def test_compare_segmentation_can_filter_short_rr_false_positives():
    result = ecg.compare_segmentation(
        reference=[100, 300, 500],
        test=[100, 120, 300, 500],
        sampling_rate=1000,
        minRR=0.05,
        tol=0.01,
    )

    assert result["TP"] == 3
    assert result["FP"] == 0
