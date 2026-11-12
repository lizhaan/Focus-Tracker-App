from attention_tracker.focus.metrics import Segment
from attention_tracker.models import FocusState


def test_chart_set_data_and_clear(qapp):
    from attention_tracker.ui.charts import FocusChart

    chart = FocusChart()
    assert chart.bar_count == 0 and chart.segment_count == 0
    segments = [Segment(0, 10, FocusState.FOCUSED), Segment(10, 13, FocusState.NOT_FOCUSED),
                Segment(13, 17, FocusState.LOST_FOCUS), Segment(17, 25, FocusState.FOCUSED)]
    chart.set_data([1.0, 0.6, None], bin_s=10, segments=segments, end=25)
    assert chart.bar_count == 2  # порожній бін (перерва) не малюється
    assert chart.segment_count == 4
    chart.grab()
    chart.clear()
    assert chart.bar_count == 0 and chart.segment_count == 0


def test_chart_handles_empty_session(qapp):
    from attention_tracker.ui.charts import FocusChart

    chart = FocusChart()
    chart.set_data([], bin_s=10, segments=[], end=0)
    assert chart.bar_count == 0
    chart.grab()
