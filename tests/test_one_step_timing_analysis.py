import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "official_verl" / "analyze_one_step_timing.py"
SPEC = importlib.util.spec_from_file_location("one_step_timing", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OneStepTimingAnalysisTest(unittest.TestCase):
    def test_parses_ansi_prefixed_metric_line(self):
        line = (
            "\x1b[36m(Runner)\x1b[0m step:7 - timing_s/generate_async:12.5 - "
            "timing_s/sync_rollout_weights:30.0 - timing_s/ref:14.0 - "
            "timing_s/update_actor:40.0 - timing_s/step:80.0 - perf/throughput:20.0"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.log"
            path.write_text(line + "\n", encoding="utf-8")
            report = MODULE.analyze(path)
        self.assertEqual(report["steps"], 1)
        self.assertEqual(report["first_step"], 7)
        self.assertEqual(report["inclusive_timing_summary_seconds"]["timing_s/step"]["p50"], 80.0)
        self.assertEqual(report["mean_inclusive_stage_sum_minus_step_seconds"], 16.5)
