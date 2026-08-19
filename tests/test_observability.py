from unittest.mock import patch
import unittest

from mini_verl.observability import CudaMemoryMonitor, GpuUtilizationMonitor, _parse_nvidia_smi_sample


class CudaMemoryMonitorTest(unittest.TestCase):
    def test_cpu_measurement_is_a_portable_no_op(self):
        monitor = CudaMemoryMonitor("cpu")
        monitor.start()
        self.assertIsNone(monitor.stop())

    def test_rejects_invalid_lifecycle(self):
        monitor = CudaMemoryMonitor("cpu")
        with self.assertRaisesRegex(RuntimeError, "not started"):
            monitor.stop()
        monitor.start()
        with self.assertRaisesRegex(RuntimeError, "already running"):
            monitor.start()
        monitor.stop()


class GpuUtilizationMonitorTest(unittest.TestCase):
    def test_parses_nvidia_smi_row(self):
        self.assertEqual(_parse_nvidia_smi_sample("37, 1024\n"), (37, 1024))
        with self.assertRaisesRegex(RuntimeError, "non-numeric"):
            _parse_nvidia_smi_sample("N/A, 1024\n")
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            _parse_nvidia_smi_sample("37\n")

    def test_rejects_invalid_config_or_lifecycle(self):
        with self.assertRaisesRegex(ValueError, "gpu_index"):
            GpuUtilizationMonitor(-1)
        with self.assertRaisesRegex(ValueError, "interval_seconds"):
            GpuUtilizationMonitor(0, interval_seconds=0)
        monitor = GpuUtilizationMonitor(0)
        with self.assertRaisesRegex(RuntimeError, "not started"):
            monitor.stop()

    def test_collects_stats_from_the_injected_query(self):
        monitor = GpuUtilizationMonitor(0, interval_seconds=10)
        with patch.object(monitor, "_query_once", side_effect=[(10, 20), (30, 40)]):
            monitor.start()
            monitor._stop_event.wait(0.001)
            stats = monitor.stop()
        self.assertGreaterEqual(stats.sample_count, 1)
        self.assertGreaterEqual(stats.mean_utilization_percent, 10.0)
        self.assertGreaterEqual(stats.max_utilization_percent, 10)
        self.assertGreaterEqual(stats.max_memory_used_mib, 20)


if __name__ == "__main__":
    unittest.main()
