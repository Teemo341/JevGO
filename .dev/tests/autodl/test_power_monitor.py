"""CPU-only tests for fail-closed GPU power observations and idle timing."""

import copy
import math
import unittest

import power_monitor


def gpu(index=0, power=36.43):
    return {
        "index": index, "uuid": f"GPU-test-{index}", "name": "RTX PRO 6000",
        "power_w": power, "pstate": "P8", "utilization_percent": 0,
    }


def sample(*gpus):
    return {
        "identity": {"hostname": "autodl-container-test", "boot_id": "boot-test", "pid1_start_ticks": 1234},
        "gpus": list(gpus) if gpus else [gpu()],
        "compute_processes": [],
    }


class PowerCsvTest(unittest.TestCase):
    def parse(self, row="0, GPU-test-0, RTX PRO 6000, 36.43, P8, 0"):
        return power_monitor.parse_power_csv(row)

    def test_fractional_power_and_multiple_gpu_rows(self):
        result = self.parse("0, GPU-first, RTX PRO 6000, 36.43, P8, 0\n1, GPU-second, A100, 145.1, P0, 99\n")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["power_w"], 36.43)
        self.assertEqual(result[1]["utilization_percent"], 99)

    def test_csv_quoted_name_supported(self):
        self.assertEqual(self.parse('0, GPU-first, "Vendor, GPU", 36.43, P8, 0')[0]["name"], "Vendor, GPU")

    def test_zero_power_supported(self):
        self.assertEqual(self.parse("0, GPU-test, GPU, 0, P8, 0")[0]["power_w"], 0)

    def test_empty_output_rejected(self):
        for text in ("", "\n", "  ", None):
            with self.subTest(text=text), self.assertRaises(power_monitor.PowerSampleError):
                self.parse(text)

    def test_invalid_power_rejected(self):
        for value in ("N/A", "[N/A]", "[Not Supported]", "NaN", "nan", "inf", "-inf", "-0.1", "", "36 W", "1e9999"):
            with self.subTest(value=value), self.assertRaises(power_monitor.PowerSampleError):
                self.parse(f"0, GPU-test, GPU, {value}, P8, 0")

    def test_invalid_utilization_rejected(self):
        for value in ("N/A", "NaN", "inf", "-1", "101", "0%", ""):
            with self.subTest(value=value), self.assertRaises(power_monitor.PowerSampleError):
                self.parse(f"0, GPU-test, GPU, 36, P8, {value}")

    def test_missing_extra_columns_and_internal_blank_rows_rejected(self):
        for text in (
            "0, GPU-test, GPU, 36, P8",
            "0, GPU-test, GPU, 36, P8, 0, EXTRA",
            "0, GPU-test, GPU, 36, P8, 0\n\n1, GPU-second, GPU, 36, P8, 0",
        ):
            with self.subTest(text=text), self.assertRaises(power_monitor.PowerSampleError):
                self.parse(text)

    def test_invalid_identity_fields_rejected(self):
        for text in (
            "-1, GPU-test, GPU, 36, P8, 0",
            "0.5, GPU-test, GPU, 36, P8, 0",
            "0, N/A, GPU, 36, P8, 0",
            "0, GPU-test, N/A, 36, P8, 0",
            "0, GPU-test, GPU, 36, N/A, 0",
            "0, GPU-test, GPU, 36, P16, 0",
        ):
            with self.subTest(text=text), self.assertRaises(power_monitor.PowerSampleError):
                self.parse(text)

    def test_duplicate_uuid_or_index_rejected(self):
        for second in ("1, GPU-test, GPU, 36, P8, 0", "0, GPU-second, GPU, 36, P8, 0"):
            with self.subTest(second=second), self.assertRaises(power_monitor.PowerSampleError):
                self.parse("0, GPU-test, GPU, 36, P8, 0\n" + second)

    def test_malformed_quoted_csv_rejected(self):
        with self.assertRaises(power_monitor.PowerSampleError):
            self.parse('0, GPU-test, "unclosed name, 36, P8, 0')


class IdleWindowTest(unittest.TestCase):
    def window(self, **kwargs):
        return power_monitor.IdleWindow(45, **kwargs)

    def observe_until(self, window, stop, observed=None):
        observed = sample() if observed is None else observed
        for timestamp in range(0, stop + 1, 30):
            result = window.observe(observed, timestamp)
        return result

    def test_idle_requires_full_1800_seconds(self):
        window = self.window()
        first = window.observe(sample(), 0)
        self.assertFalse(first["eligible"])
        self.assertEqual(first["idle_seconds"], 0)
        self.observe_until(window, 1770)
        almost = window.observe(sample(), 1799)
        self.assertEqual(almost["idle_seconds"], 1799)
        self.assertFalse(almost["eligible"])
        complete = window.observe(sample(), 1800)
        self.assertEqual(complete["idle_seconds"], 1800)
        self.assertTrue(complete["eligible"])
        self.assertEqual(complete["reason"], "idle_timeout")

    def test_nonzero_monotonic_origin_and_fractional_seconds(self):
        window = self.window(idle_seconds=1.5, max_gap_seconds=2)
        window.observe(sample(), 4000.25)
        self.assertFalse(window.observe(sample(), 4001.749)["eligible"])
        self.assertTrue(window.observe(sample(), 4001.75)["eligible"])

    def test_equal_threshold_is_idle(self):
        window = self.window(idle_seconds=1)
        window.observe(sample(gpu(power=45)), 0)
        self.assertTrue(window.observe(sample(gpu(power=45)), 1)["eligible"])

    def test_short_power_peak_restarts_full_window(self):
        window = self.window()
        self.observe_until(window, 1770)
        peak = window.observe(sample(gpu(power=45.01)), 1799)
        self.assertEqual(peak["reason"], "power_above_threshold")
        self.assertEqual(peak["idle_seconds"], 0)
        self.assertFalse(window.observe(sample(), 1800)["eligible"])
        self.assertEqual(window.observe(sample(), 1830)["idle_seconds"], 30)

    def test_any_busy_gpu_prevents_idle(self):
        window = self.window(idle_seconds=1)
        window.observe(sample(gpu(), gpu(1)), 0)
        result = window.observe(sample(gpu(), gpu(1, power=46)), 1)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["max_power_w"], 46)
        self.assertEqual(result["gpu_count"], 2)

    def test_gpu_process_policy_is_left_to_controller(self):
        window = self.window(idle_seconds=1)
        observed = sample()
        observed["compute_processes"] = [{"pid": 10}]
        window.observe(observed, 0)
        self.assertTrue(window.observe(observed, 1)["eligible"])

    def test_explicit_error_reset_discards_unobserved_time(self):
        window = self.window()
        self.observe_until(window, 1770)
        reset = window.reset("ssh_error")
        self.assertEqual(reset["reason"], "ssh_error")
        self.assertFalse(reset["eligible"])
        self.assertEqual(window.observe(sample(), 1830)["idle_seconds"], 0)

    def test_long_sampling_gap_restarts_with_current_sample(self):
        window = self.window()
        self.observe_until(window, 1770)
        result = window.observe(sample(), 1861)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "sample_gap")
        self.assertEqual(result["idle_seconds"], 0)
        self.assertEqual(window.observe(sample(), 1891)["idle_seconds"], 30)

    def test_exact_maximum_gap_is_allowed(self):
        window = self.window(idle_seconds=90)
        window.observe(sample(), 0)
        self.assertTrue(window.observe(sample(), 90)["eligible"])

    def test_clock_rollback_restarts_window(self):
        window = self.window(idle_seconds=60)
        window.observe(sample(), 1000)
        window.observe(sample(), 1030)
        result = window.observe(sample(), 1010)
        self.assertEqual(result["reason"], "clock_reversed")
        self.assertFalse(result["eligible"])
        self.assertEqual(window.observe(sample(), 1040)["idle_seconds"], 30)

    def test_repeated_timestamp_does_not_accumulate(self):
        window = self.window(idle_seconds=1)
        for _ in range(2000):
            result = window.observe(sample(), 1000)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["idle_seconds"], 0)

    def test_each_instance_identity_component_resets(self):
        for key, value in (("hostname", "autodl-other"), ("boot_id", "new-boot"), ("pid1_start_ticks", 9999)):
            with self.subTest(key=key):
                window = self.window(idle_seconds=1)
                observed = sample()
                window.observe(observed, 0)
                changed = copy.deepcopy(observed)
                changed["identity"][key] = value
                result = window.observe(changed, 1)
                self.assertFalse(result["eligible"])
                self.assertEqual(result["reason"], "identity_changed")

    def test_reset_keeps_instance_change_guard(self):
        window = self.window()
        window.observe(sample(), 0)
        window.reset()
        changed = sample()
        changed["identity"]["boot_id"] = "changed"
        self.assertEqual(window.observe(changed, 4000)["reason"], "identity_changed")

    def test_changed_gpu_set_resets(self):
        window = self.window(idle_seconds=1)
        window.observe(sample(gpu()), 0)
        result = window.observe(sample(gpu(), gpu(1)), 1)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "gpu_set_changed")

    def test_gpu_order_does_not_reset(self):
        window = self.window(idle_seconds=1)
        window.observe(sample(gpu(), gpu(1)), 0)
        self.assertTrue(window.observe(sample(gpu(1), gpu()), 1)["eligible"])

    def test_invalid_sample_resets_before_raising(self):
        for change in (
            lambda item: item.update(gpus=[]),
            lambda item: item.pop("identity"),
            lambda item: item["identity"].update(boot_id=""),
            lambda item: item["identity"].update(pid1_start_ticks=True),
            lambda item: item["gpus"][0].update(power_w=math.nan),
            lambda item: item["gpus"][0].pop("pstate"),
        ):
            with self.subTest(change=change):
                window = self.window(idle_seconds=1)
                window.observe(sample(), 0)
                bad = sample()
                change(bad)
                with self.assertRaises(power_monitor.PowerSampleError):
                    window.observe(bad, 1)
                self.assertEqual(window.observe(sample(), 2)["idle_seconds"], 0)

    def test_nonfinite_and_boolean_clock_values_rejected(self):
        for now in (math.nan, math.inf, -math.inf, -1, True):
            with self.subTest(now=now), self.assertRaises(power_monitor.PowerSampleError):
                self.window().observe(sample(), now)

    def test_invalid_constructor_values_rejected(self):
        for kwargs in (
            {"threshold_w": math.nan}, {"threshold_w": math.inf}, {"threshold_w": -1},
            {"threshold_w": True}, {"idle_seconds": 0}, {"idle_seconds": -1},
            {"idle_seconds": math.inf}, {"max_gap_seconds": 0}, {"max_gap_seconds": math.nan},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(power_monitor.PowerSampleError):
                power_monitor.IdleWindow(**({"threshold_w": 45} | kwargs))


if __name__ == "__main__":
    unittest.main()
