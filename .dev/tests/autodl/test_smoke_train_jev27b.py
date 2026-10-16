"""CPU integration check of the same training/checkpoint path used on AutoDL."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import smoke_train_jev27b as smoke


class ModelProfileAndReportPathTest(unittest.TestCase):
    def setUp(self):
        temporary_root = smoke.ROOT / ".dev/tmp/autodl-training-test"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = self.enterContext(tempfile.TemporaryDirectory(dir=temporary_root))
        self.root = Path(temporary)
        self.project = self.root / "project"
        self.fast = self.root / "fast"
        self.shared = self.root / "nas/models"
        for directory in (self.project, self.fast, self.shared / "JEV-9B", self.shared / "JEV-27B"):
            directory.mkdir(parents=True)
        self.enterContext(patch.object(smoke, "ROOT", self.project))
        self.enterContext(patch.object(smoke, "FAST_STORAGE", self.fast))
        self.enterContext(patch.object(smoke, "SHARED_MODELS", self.shared))

    def test_model_profile_requires_matching_shared_model_and_architecture(self):
        models = self.project / "models"
        models.symlink_to(self.shared, target_is_directory=True)
        nine = smoke.MODEL_PROFILES["jev_9b"]
        large = smoke.MODEL_PROFILES["jev_27b"]
        self.assertEqual(smoke.model_directory(nine, models / "JEV-9B"), self.shared / "JEV-9B")
        with self.assertRaisesRegex(RuntimeError, "original JEV-9B"):
            smoke.model_directory(nine, models / "JEV-27B")
        nine_config = SimpleNamespace(model_type="qwen3_5_text", hidden_size=4096, num_hidden_layers=32)
        large_config = SimpleNamespace(model_type="qwen3_5_text", hidden_size=5120, num_hidden_layers=64)
        smoke.check_architecture(nine, nine_config)
        smoke.check_architecture(large, large_config)
        with self.assertRaisesRegex(RuntimeError, "Unexpected JEV-9B"):
            smoke.check_architecture(nine, large_config)
        with self.assertRaisesRegex(RuntimeError, "Unexpected JEV-27B"):
            smoke.check_architecture(large, nine_config)

    def test_capacity_gate_distinguishes_profiles_and_rejects_insufficient_free_memory(self):
        nine = smoke.MODEL_PROFILES["jev_9b"]
        large = smoke.MODEL_PROFILES["jev_27b"]
        smoke.check_capacity(nine, 24 * smoke.GIB, 22 * smoke.GIB)
        smoke.check_capacity(large, 96 * smoke.GIB, 70 * smoke.GIB)
        for total, free in ((24, 21), (23, 22)):
            with self.subTest(total=total, free=free), self.assertRaisesRegex(RuntimeError, "Estimated gate"):
                smoke.check_capacity(nine, total * smoke.GIB, free * smoke.GIB)
        with self.assertRaisesRegex(RuntimeError, "JEV-27B"):
            smoke.check_capacity(large, 48 * smoke.GIB, 46 * smoke.GIB)

    def test_project_report_creates_real_parents_and_refuses_overwrite(self):
        report = self.project / ".dev/artifacts/cycle/training.json"
        self.assertEqual(smoke.report_path(report), report)
        self.assertTrue(report.parent.is_dir())
        report.write_text("saved result")
        with self.assertRaisesRegex(RuntimeError, "new file"):
            smoke.report_path(report)
        self.assertEqual(report.read_text(), "saved result")

    def test_project_report_rejects_nas_symlink_and_dangling_file_link(self):
        artifacts = self.project / ".dev/artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "checkpoints").symlink_to(self.shared, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "must not be a symlink"):
            smoke.report_path(artifacts / "checkpoints/training.json")
        self.assertFalse((self.shared / "training.json").exists())
        (artifacts / "training.json").symlink_to(self.root / "missing")
        with self.assertRaisesRegex(RuntimeError, "new file"):
            smoke.report_path(artifacts / "training.json")

    def test_fast_report_remains_supported_but_escape_is_rejected(self):
        report = self.fast / "runs/cycle/training.json"
        self.assertEqual(smoke.report_path(report), report)
        with self.assertRaisesRegex(RuntimeError, "under project .dev/artifacts"):
            smoke.report_path(self.root / "outside/training.json")
        with self.assertRaisesRegex(RuntimeError, "traversal"):
            smoke.report_path(self.project / ".dev/artifacts/../training.json")


class TrainingRoundTripTest(unittest.TestCase):
    def test_linear_and_full_attention_lora_checkpoint_round_trip(self):
        self.enterContext(patch.dict(os.environ, {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_HUB_KERNELS": "0"}))
        import torch
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        torch.manual_seed(20261016)
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["linear_attention", "full_attention"], pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
        )
        base = Qwen3_5ForCausalLM(config).cpu()
        originals = [(param, param.detach().clone()) for param in base.parameters()]
        batches = []
        for _ in range(3):
            ids = torch.randint(1, 128, (1, 16))
            batches.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()})
        temporary_root = smoke.ROOT / ".dev/tmp/autodl-training-test"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            checkpoint = Path(temporary) / "checkpoint"
            save_checks = []

            def before_save():
                self.assertFalse(checkpoint.exists())
                save_checks.append(True)

            result = smoke.train_and_reload(base, batches, checkpoint, steps=3, rank=4,
                                            before_save=before_save)
            self.assertEqual(save_checks, [True])
            self.assertEqual(result["target_module_counts"], dict.fromkeys(smoke.TARGETS, 1))
            self.assertGreater(result["changed_trainable_tensors"], 0)
            self.assertTrue(result["adapter_reload"]["passed"])
            self.assertTrue(result["adapter_reload"]["optimizer_reloaded"])
            self.assertEqual(len(result["losses"]), 3)
            self.assertEqual(result["trainable_dtype"], "float32")
            self.assertEqual(result["reloaded_trainable_dtype"], "float32")
            self.assertFalse(result["step_timing"]["cuda_synchronized"])
            self.assertTrue(all(item["step_seconds"] > 0 for item in result["losses"]))
            files = {item["path"]: item for item in result["checkpoint_files"]}
            self.assertIn("adapter/adapter_model.safetensors", files)
            self.assertIn("optimizer.pt", files)
            self.assertTrue(all(torch.equal(param.detach(), before) for param, before in originals),
                            "Frozen base tensors must remain unchanged across training and adapter reload")


class FastKernelProbeTest(unittest.TestCase):
    @staticmethod
    def decorated_provider(package):
        def implementation(value):
            return value * 2

        implementation.__module__ = package
        original = implementation
        is_new_implementation = True

        def decorated(value):
            if is_new_implementation:
                return implementation(value)
            raise AssertionError("Unexpected reference branch")

        return decorated, original

    def test_counts_selected_provider_calls_and_restores_provider(self):
        import torch

        conv, conv_original = self.decorated_provider("causal_conv1d.causal_conv1d_interface")
        delta, delta_original = self.decorated_provider("fla.ops.gated_delta_rule.chunk")
        module = SimpleNamespace(causal_conv1d_fn=conv, torch_chunk_gated_delta_rule=delta)
        probe = smoke.FastKernelProbe(module)
        self.addCleanup(probe.close)
        with self.assertRaisesRegex(RuntimeError, "must execute"):
            probe.assert_used()
        self.assertEqual(module.causal_conv1d_fn(3), conv_original(3))
        self.assertEqual(module.torch_chunk_gated_delta_rule(4), delta_original(4))
        with torch.no_grad():
            module.causal_conv1d_fn(5)
        probe.assert_used()
        report = probe.snapshot()["providers"]
        self.assertEqual(report["causal_conv1d_fn"]["calls"], 2)
        self.assertEqual(report["causal_conv1d_fn"]["completed_calls"], 2)
        self.assertEqual(report["causal_conv1d_fn"]["grad_enabled_calls"], 1)
        self.assertEqual(report["torch_chunk_gated_delta_rule"]["calls"], 1)
        probe.close()
        module.causal_conv1d_fn(8)
        self.assertEqual(probe.snapshot()["providers"], report)

    def test_rejects_reference_fallback_and_restores_partial_instrumentation(self):
        import inspect

        conv, conv_original = self.decorated_provider("causal_conv1d.causal_conv1d_interface")
        fallback, _ = self.decorated_provider("transformers.models.qwen3_5.modeling_qwen3_5")
        module = SimpleNamespace(causal_conv1d_fn=conv, torch_chunk_gated_delta_rule=fallback)
        with self.assertRaisesRegex(RuntimeError, "expected an installed fla kernel"):
            smoke.FastKernelProbe(module)
        self.assertIs(inspect.getclosurevars(conv).nonlocals["implementation"], conv_original)


if __name__ == "__main__":
    unittest.main()
