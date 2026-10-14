"""Byte-level fixtures for development-only text reconstruction."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rebuild_text_from_vl", PROJECT / ".dev/tools/rebuild_text_from_vl.py")
rebuild = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rebuild)


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / ".dev/tmp", prefix="rebuild-test-")
        self.root = Path(self.temp.name)
        self.old_root = rebuild.ROOT
        rebuild.ROOT = self.root
        self.layout_dir = self.root / rebuild.LAYOUT_DIRECTORY
        self.layout_dir.mkdir(parents=True)
        (self.root / "configs").mkdir()
        (self.root / "models/text").mkdir(parents=True)
        (self.root / "models/vl").mkdir()
        self.a = b"unused-0123456789-tail"
        self.b = b"prefix-ABCDEFGHIJ-tail"
        self.source_a = self.root / "models/vl/a.safetensors"
        self.source_b = self.root / "models/vl/b.safetensors"
        self.source_a.write_bytes(self.a)
        self.source_b.write_bytes(self.b)
        header_json = b'{"__metadata__": {"format": "pt"}}   '
        self.header = len(header_json).to_bytes(8, "little") + header_json
        (self.layout_dir / "target.header.bin").write_bytes(self.header)
        self.expected = self.header + self.b[9:14] + self.a[8:12]
        self.target = self.root / "models/text/target.safetensors"
        self.target_entry = self.entry("target.safetensors", self.expected)
        self.manifest = {"version": 1, "models": [
            {"id": "jev_27b", "local_directory": "models/text", "files": [self.target_entry]},
            {"id": "jev_27b_vl", "local_directory": "models/vl", "files": [self.entry("a.safetensors", self.a), self.entry("b.safetensors", self.b)]},
        ]}
        self.output = {"file": "target.safetensors", "header_file": "target.header.bin",
                       "size": len(self.expected), "sha256": hashlib.sha256(self.expected).hexdigest(),
                       "ranges": [
                           {"target_offset": len(self.header) + 5, "file": "a.safetensors", "offset": 8, "nbytes": 4},
                           {"target_offset": len(self.header), "file": "b.safetensors", "offset": 9, "nbytes": 5},
                       ]}
        self.save()

    @staticmethod
    def entry(path, data):
        return {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def save(self):
        (self.root / "configs/weights-manifest.json").write_text(json.dumps(self.manifest))
        (self.layout_dir / "tensor-layout-map.json").write_text(json.dumps({"outputs": [self.output]}))

    def tearDown(self):
        rebuild.ROOT = self.old_root
        self.temp.cleanup()

    def test_exact_header_sorted_offsets_and_concat(self):
        plan = rebuild.load_plan(["target.safetensors"])
        self.assertTrue(rebuild.rebuild(plan[0], rebuild.VerifiedHashes()))
        self.assertEqual(self.target.read_bytes(), self.expected)
        self.assertEqual(self.target.read_bytes()[:len(self.header)], self.header)
        self.assertFalse(self.target.with_name(self.target.name + ".rebuild.part").exists())

    def test_wrong_target_hash_preserves_existing_final_and_part(self):
        self.output["sha256"] = self.target_entry["sha256"] = "0" * 64
        self.save()
        self.target.write_bytes(b"old target")
        with self.assertRaisesRegex(RuntimeError, "Reconstructed SHA256 mismatch"):
            rebuild.rebuild(rebuild.load_plan()[0], rebuild.VerifiedHashes())
        self.assertEqual(self.target.read_bytes(), b"old target")
        self.assertEqual(self.target.with_name(self.target.name + ".rebuild.part").read_bytes(), self.expected)

    def test_missing_source_waits_without_partial(self):
        self.source_a.unlink()
        self.assertFalse(rebuild.rebuild(rebuild.load_plan()[0], rebuild.VerifiedHashes()))
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_name(self.target.name + ".rebuild.part").exists())

    def test_invalid_source_hash_prevents_assembly(self):
        self.source_a.write_bytes(b"x" * len(self.a))
        with self.assertRaisesRegex(RuntimeError, "Source fails official"):
            rebuild.rebuild(rebuild.load_plan()[0], rebuild.VerifiedHashes())
        self.assertFalse(self.target.with_name(self.target.name + ".rebuild.part").exists())

    def test_verified_existing_target_skips_missing_sources(self):
        self.target.write_bytes(self.expected)
        self.source_a.unlink()
        self.source_b.unlink()
        self.assertTrue(rebuild.rebuild(rebuild.load_plan()[0], rebuild.VerifiedHashes()))
        self.assertEqual(self.target.read_bytes(), self.expected)

    def test_stat_cache_reuses_success_and_invalidates_changed_file(self):
        cache = rebuild.VerifiedHashes()
        source_entry = self.entry("a.safetensors", self.a)
        original_hash = hashlib.sha256
        with patch.object(rebuild.hashlib, "sha256", wraps=original_hash) as sha256:
            with rebuild.open_regular(self.source_a) as source:
                self.assertTrue(cache.matches(source, source_entry))
            with rebuild.open_regular(self.source_a) as source:
                self.assertTrue(cache.matches(source, source_entry))
            self.assertEqual(sha256.call_count, 1)
            self.source_a.write_bytes(b"x" * len(self.a))
            with rebuild.open_regular(self.source_a) as source:
                self.assertFalse(cache.matches(source, source_entry))
            self.assertEqual(sha256.call_count, 2)

    def test_map_must_match_official_manifest(self):
        self.output["sha256"] = "0" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            rebuild.load_plan()

    def test_gap_and_out_of_bounds_ranges_rejected(self):
        self.output["ranges"][0]["target_offset"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "Noncontiguous"):
            rebuild.load_plan()
        self.output["ranges"][0]["target_offset"] -= 1
        self.output["ranges"][0]["offset"] = 10000
        self.save()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            rebuild.load_plan()

    def test_symlinks_and_traversal_rejected(self):
        with self.assertRaises(ValueError):
            rebuild.safe_path("../escape")
        self.source_a.unlink()
        self.source_a.symlink_to(self.source_b)
        with self.assertRaisesRegex(ValueError, "symlink"):
            rebuild.load_plan()

    def test_unknown_only_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown --only"):
            rebuild.load_plan(["missing.safetensors"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
