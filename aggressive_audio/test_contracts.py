"""Tests target label errors, split leakage and per-file fusion behavior."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from augment import AugmentConfig, augment, transform, overlay_voice
from contracts import eer, fuse, load_manifest


class Contracts(unittest.TestCase):
    def test_mean_retains_presence_gates(self):
        self.assertAlmostEqual(fuse(0.9, 0.1, 1, 1), 0.5)
        self.assertAlmostEqual(fuse(0.9, 1, 1, 0), 0.45)
        self.assertAlmostEqual(fuse(0.9, 0.1, 1, 1, "max"), 0.9)
        self.assertEqual(fuse(0.3, 0.8, 0.7, 0.2), fuse(0.8, 0.3, 0.2, 0.7))
        with self.assertRaises(ValueError):
            fuse(float("nan"), 0, 1, 1)

    def test_eer_ties_follow_roc_thresholds(self):
        self.assertEqual(eer([0, 1, 0, 1], [0.5] * 4), 0.5)
        self.assertEqual(eer([0, 1], [0.1, 0.9]), 0)
        self.assertEqual(eer([0, 1], [0.9, 0.1]), 1)

    def test_source_group_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "a.wav").write_bytes(b"external-test-fixture")
            (base / "b.wav").write_bytes(b"external-test-fixture-2")
            row = {"id": "a", "path": str(base / "a.wav"), "source_id": "one-source", "split": "train",
                   "label_fake": 0, "domain": "music", "generator": "human", "source_url": "https://example.org",
                   "license": "test fixture", "origin": "public_external"}
            other = dict(row, id="b", path=str(base / "b.wav"), split="dev", label_fake=1)
            manifest = base / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n" + json.dumps(other))
            with self.assertRaisesRegex(ValueError, "Cross-split leakage"):
                load_manifest(manifest)

    def test_unknown_generation_label_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "a.wav").touch()
            row = {"id": "a", "path": str(base / "a.wav"), "source_id": "a", "split": "train", "label_fake": 1,
                   "domain": "speech", "generator": "codec-roundtrip", "source_url": "https://example.org",
                   "license": "fixture", "origin": "public_external", "label_status": "ambiguous"}
            manifest = base / "manifest.jsonl"
            manifest.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                load_manifest(manifest)

    def test_augmentations_preserve_shape_and_provenance(self):
        x = (0.2 * np.sin(2 * np.pi * 440 * np.arange(64000) / 16000)).astype(np.float32)
        for name in AugmentConfig().transforms:
            y, info = transform(x, name, np.random.default_rng(2))
            self.assertEqual(y.shape, x.shape, name)
            self.assertTrue(np.isfinite(y).all(), name)
            self.assertEqual(info["label_policy"], "preserve_source", name)
        a, trace_a = augment(x, np.random.default_rng(8))
        b, trace_b = augment(x, np.random.default_rng(8))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(trace_a, trace_b)
        _, info = overlay_voice(x, x, np.random.default_rng(0))
        self.assertEqual(info["music_label_policy"], "preserve_music_source")


if __name__ == "__main__":
    unittest.main()
