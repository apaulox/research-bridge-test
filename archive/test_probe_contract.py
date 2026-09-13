"""Dependency-free checks before a trained checkpoint is available."""
import json
from pathlib import Path
import tempfile
import unittest

from musicdet_probe import load_configuration, window_starts


class ProbeContractTests(unittest.TestCase):
    def test_missing_weights_cannot_silently_use_random_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_configuration(Path(directory))

    def test_reject_wrong_dataset_and_missing_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'anti-spoofing_feat_model.pt').touch()
            config = dict(model='spec-nf', task='sonics', K=2, L=1, R=5.,
                          audio_len=64600, only_real=True)
            (folder / 'args.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_configuration(folder)
            config['task'] = 'fakemusiccaps'
            del config['only_real']
            (folder / 'args.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_configuration(folder)

    def test_all_long_audio_samples_covered_without_duplicate_windows(self):
        size = 64600
        for length in [size, size + 1, 2 * size, 2 * size + 1, 60 * 16000]:
            starts = window_starts(length, size, 'sliding-mean')
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1] + size, length)
            self.assertEqual(len(starts), len(set(starts)))
            for left, right in zip(starts, starts[1:]):
                self.assertLessEqual(right, left + size)

    def test_center_and_four_second_boundaries(self):
        self.assertEqual(window_starts(64000, 64600, 'center'), [0])
        self.assertEqual(window_starts(160000, 64600, 'center'), [47700])
        with self.assertRaises(ValueError):
            window_starts(0, 64600, 'center')


if __name__ == '__main__':
    unittest.main()
