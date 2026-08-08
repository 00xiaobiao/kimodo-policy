from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bucket_rerecord_by_reward import (
    RerecordPair,
    build_move_plan,
    parse_summary_pairs,
    parse_worker_log_pair,
)


class BucketRerecordByRewardTests(unittest.TestCase):
    def test_parse_summary_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary_path = tmp_path / "rerecord_conversion.log"
            summary_path.write_text(
                "\n".join(
                    [
                        "[2/111] rerecording /tmp/dataset/twist2/zk/foo.npz",
                        "  output_dir=/tmp/dataset/twist2_multicam_rerecord/zk",
                        "  log=/tmp/dataset/twist2_multicam_rerecord/rerecord_logs/foo.log",
                        (
                            "  success -> /tmp/dataset/twist2_multicam_rerecord/zk/bar.npz"
                            " final_reward=0.0000 max_reward=0.0200 any_success=true"
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            pairs = parse_summary_pairs(summary_path, "twist2")
            pair = pairs[str(Path("/tmp/dataset/twist2/zk/foo.npz").resolve())]
            self.assertEqual(pair.source_npz, Path("/tmp/dataset/twist2/zk/foo.npz").resolve())
            self.assertEqual(pair.rerecorded_npz, Path("/tmp/dataset/twist2_multicam_rerecord/zk/bar.npz").resolve())
            self.assertEqual(pair.final_reward, 0.0)
            self.assertEqual(pair.max_reward, 0.02)
            self.assertTrue(pair.any_success)

    def test_parse_summary_pairs_legacy_format_falls_back_to_final_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary_path = tmp_path / "rerecord_conversion.log"
            summary_path.write_text(
                "\n".join(
                    [
                        "[2/111] rerecording /tmp/dataset/twist2/zk/foo.npz",
                        "  output_dir=/tmp/dataset/twist2_multicam_rerecord/zk",
                        "  log=/tmp/dataset/twist2_multicam_rerecord/rerecord_logs/foo.log",
                        "  success -> /tmp/dataset/twist2_multicam_rerecord/zk/bar.npz final_reward=0.0000",
                    ]
                ),
                encoding="utf-8",
            )

            pairs = parse_summary_pairs(summary_path, "twist2")
            pair = pairs[str(Path("/tmp/dataset/twist2/zk/foo.npz").resolve())]
            self.assertEqual(pair.final_reward, 0.0)
            self.assertEqual(pair.max_reward, 0.0)
            self.assertFalse(pair.any_success)

    def test_parse_worker_log_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_root = tmp_path / "twist2"
            rerecord_root = tmp_path / "twist2_multicam_rerecord"
            log_path = rerecord_root / "rerecord_logs" / "foo.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "\n".join(
                    [
                        "COMMAND: python sim_main.py --replay_file "
                        f"{source_root / 'zk' / 'foo.npz'} --recording_save_dir "
                        f"{rerecord_root / '.tmp_rerecord' / 'foo_tmp'}",
                        "[TWIST2ActionProvider] Final rerecord reward=0.0000",
                        "[TWIST2ActionProvider] Max rerecord reward=0.0200 any_success=true",
                        "[RecordingManager] Renaming to final file: bar.npz",
                    ]
                ),
                encoding="utf-8",
            )

            pair = parse_worker_log_pair(
                log_path,
                backend="twist2",
                source_root=source_root,
                rerecord_root=rerecord_root,
            )
            self.assertIsNotNone(pair)
            assert pair is not None
            self.assertEqual(pair.source_npz, (source_root / "zk" / "foo.npz").resolve())
            self.assertEqual(pair.rerecorded_npz, (rerecord_root / "zk" / "bar.npz").resolve())
            self.assertEqual(pair.final_reward, 0.0)
            self.assertEqual(pair.max_reward, 0.02)
            self.assertTrue(pair.any_success)

    def test_build_move_plan_uses_max_reward_and_preserves_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_npz = tmp_path / "sonic" / "zk" / "foo.npz"
            source_video = tmp_path / "sonic" / "zk" / "videos" / "foo_front_rgb.mp4"
            rerecord_npz = tmp_path / "sonic_multicam_rerecord" / "zk" / "bar.npz"
            rerecord_video = tmp_path / "sonic_multicam_rerecord" / "zk" / "videos" / "bar_front_rgb.mp4"

            for path in [source_npz, source_video, rerecord_npz, rerecord_video]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            pair = RerecordPair(
                backend="sonic",
                source_npz=source_npz.resolve(),
                rerecorded_npz=rerecord_npz.resolve(),
                final_reward=0.0,
                max_reward=0.02,
                any_success=True,
                origin="test",
                priority=100,
            )

            moves, matched_pairs = build_move_plan([pair], target_reward=0.02, reward_tol=1e-8)
            destinations = {move.destination for move in moves}

            self.assertEqual(len(matched_pairs), 1)
            self.assertIn((tmp_path / "sonic_bad" / "zk" / "foo.npz").resolve(), destinations)
            self.assertIn(
                (tmp_path / "sonic_bad" / "zk" / "videos" / "foo_front_rgb.mp4").resolve(),
                destinations,
            )
            self.assertIn(
                (tmp_path / "sonic_multicam_rerecord_bad" / "zk" / "bar.npz").resolve(),
                destinations,
            )
            self.assertIn(
                (tmp_path / "sonic_multicam_rerecord_bad" / "zk" / "videos" / "bar_front_rgb.mp4").resolve(),
                destinations,
            )


if __name__ == "__main__":
    unittest.main()
