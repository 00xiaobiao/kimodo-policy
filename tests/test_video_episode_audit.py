import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import av

from data.multisource_dataset import EpisodeRecord, SOURCE_HIW500
from scripts import audit_video_episodes as audit


def _episode() -> EpisodeRecord:
    return EpisodeRecord(
        source=SOURCE_HIW500,
        task_id="task",
        task_name="task",
        instruction="task",
        episode_id="1",
        data_path=Path("episode.parquet"),
        source_length=6,
        source_fps=30.0,
        target_fps=30.0,
        video_path=Path("episode.mp4"),
        video_from_timestamp=0.0,
        first_cut=0,
        sample_count=1,
        metadata={"row_start": 0, "row_end": 6},
    )


def _container_context(*, frames=None, decode_error=None):
    stream = MagicMock()
    stream.time_base = Fraction(1, 30)
    container = MagicMock()
    container.streams.video = [stream]
    if decode_error is None:
        container.decode.return_value = list(frames or [])
    else:
        container.decode.side_effect = decode_error
    context = MagicMock()
    context.__enter__.return_value = container
    context.__exit__.return_value = False
    return context, container


class VideoEpisodeAuditTest(unittest.TestCase):
    def setUp(self):
        audit._initialize_worker(action_chunk=2)

    def test_trainable_interval_decodes_as_valid(self):
        context, container = _container_context(
            frames=[SimpleNamespace(pts=0), SimpleNamespace(pts=4)]
        )
        with patch.object(audit.av, "open", return_value=context):
            result = audit._audit_video_episode((SOURCE_HIW500, _episode()))

        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["decoded_frames"], 2)
        container.seek.assert_called_once_with(0)

    def test_invalid_packet_is_counted_as_video_invalid(self):
        error = av.error.InvalidDataError(
            1094995529, "Invalid data found when processing input"
        )
        context, _ = _container_context(decode_error=error)
        with patch.object(audit.av, "open", return_value=context):
            result = audit._audit_video_episode((SOURCE_HIW500, _episode()))

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["error_type"], "InvalidDataError")
        self.assertEqual(result["video_path"], "episode.mp4")

    def test_video_without_decodable_frames_is_invalid(self):
        context, _ = _container_context(frames=[])
        with patch.object(audit.av, "open", return_value=context):
            result = audit._audit_video_episode((SOURCE_HIW500, _episode()))

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["error_type"], "VideoEpisodeDecodeError")

    def test_unexpected_failure_is_reported_separately(self):
        with patch.object(
            audit, "_scan_episode_video", side_effect=AssertionError("bug")
        ):
            result = audit._audit_video_episode((SOURCE_HIW500, _episode()))

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "AssertionError")


if __name__ == "__main__":
    unittest.main()
