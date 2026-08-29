#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

pytest.importorskip("sarvamai")

from pipecat.services.sarvam.stt import SarvamSTTService, language_to_sarvam_language
from pipecat.transcriptions.language import Language


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.HI_IN, "hi-IN"),
        (Language.UR_IN, "ur-IN"),
        (Language.KOK_IN, "kok-IN"),
        (Language.MAI_IN, "mai-IN"),
        (Language.SD_IN, "sd-IN"),
    ],
)
def test_language_to_sarvam_language_maps_enum_values(language, expected):
    assert language_to_sarvam_language(language) == expected


@pytest.mark.parametrize("language_code", ["ne-IN", "sat-IN"])
def test_get_language_string_passes_through_string_values(language_code):
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = language_code

    assert service._get_language_string() == language_code


def test_get_language_string_resolves_enum_via_mapping():
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = Language.HI_IN

    assert service._get_language_string() == "hi-IN"


def test_get_language_string_returns_model_default_when_unset():
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = None

    assert service._get_language_string() == service._config.default_language
    assert service._config.default_language == "unknown"


class TestFinalTranscriptsAreMarkedFinalized:
    """Sarvam emits one data message per utterance, and it is always final.

    The turn stop strategies keep a safety net sized to the service's p99
    time-to-final-segment -- ``SARVAM_TTFS_P99`` (1.17s) less the VAD's
    ``stop_secs`` (0.2s), so ~0.97s -- for the case where speech has ended but
    the transcript has not arrived. ``TranscriptionFrame.finalized`` is how an
    STT says "nothing more is coming" and collapses that wait.

    Sarvam had the answer and was not saying it: there is no interim path in
    this service, ``on_utterance_end`` fires immediately before the frame is
    built, and ``_handle_transcription`` is already passed ``is_final=True``.
    Every turn paid the full net for a transcript already in hand.
    """

    @staticmethod
    def _data_message(transcript: str):
        class _Data:
            def __init__(self):
                self.transcript = transcript
                self.language_code = "hi-IN"

        class _Message:
            def __init__(self):
                self.type = "data"
                self.data = _Data()

            def dict(self):
                return {"type": "data", "transcript": transcript}

        return _Message()

    async def _push_one(self, transcript: str):
        from unittest.mock import AsyncMock

        service = SarvamSTTService(api_key="test-key")
        service.push_frame = AsyncMock()
        service.stop_processing_metrics = AsyncMock()
        service._call_event_handler = AsyncMock()
        await service._handle_message(self._data_message(transcript))
        return [c.args[0] for c in service.push_frame.await_args_list]

    @pytest.mark.asyncio
    async def test_the_transcript_frame_is_marked_finalized(self):
        frames = await self._push_one("नमस्ते")

        assert len(frames) == 1
        assert frames[0].text == "नमस्ते"
        assert frames[0].finalized is True, (
            "Without this the turn waits out SARVAM_TTFS_P99 minus the VAD's "
            "stop_secs on every turn, for a final transcript it already has."
        )

    @pytest.mark.asyncio
    async def test_a_blank_transcript_pushes_nothing(self):
        """Guard the branch the flag lives in, so the fix cannot move out of it."""
        assert await self._push_one("   ") == []
