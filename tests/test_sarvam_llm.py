#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the chunk sequence SarvamLLMService replays.

Sarvam is requested without streaming, so this adapter rebuilds the chunks a
stream would have sent. The consuming loop in
``BaseOpenAILLMService._process_context`` reads them the way it reads a real
provider's stream, and the failure mode when the shapes disagree is silent:
tool calls are dropped without an error or a log.
"""

import inspect
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai import base_llm
from pipecat.services.sarvam.llm import SarvamLLMService


def _service() -> SarvamLLMService:
    service = SarvamLLMService(
        api_key="test-key",
        settings=SarvamLLMService.Settings(model="sarvam-105b"),
    )
    service._client = MagicMock()
    return service


def _completion(content=None, tool_calls=(), finish_reason="stop"):
    """A Sarvam non-streaming completion, shaped like the OpenAI SDK's."""
    calls = []
    for position, (name, arguments) in enumerate(tool_calls):
        call = MagicMock()
        call.model_dump.return_value = {
            "id": f"call_{position}",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        calls.append(call)

    message = types.SimpleNamespace(role="assistant", content=content, tool_calls=calls)
    choice = types.SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(
        id="resp-1",
        created=1,
        model="sarvam-105b",
        choices=[choice],
        usage=None,
        system_fingerprint=None,
    )


async def _chunks(response):
    service = _service()
    service._client.chat.completions.create = AsyncMock(return_value=response)
    context = LLMContext(messages=[{"role": "user", "content": "hello"}])
    stream = await service.get_chat_completions(context)
    return [chunk async for chunk in stream]


def _collect_tool_calls(chunks):
    """Mirror of the tool-call assembly in ``_process_context``.

    Kept deliberately literal rather than tidied: its value is that it reads
    exactly one tool call per chunk and separates calls by watching ``index``
    change, which is the contract these chunks have to satisfy.
    """
    collected = []
    name, arguments, call_id, expected_index = "", "", "", 0

    for chunk in chunks:
        if not chunk.choices or not chunk.choices[0].delta:
            continue
        tool_calls = chunk.choices[0].delta.tool_calls
        if not tool_calls:
            continue

        tool_call = tool_calls[0]
        if tool_call.index != expected_index:
            collected.append((name, arguments or "{}", call_id))
            name, arguments, call_id = "", "", ""
            expected_index += 1
        if tool_call.function and tool_call.function.name:
            name += tool_call.function.name
            call_id = tool_call.id
        if tool_call.function and tool_call.function.arguments:
            arguments += tool_call.function.arguments

    if name:
        collected.append((name, arguments or "{}", call_id))
    return collected


def _text(chunks):
    """What the loop would push downstream: content outside a tool-call chunk."""
    return "".join(
        chunk.choices[0].delta.content
        for chunk in chunks
        if chunk.choices
        and chunk.choices[0].delta
        and not chunk.choices[0].delta.tool_calls
        and chunk.choices[0].delta.content
    )


class TestEveryToolCallSurvives:
    @pytest.mark.asyncio
    async def test_a_single_tool_call_arrives(self):
        chunks = await _chunks(
            _completion(tool_calls=[("get_slots", '{"day":"monday"}')], finish_reason="tool_calls")
        )

        assert _collect_tool_calls(chunks) == [("get_slots", '{"day":"monday"}', "call_0")]

    @pytest.mark.asyncio
    async def test_three_tool_calls_all_arrive(self):
        """The one that fails when they share a chunk.

        Only the first is read, so an agent that decided to do three things
        does one of them and reports no error.
        """
        chunks = await _chunks(
            _completion(
                tool_calls=[
                    ("get_slots", '{"day":"monday"}'),
                    ("book", '{"slot":7}'),
                    ("notify", '{"channel":"sms"}'),
                ],
                finish_reason="tool_calls",
            )
        )

        assert [name for name, _, _ in _collect_tool_calls(chunks)] == [
            "get_slots",
            "book",
            "notify",
        ]

    @pytest.mark.asyncio
    async def test_each_tool_call_keeps_its_own_id_and_arguments(self):
        """Arguments concatenate across chunks, so a mixed-up boundary merges
        two calls' JSON into one unparseable string."""
        chunks = await _chunks(
            _completion(
                tool_calls=[("first", '{"a":1}'), ("second", '{"b":2}')],
                finish_reason="tool_calls",
            )
        )

        assert _collect_tool_calls(chunks) == [
            ("first", '{"a":1}', "call_0"),
            ("second", '{"b":2}', "call_1"),
        ]

    @pytest.mark.asyncio
    async def test_indexes_are_sequential_from_zero(self):
        chunks = await _chunks(
            _completion(tool_calls=[("a", "{}"), ("b", "{}")], finish_reason="tool_calls")
        )
        indexes = [
            chunk.choices[0].delta.tool_calls[0].index
            for chunk in chunks
            if chunk.choices[0].delta.tool_calls
        ]

        assert indexes == [0, 1]


class TestTextIsNotLostToAToolCall:
    @pytest.mark.asyncio
    async def test_plain_content_is_pushed(self):
        chunks = await _chunks(_completion(content="Hello there, how can I help?"))

        assert _text(chunks) == "Hello there, how can I help?"

    @pytest.mark.asyncio
    async def test_content_alongside_tool_calls_is_still_pushed(self):
        """Content sharing a chunk with a tool call is never read, because the
        loop reads content only in the ``elif``."""
        chunks = await _chunks(
            _completion(
                content="One moment.",
                tool_calls=[("lookup", "{}")],
                finish_reason="tool_calls",
            )
        )

        assert _text(chunks) == "One moment."
        assert [name for name, _, _ in _collect_tool_calls(chunks)] == ["lookup"]

    @pytest.mark.asyncio
    async def test_whitespace_is_preserved_exactly(self):
        """The reason this service does not stream at all."""
        spoken = "Sure  —  I can do that. One moment, please."
        chunks = await _chunks(_completion(content=spoken))

        assert _text(chunks) == spoken


class TestTheStreamStillCarriesItsMetadata:
    @pytest.mark.asyncio
    async def test_an_empty_reply_still_yields_one_chunk(self):
        """The loop reads the model name off the stream; nothing yielded
        reports nothing."""
        chunks = await _chunks(_completion(content=None))

        assert len(chunks) == 1
        assert chunks[0].model == "sarvam-105b"

    @pytest.mark.asyncio
    async def test_finish_reason_lands_on_the_last_chunk_only(self):
        """Earlier would end the turn before the tool calls behind it are read."""
        chunks = await _chunks(
            _completion(
                content="Checking.",
                tool_calls=[("a", "{}"), ("b", "{}")],
                finish_reason="tool_calls",
            )
        )
        reasons = [chunk.choices[0].finish_reason for chunk in chunks]

        assert reasons[-1] == "tool_calls"
        assert all(reason is None for reason in reasons[:-1])


def test_the_consuming_loop_still_reads_one_tool_call_per_chunk():
    """A guard on the assumption every test above rests on.

    If the base loop learns to read the whole ``delta.tool_calls`` list, the
    one-chunk-per-call split stops being necessary and this file is describing
    a contract that no longer exists.
    """
    source = inspect.getsource(base_llm.BaseOpenAILLMService._process_context)

    assert "chunk.choices[0].delta.tool_calls[0]" in source, (
        "The base loop no longer indexes the first tool call. Re-check whether "
        "SarvamLLMService._as_chunks still needs one chunk per call."
    )
