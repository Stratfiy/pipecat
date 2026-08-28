#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam LLM service implementation using OpenAI-compatible interface."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionChunk

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.adapters.services.open_ai_adapter import is_given as openai_is_given
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import _NotGiven, assert_given, is_given


@dataclass
class SarvamLLMSettings(OpenAILLMSettings):
    """Settings for SarvamLLMService.

    Parameters:
        wiki_grounding: Sarvam wiki grounding toggle.
        reasoning_effort: Reasoning effort level (low, medium, high).
    """

    wiki_grounding: bool | None | _NotGiven = field(default_factory=lambda: _NOT_GIVEN)
    reasoning_effort: Literal["low", "medium", "high"] | None | _NotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )


class SarvamLLMService(OpenAILLMService):
    """A service for interacting with Sarvam's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Sarvam's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Sarvam doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    _SUPPORTED_MODELS = frozenset(
        {"sarvam-30b", "sarvam-30b-16k", "sarvam-105b", "sarvam-105b-32k"}
    )
    Settings = SarvamLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.sarvam.ai/v1",
        settings: Settings | None = None,
        default_headers: Mapping[str, str] | None = None,
        **kwargs,
    ):
        """Initialize Sarvam LLM service.

        Args:
            api_key: Sarvam API key used for both OpenAI auth and Sarvam subscription header.
            base_url: Sarvam OpenAI-compatible base URL.
            settings: Runtime-updatable settings.
            default_headers: Additional HTTP headers to include in requests.
            **kwargs: Additional keyword arguments passed to ``OpenAILLMService``.
        """
        # Initialize only Sarvam-specific defaults; inherited defaults are
        # provided by the OpenAI base service initialization.
        default_settings = self.Settings(
            model="sarvam-30b",
            wiki_grounding=None,
            reasoning_effort=None,
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        model = default_settings.model
        if not isinstance(model, str):
            raise ValueError("Sarvam LLM requires a non-empty model string.")
        self._validate_model(model)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            default_headers=default_headers,
            **kwargs,
        )

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create OpenAI-compatible client for Sarvam API endpoint.

        Ensures Sarvam auth and SDK identification headers are always attached.
        """
        merged_headers = dict(default_headers or {})
        # sdk_headers() carries Pipecat User-Agent and should override caller-provided value.
        merged_headers.update(sdk_headers())
        if api_key:
            merged_headers["api-subscription-key"] = api_key

        logger.debug(f"Creating Sarvam client with API {base_url}")
        return super().create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=merged_headers,
            **kwargs,
        )

    async def get_chat_completions(self, context: LLMContext):
        """Return one complete chunk so Sarvam cannot lose token whitespace.

        Sarvam's OpenAI-compatible streaming endpoint has been observed emitting
        word pieces without their leading whitespace. Forwarding those deltas
        directly produced joined text in both the transcript and TTS input. A
        non-streaming completion preserves the provider's canonical message
        while this adapter converts it back into one Pipecat-compatible chunk.
        """
        adapter = self.get_llm_adapter()
        params_from_context = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=True,
        )
        params = self.build_chat_completion_params(params_from_context)
        params["stream"] = False
        params.pop("stream_options", None)

        response = await self._client.chat.completions.create(**params)

        choices = []
        for choice in response.choices:
            message = choice.message
            tool_calls = None
            if message.tool_calls:
                tool_calls = []
                for index, tool_call in enumerate(message.tool_calls):
                    item = tool_call.model_dump(exclude_none=True)
                    item["index"] = index
                    tool_calls.append(item)
            choices.append(
                {
                    "index": choice.index,
                    "delta": {
                        "role": message.role,
                        "content": message.content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": choice.finish_reason,
                }
            )

        payload = {
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": choices,
        }
        if response.usage is not None:
            payload["usage"] = response.usage.model_dump(exclude_none=True)
        if getattr(response, "system_fingerprint", None) is not None:
            payload["system_fingerprint"] = response.system_fingerprint

        chunk = ChatCompletionChunk.model_validate(payload)

        async def one_complete_chunk():
            yield chunk

        return one_complete_chunk()

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Sarvam chat completion request.

        Starts from OpenAI-compatible defaults, then removes unsupported
        request fields and applies Sarvam-specific options.
        """
        self._validate_tool_parameters(params_from_context)

        params = super().build_chat_completion_params(params_from_context)
        params.pop("stream_options", None)
        params.pop("max_completion_tokens", None)
        params.pop("service_tier", None)

        if is_given(self._settings.wiki_grounding) and self._settings.wiki_grounding is not None:
            params["wiki_grounding"] = self._settings.wiki_grounding
        if (
            is_given(self._settings.reasoning_effort)
            and self._settings.reasoning_effort is not None
        ):
            params["reasoning_effort"] = self._settings.reasoning_effort

        return params

    def _validate_model(self, model: str):
        if model not in self._SUPPORTED_MODELS:
            allowed = ", ".join(sorted(self._SUPPORTED_MODELS))
            raise ValueError(f"Unsupported Sarvam LLM model '{model}'. Allowed values: {allowed}.")

    def _validate_tool_parameters(self, params_from_context: OpenAILLMInvocationParams):
        tools = params_from_context.get("tools", NOT_GIVEN)
        tool_choice = params_from_context.get("tool_choice", NOT_GIVEN)

        has_tools = (
            openai_is_given(tools)
            and tools is not None
            and (not isinstance(tools, list) or len(tools) > 0)
        )
        has_tool_choice = openai_is_given(tool_choice) and tool_choice is not None

        if has_tool_choice and not has_tools:
            raise ValueError("Sarvam requires non-empty `tools` when `tool_choice` is provided.")
