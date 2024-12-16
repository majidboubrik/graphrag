import json
import logging
import re
from typing import Any, Literal

import dataiku
from fnllm import ChatLLM
from fnllm.openai.types.aliases import OpenAIChatCompletionUserMessageParam

from fnllm.openai.types.chat.io import (
    OpenAIChatOutput,
    OpenAIChatCompletionMessageModel,
    OpenAIChatMessageInput,
    OpenAIChatCompletionInput
)
from fnllm.types.generics import TJsonModel, THistoryEntry, TModelParameters
from fnllm.types.io import LLMInput, LLMOutput
from typing_extensions import Unpack



class DataikuChatLLM(ChatLLM[OpenAIChatCompletionInput, OpenAIChatOutput, THistoryEntry, TModelParameters]):
    """
    DataikuChatLLM integrates a Dataiku project-provided LLM endpoint with the fnllm ChatLLM protocol.

    This class simulates OpenAI-like chat completions on top of a Dataiku LLM endpoint.
    Note: Currently, caching and streaming are not implemented in this class.
    """

    def __init__(self, project_key: str, llm_id: str):
        """
        Initialize the DataikuChatLLM.

        :param project_key: The Dataiku project key where the LLM is defined.
        :param llm_id: The LLM identifier within the project.
        """
        self.project_key = project_key
        self.llm_id = llm_id
        self.client = dataiku.api_client()
        self.project = self.client.get_project(self.project_key)
        self.llm = self.project.get_llm(self.llm_id)
        self.logger = logging.getLogger(__name__)


    def _build_prompt_message(self, prompt: OpenAIChatCompletionInput) -> tuple[list[OpenAIChatMessageInput], OpenAIChatMessageInput]:
        """
        Convert the given prompt into a list of OpenAIChatMessageInput messages.
        If prompt is a string, treat it as a user message.
        """
        if isinstance(prompt, str):
            prompt_message = OpenAIChatCompletionUserMessageParam(
                content=prompt,
                role="user",
            )
        else:
            # If prompt is already a structured message, use it directly.
            prompt_message = prompt

        return [prompt_message], prompt_message

    async def __call__(
        self,
        prompt: OpenAIChatCompletionInput,
        *,
        stream: Literal[False] | None = None,
        **kwargs: Unpack[LLMInput[TJsonModel, THistoryEntry, TModelParameters]],
    ) -> LLMOutput[OpenAIChatOutput, TJsonModel, THistoryEntry]:
        """
        Invoke the Dataiku LLM endpoint with the given prompt and optional parameters.

        :param prompt: The user prompt or message to send to the LLM.
        :param stream: Currently not supported; must be None or False.
        :param kwargs: Additional LLM parameters, such as json, json_model, history, etc.

        :return: LLMOutput containing OpenAIChatOutput and optional parsed JSON.
        """
        self.logger.info("Calling DataikuChatLLM")

        history = kwargs.get("history", [])
        json_mode = kwargs.get("json", False)
        json_model = kwargs.get("json_model", None)
        bypass_cache = kwargs.get("bypass_cache", False)  # Currently unused, but potentially useful later.

        self.logger.debug("Called with prompt=%r, json_mode=%s, bypass_cache=%s", prompt, json_mode, bypass_cache)
        self.logger.debug("Additional kwargs=%s", kwargs)

        # Build messages from prompt and history
        messages, prompt_message = self._build_prompt_message(prompt)
        all_messages = [*history, *messages]

        # Prepare Dataiku LLM completion request
        completion = self.llm.new_completion()
        for msg in all_messages:
            if isinstance(msg, OpenAIChatCompletionMessageModel):
                msg_role = "assistant"
                msg_content = msg.content
            else:
                msg_role = msg.get("role", "user")
                msg_content = msg.get("content", "")
            completion.with_message(msg_content, role=msg_role)

        # If JSON output is requested
        if json_mode:
            completion.with_json_output()

        # Execute the request
        resp = completion.execute()
        clean_response = re.sub(r'```json\s*|\s*```', '', resp.text or "")

        if not resp.success:
            self.logger.error("LLM call failed. Response: %s", resp.text)
            # Consider raising an exception or handling error scenarios more robustly
            # raise RuntimeError("Dataiku LLM returned an unsuccessful response.")

        # Construct raw assistant message
        raw_output = OpenAIChatCompletionMessageModel(
            role="assistant",
            content=clean_response if resp.success else "An error occurred.",
            function_call=None,
            name=None,
            tool_calls=[],
        )

        output = OpenAIChatOutput(
            raw_input=prompt_message,
            raw_output=raw_output,
            content=raw_output.content,
            usage=None,  # No usage metrics available from the Dataiku LLM currently
        )

        llm_output = LLMOutput(
            output=output,
            raw_json=None,
            parsed_json=None,
            history=all_messages + [raw_output],
            tool_calls=[],
        )

        # If json mode is requested and a json_model is provided, attempt to parse JSON
        if json_mode and json_model is not None:
            try:
                json_data = json.loads(clean_response)
                parsed_instance = json_model(**json_data)
                llm_output.raw_json = clean_response
                llm_output.parsed_json = parsed_instance
                self.logger.debug("Successfully parsed JSON output.")
            except Exception:
                self.logger.exception("Failed to parse JSON from the LLM response.")
                llm_output.raw_json = clean_response
                llm_output.parsed_json = None

        self.logger.info("Completed call with cleaned response=%r", clean_response)
        return llm_output

    def child(self, name: str) -> "DataikuChatLLM":
        """
        Create a child LLM instance with the same configuration. Typically used to
        manage multiple related LLM instances under different identifiers.
        """
        self.logger.debug("Creating child LLM with name=%s", name)
        return DataikuChatLLM(self.project_key, self.llm_id)