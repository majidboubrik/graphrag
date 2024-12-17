import asyncio
import logging
from typing import Any, Generator, AsyncGenerator, Literal

import dataiku
from graphrag.callbacks.llm_callbacks import BaseLLMCallback
from graphrag.query.llm.base import BaseLLM

dataiku_logger = logging.getLogger("QueryDataikuChatLLM")
dataiku_logger.setLevel(logging.INFO)
dataiku_handler = logging.FileHandler("/tmp/graphrag/query_dataiku_chat_llm.log")
dataiku_handler.setLevel(logging.DEBUG)
dataiku_logger.addHandler(dataiku_handler)

class QueryDataikuChatLLM(BaseLLM):
    def __init__(self, project_key: str, llm_id: str):
        self.project_key = project_key
        self.llm_id = llm_id
        self.client = dataiku.api_client()
        self.project = self.client.get_project(self.project_key)
        self.llm = self.project.get_llm(self.llm_id)

    def _prepare_completion(self, messages: str | list[Any], **kwargs: Any):
        completion = self.llm.new_completion()
        # messages can be either a string or a list of {role, content} dicts
        if isinstance(messages, str):
            # Treat as a single user message
            completion.with_message(messages, role="user")
        else:
            # Expecting a list of dicts with "role" and "content"
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                completion.with_message(content, role=role)

        # If JSON mode is requested
        if kwargs.get('json', False):
            completion.with_json_output()

        return completion

    def generate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous generation."""
        dataiku_logger.debug(f"messages: {messages}, kwargs: {kwargs}")
        completion = self._prepare_completion(messages, **kwargs)
        resp = completion.execute()
        clean_response = resp.text
        # If streaming is True but we have no streaming support, just return the full response
        return clean_response

    def stream_generate(
        self,
        messages: str | list[Any],
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Synchronous streaming generation (if not supported, yield the entire response)."""
        # For now, just yield the entire response as one chunk
        full_response = self.generate(messages, streaming=True, callbacks=callbacks, **kwargs)
        yield full_response

    async def agenerate(
        self,
        messages: str | list[Any],
        streaming: bool = True,
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous generation."""
        loop = asyncio.get_running_loop()

        # Create a wrapper lambda that calls self.generate with all arguments
        # Here we use keyword arguments explicitly so run_in_executor only gets a function with no extra arguments
        def sync_generate():
            return self.generate(messages=messages, streaming=streaming, callbacks=callbacks, **kwargs)

        return await loop.run_in_executor(None, sync_generate)

    async def astream_generate(
        self,
        messages: str | list[Any],
        callbacks: list[BaseLLMCallback] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Asynchronous streaming generation."""
        # If streaming is not truly supported, just yield the full response once
        response = await self.agenerate(messages, streaming=True, callbacks=callbacks, **kwargs)
        yield response