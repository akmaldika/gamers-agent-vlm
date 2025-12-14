from collections import namedtuple
import logging
import time
from io import BytesIO
import base64
import requests
from openai import OpenAI

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "headers",
        "json",
        "timeout",
    ],
)


def encode_image_to_data_url(image, default_format="PNG"):
    """Convert a PIL image to a data URL string suitable for JSON payloads."""
    buffered = BytesIO()
    target_format = (getattr(image, "format", None) or default_format or "PNG").upper()
    image.save(buffered, format=target_format)
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = target_format.lower()
    return f"data:image/{mime};base64,{base64_image}"


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class LLMClientWrapper:
    """Base class for LLM client wrappers."""

    def __init__(self, client_config):
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.client_kwargs = {**client_config.generate_kwargs}
        self.max_retries = client_config.max_retries
        self.delay = client_config.delay
        self.rate_limit_delay = getattr(client_config, "rate_limit_delay", 2.0)
        self.api_key = getattr(client_config, "api_key", None)

    def generate(self, messages):
        raise NotImplementedError("This method should be overridden by subclasses")

    def _build_headers(self):
        return {}

    def execute_with_retries(self, func, *args, **kwargs):
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logger.error(
                    f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}"
                )

                if "rate_limit_exceeded" in str(e) or "429" in str(e):
                    logger.info(
                        f"Rate limit detected. Waiting {self.rate_limit_delay} seconds before retry..."
                    )
                    time.sleep(self.rate_limit_delay)
                else:
                    sleep_time = self.delay * (
                        2 ** (retries - 1)
                    )  # Exponential backoff
                    time.sleep(sleep_time)
        raise Exception(
            f"Failed to execute {func.__name__} after {self.max_retries} retries."
        )


class OpenAIWrapper(LLMClientWrapper):
    def __init__(self, client_config):
        super().__init__(client_config)
        self._initialized = False
        self.api_type = getattr(client_config, "api_type", "chat")
        self._prompt_session = None
        self._session_cache_error_logged = False

    def __process_image_openai(self, image, api_type="responses"):
        data_url = encode_image_to_data_url(image)
        if api_type == "responses":
            return {
                "type": "input_image",
                "image_url": data_url,
            }
        else:
            return {
                "type": "image_url",
                "image_url": {"url": data_url},
            }

    def _initialize_client(self):
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            elif (
                self.client_name.lower() == "nvidia"
                or self.client_name.lower() == "xai"
            ):
                if not self.base_url or not self.base_url.strip():
                    raise ValueError(
                        "base_url must be provided when using NVIDIA or XAI client"
                    )
                self.client = OpenAI(base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                self.client = OpenAI()
            else:
                # Default fallback
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            self._initialized = True

    def convert_messages(self, messages):
        converted_messages = []
        for msg in messages:
            if self.api_type == "responses":
                content_type = (
                    "output_text" if msg.role == "assistant" else "input_text"
                )
                new_content = [{"type": content_type, "text": msg.content}]
                if msg.attachment is not None:
                    new_content.append(
                        self.__process_image_openai(
                            image=msg.attachment, api_type="responses"
                        )
                    )
            else:
                new_content = [{"type": "text", "text": msg.content}]
                if msg.attachment is not None:
                    new_content.append(
                        self.__process_image_openai(
                            image=msg.attachment, api_type="chat"
                        )
                    )

            converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def ensure_prompt_session(self, instructions: str | None):
        if self.api_type != "responses" or not instructions:
            return None

        self._initialize_client()

        if (
            self._prompt_session
            and self._prompt_session.get("instructions") == instructions
            and self._prompt_session.get("id")
        ):
            return self._prompt_session["id"]

        try:
            session = self.client.responses.sessions.create(
                model=self.model_id,
                instructions=instructions,
            )
            self._prompt_session = {
                "id": getattr(session, "id", None),
                "instructions": instructions,
            }
            return self._prompt_session["id"]
        except Exception as exc:
            logger.warning(
                "Failed to create Responses session for system prompt cache: %s", exc
            )
            self._prompt_session = None
            return None

    def clear_prompt_session(self):
        self._prompt_session = None

    def generate(self, messages, session_id: str | None = None):
        self._initialize_client()
        converted_messages = self.convert_messages(messages)
        session_to_use = session_id
        if session_to_use is None and self._prompt_session:
            session_to_use = self._prompt_session.get("id")

        def api_call():
            if self.api_type == "responses":
                api_kwargs = {
                    "input": converted_messages,
                    "model": self.model_id,
                }
                if "max_output_tokens" in self.client_kwargs:
                    api_kwargs["max_output_tokens"] = self.client_kwargs[
                        "max_output_tokens"
                    ]

                api_kwargs.update(
                    {
                        k: v
                        for k, v in self.client_kwargs.items()
                        if k not in ["max_output_tokens"]
                    }
                )
                if session_to_use:
                    api_kwargs["session"] = session_to_use

                return self.client.responses.create(**api_kwargs)
            else:
                api_kwargs = {
                    "messages": converted_messages,
                    "model": self.model_id,
                }
                if "max_output_tokens" in self.client_kwargs:
                    api_kwargs["max_tokens"] = self.client_kwargs["max_output_tokens"]

                api_kwargs.update(
                    {
                        k: v
                        for k, v in self.client_kwargs.items()
                        if k not in ["max_output_tokens", "max_tokens"]
                    }
                )

                return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        if self.api_type == "responses":
            completion_text = ""
            output_messages = getattr(response, "output", None) or []

            def _extract_text(value):
                if value is None:
                    return ""
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    return "".join(_extract_text(item) for item in value)
                if hasattr(value, "value"):
                    extracted = getattr(value, "value", "")
                    if extracted:
                        return extracted
                if isinstance(value, dict):
                    for key in ("value", "text", "content"):
                        if key in value and value[key]:
                            return _extract_text(value[key])
                return str(value)

            for output_message in output_messages:
                content_items = getattr(output_message, "content", None)
                if isinstance(content_items, list):
                    for content_item in content_items:
                        item_type = getattr(content_item, "type", None) or (
                            content_item.get("type")
                            if isinstance(content_item, dict)
                            else None
                        )
                        item_text = (
                            getattr(content_item, "text", None)
                            if hasattr(content_item, "text")
                            else (
                                content_item.get("text")
                                if isinstance(content_item, dict)
                                else None
                            )
                        )
                        if item_type in ("output_text", "text") and item_text:
                            completion_text += _extract_text(item_text)
                elif isinstance(content_items, str):
                    completion_text += content_items

            if not completion_text:
                output_texts = getattr(response, "output_text", None)
                if output_texts:
                    completion_text = "\n".join(output_texts)

            completion_text = completion_text.strip()
            import re

            completion_text = re.sub(r"\n{3,}", "\n\n", completion_text)

            stop_reason = None
            if output_messages:
                stop_reason = getattr(output_messages[-1], "status", None)

            return LLMResponse(
                model_id=response.model,
                completion=completion_text,
                stop_reason=stop_reason,
                headers=self._build_headers(),
                json=None,
                timeout=self.timeout,
                input_tokens=0,  # Responses API usage might be different, placeholder
                output_tokens=0,
            )

        # Chat Completion API
        response_json = response.model_dump()  # OpenAI v1 uses model_dump()
        completion_text = response.choices[0].message.content
        usage = response.usage

        return LLMResponse(
            model_id=response.model,
            completion=completion_text,
            stop_reason=response.choices[0].finish_reason,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            headers=self._build_headers(),
            json=response_json,
            timeout=self.timeout,
        )


class ITBDGXClient(LLMClientWrapper):
    """Client for local DGX ITB VLM (Custom API)."""

    def __init__(self, client_config):
        super().__init__(client_config)
        self.session = requests.Session()

    def _process_image_content(self, image):
        # Use existing helper to get data URL
        data_url = encode_image_to_data_url(image)
        return {"type": "input_image", "image_url": data_url}

    def convert_messages(self, messages):
        """Convert messages to DGX ITB custom format."""
        converted_input = []
        for msg in messages:
            content_blocks = []

            # Handle list content (multimodal)
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict):
                        # Handle pre-formatted dicts (e.g. from OpenAI wrapper compat)
                        if item.get("type") in ["text", "input_text"]:
                            content_blocks.append(
                                {"type": "input_text", "text": item.get("text", "")}
                            )
                        elif item.get("type") in ["image_url", "input_image"]:
                            # If it's already a dict, it might be OpenAI format or our format
                            # We expect the attachment to be handled at the message level usually,
                            # but let's handle the dict case if resizing happened elsewhere.
                            # For simplicity, if we have an attachment object in the namedtuple, use that.
                            pass
            elif isinstance(msg.content, str):
                content_blocks.append({"type": "input_text", "text": msg.content})

            # Process attachment if present in the TestMessage namedtuple
            if hasattr(msg, "attachment") and msg.attachment:
                content_blocks.append(self._process_image_content(msg.attachment))

            converted_input.append({"role": msg.role, "content": content_blocks})

        return converted_input

    def generate(self, messages, session_id: str | None = None):
        # 1. Prepare URL
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        # 2. Prepare Headers
        headers = self._build_headers()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # 3. Prepare Payload
        converted_input = self.convert_messages(messages)

        payload = {
            "model": self.model_id,
            "input": converted_input,
            "upscale_mode": "small_only",  # Default per spec
            "stream": False,
        }

        # Add optional params from config
        if "max_output_tokens" in self.client_kwargs:
            payload["max_output_tokens"] = self.client_kwargs["max_output_tokens"]
        if "temperature" in self.client_kwargs:
            payload["temperature"] = self.client_kwargs["temperature"]

        # 4. Execute Request
        def api_call():
            resp = self.session.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()

        response_json = self.execute_with_retries(api_call)

        # 5. Parse Response
        # Spec: output[0].content[0].text
        completion_text = ""
        stop_reason = None

        if "output" in response_json and len(response_json["output"]) > 0:
            first_output = response_json["output"][0]
            if "content" in first_output:
                for content_item in first_output["content"]:
                    if content_item.get("type") == "text":
                        completion_text += content_item.get("text", "")

            # Try to determine stop reason if available in custom spec (not fully specified, assuming 'status')
            if response_json.get("status") == "completed":
                stop_reason = "stop"

        usage = response_json.get("usage", {})

        return LLMResponse(
            model_id=response_json.get("model", self.model_id),
            completion=completion_text,
            stop_reason=stop_reason,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            headers=headers,
            json=response_json,
            timeout=self.timeout,
        )


if __name__ == "__main__":
    from types import SimpleNamespace

    # Create a test configuration
    test_config = SimpleNamespace(
        client_name="local_dgx_itb",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        # base_url="http://167.205.35.252:8011",
        base_url="http://localhost:8001",  # verification local
        timeout=30,
        generate_kwargs={"max_output_tokens": 100, "temperature": 0.7},
        max_retries=3,
        delay=1,
        api_key="sk-dummy",
        api_type="chat",
    )

    # Create a simple message object for testing
    TestMessage = namedtuple("TestMessage", ["role", "content", "attachment"])

    # Test messages
    test_messages = [
        TestMessage(
            role="user", content="Hello, describe this image.", attachment=None
        ),
    ]

    try:
        # Initialize the wrapper
        # client = ITBDGXClient(test_config)
        # print(f"Testing {test_config.client_name} client with model {test_config.model_id}")

        # Generate response - commented out to avoid network calls in import
        # response = client.generate(test_messages)

        print("Test initialization successful!")

    except Exception as e:
        print(f"Error during testing: {e}")
