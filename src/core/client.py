from collections import namedtuple

import logging
import time

from io import BytesIO
import base64

from openai import OpenAI

LLMResponse = namedtuple(
    "LLMResponse",
    [
        "model_id",
        "completion",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "reasoning",
    ],
)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class LLMClientWrapper:
    """Base class for LLM client wrappers.

    Provides common functionality for interacting with different LLM APIs, including
    handling retries and common configuration settings. Subclasses should implement
    the `generate` method specific to their LLM API.
    """

    def __init__(self, client_config):
        """Initialize the LLM client wrapper with configuration settings.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        self.client_name = client_config.client_name
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.timeout = client_config.timeout
        self.client_kwargs = {**client_config.generate_kwargs}
        self.max_retries = client_config.max_retries
        self.delay = client_config.delay

    def generate(self, messages):
        """Generate a response from the LLM given a list of messages.

        This method should be overridden by subclasses.

        Args:
            messages (list): A list of messages to send to the LLM.

        Returns:
            LLMResponse: The response from the LLM.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retries upon failure.

        Args:
            func (callable): The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function call.

        Raises:
            Exception: If the function fails after the maximum number of retries.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logger.error(f"Retryable error during {func.__name__}: {e}. Retry {retries}/{self.max_retries}")
                sleep_time = self.delay * (2 ** (retries - 1))  # Exponential backoff
                time.sleep(sleep_time)
        raise Exception(f"Failed to execute {func.__name__} after {self.max_retries} retries.")
    
class OpenAIWrapper(LLMClientWrapper):
    """Wrapper for interacting with the OpenAI API."""

    def __init__(self, client_config):
        """Initialize the OpenAIWrapper with the given configuration.

        Args:
            client_config: Configuration object containing client-specific settings.
        """
        super().__init__(client_config)
        self._initialized = False
        
    def __process_image_openai(self, image):
        """Process an image for OpenAI API by converting it to base64.

        Args:
            image: The image to process.

        Returns:
            dict: A dictionary containing the image data formatted for OpenAI.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # Return the image content for OpenAI
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
        }

    def _initialize_client(self):
        """Initialize the OpenAI client if not already initialized."""
        if not self._initialized:
            if self.client_name.lower() == "vllm":
                self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
            elif self.client_name.lower() == "nvidia" or self.client_name.lower() == "xai":
                if not self.base_url or not self.base_url.strip():
                    raise ValueError("base_url must be provided when using NVIDIA or XAI client")
                self.client = OpenAI(base_url=self.base_url)
            elif self.client_name.lower() == "openai":
                # For OpenAI, always use the standard API regardless of base_url
                self.client = OpenAI()
            self._initialized = True

    def convert_messages(self, messages):
        """Convert messages to the format expected by the OpenAI API.

        Args:
            messages (list): A list of message objects.

        Returns:
            list: A list of messages formatted for the OpenAI API.
        """
        converted_messages = []
        for msg in messages:
            new_content = [{"type": "text", "text": msg.content}]
            if msg.attachment is not None:
                new_content.append(self.__process_image_openai(image=msg.attachment))
            
            converted_messages.append({"role": msg.role, "content": new_content})
        return converted_messages

    def generate(self, messages):
        """Generate a response from the OpenAI API given a list of messages.

        Args:
            messages (list): A list of message objects.

        Returns:
            LLMResponse: The response from the OpenAI API.
        """
        self._initialize_client()
        converted_messages = self.convert_messages(messages)

        def api_call():
            # Create kwargs for the API call
            api_kwargs = {
                "messages": converted_messages,
                "model": self.model_id,
                "max_tokens": self.client_kwargs.get("max_tokens", 1024),
            }
            
            # Add optional parameters if they exist
            api_kwargs.update(
                {k: v for k, v in self.client_kwargs.items() if k != "max_tokens"}
            )

            return self.client.chat.completions.create(**api_kwargs)

        response = self.execute_with_retries(api_call)

        return LLMResponse(
            model_id=self.model_id,
            completion=response.choices[0].message.content.strip(),
            stop_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            reasoning=None,
        )
        
if __name__ == "__main__":
    from types import SimpleNamespace
    
    # Create a test configuration
    test_config = SimpleNamespace(
        client_name="openai",
        model_id="gpt-4.1",
        base_url=None,
        timeout=30,
        generate_kwargs={
            "max_tokens": 100,
            "temperature": 0.7
        },
        max_retries=3,
        delay=1
    )
    
    # Create a simple message object for testing
    TestMessage = namedtuple("TestMessage", ["role", "content", "attachment"])
    
    # Test messages
    test_messages = [
        TestMessage(role="user", content="Hello, how are you?", attachment=None),
        TestMessage(role="assistant", content="I'm doing well, thank you!", attachment=None),
        TestMessage(role="user", content="Can you help me with a simple math problem?", attachment=None)
    ]
    
    try:
        # Initialize the OpenAI wrapper
        client = OpenAIWrapper(test_config)
        print(f"Testing {test_config.client_name} client with model {test_config.model_id}")
        
        # Generate response
        response = client.generate(test_messages)
        
        print("\n--- Response ---")
        print(f"Model ID: {response.model_id}")
        print(f"Completion: {response.completion}")
        print(f"Stop Reason: {response.stop_reason}")
        print(f"Input Tokens: {response.input_tokens}")
        print(f"Output Tokens: {response.output_tokens}")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        print("Make sure you have set your OPENAI_API_KEY environment variable")