
import sys
import os
from unittest.mock import MagicMock
from collections import namedtuple

# Add src to path
sys.path.append("src")
os.environ["OPENAI_API_KEY"] = "dummy"

from core.client import OpenAIWrapper, LLMResponse

def test_llm_response_structure():
    print("Testing LLMResponse structure...")
    fields = LLMResponse._fields
    print(f"Fields: {fields}")
    
    if "reasoning" in fields:
        print("FAILED: 'reasoning' field is still present in LLMResponse.")
        sys.exit(1)
    else:
        print("PASSED: 'reasoning' field is absent.")

def test_openai_wrapper_response():
    print("\nTesting OpenAIWrapper generate response structure...")
    
    # Mock config
    from types import SimpleNamespace
    config = SimpleNamespace(
        client_name="openai",
        model_id="gpt-test",
        base_url=None,
        timeout=10,
        generate_kwargs={"max_tokens": 10},
        max_retries=1,
        delay=0,
        api_type="chat",
        api_key="test"
    )
    
    client = OpenAIWrapper(config)
    
    # Patch _initialize_client to avoid overwriting our mock
    client._initialize_client = MagicMock()
    
    client.client = MagicMock()
    
    # Mocking chat completion
    mock_choice = MagicMock()
    mock_choice.message.content = "Test content"
    mock_choice.finish_reason = "stop"
    
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 5
    mock_usage.completion_tokens = 5
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model = "gpt-test"
    mock_response.model_dump.return_value = {}
    
    client.client.chat.completions.create.return_value = mock_response
    
    # Test Message
    TestMessage = namedtuple("TestMessage", ["role", "content", "attachment"])
    messages = [TestMessage(role="user", content="Hi", attachment=None)]
    
    response = client.generate(messages)
    
    print(f"Response object: {response}")
    
    if hasattr(response, "reasoning"):
        print("FAILED: Response object has 'reasoning' attribute.")
        sys.exit(1)
    
    try:
        val = response.reasoning
        print(f"FAILED: Accessing response.reasoning returned {val}")
        sys.exit(1)
    except AttributeError:
        print("PASSED: Accessing response.reasoning raised AttributeError as expected.")

if __name__ == "__main__":
    test_llm_response_structure()
    test_openai_wrapper_response()
    print("\nALL TESTS PASSED")
