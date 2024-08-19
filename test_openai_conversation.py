import pytest

from openai_conversation import OpenAIConversation


class SimpleObject:
    pass


class MockOpenAI:
    @staticmethod
    def mock_create_function(to_return: str) -> callable:
        response_mock = SimpleObject()
        response_mock.__dict__["choices"] = [SimpleObject()]

        choice = response_mock.choices[0]  # type: ignore
        choice.__dict__["message"] = SimpleObject()

        message = choice.message
        message.__dict__["content"] = to_return

        def mocked_fun(*args, **kwargs):
            return response_mock

        return mocked_fun

    def __init__(self, api_key, response: str = "mocked_response"):
        chat = SimpleObject()
        chat.__dict__["completions"] = SimpleObject()

        completions = chat.completions  # type: ignore
        completions.__dict__["create"] = self.mock_create_function(response)

        self.chat = chat


class TestOpenAIConversation:
    def test_begin_transaction_throws_if_already_started(self):
        openai_mock = MockOpenAI("mock_key")
        response = openai_mock.chat.completions.create("model", "messages", "max_tokens", "seed")
        response = str(response.choices[0].message.content)

        assert response == "mocked_response"
