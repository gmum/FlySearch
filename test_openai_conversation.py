import pytest
from openai import OpenAI
from PIL import Image

from openai_conversation import OpenAIConversation
from abstract_conversation import Conversation, Role
from cv2_and_numpy import pil_to_opencv, opencv_to_pil


class SimpleObject:
    pass


class MockOpenAI:
    def mock_create_function(self, to_return: str) -> callable:
        response_mock = SimpleObject()
        response_mock.__dict__["choices"] = [SimpleObject()]

        choice = response_mock.choices[0]  # type: ignore
        choice.__dict__["message"] = SimpleObject()

        message = choice.message
        message.__dict__["content"] = to_return

        def mocked_fun(*args, **kwargs):
            self.mock_create_args.append(args)
            self.mock_create_kwargs.append(kwargs)

            messages = kwargs["messages"]
            self.mock_create_messages.append(list(messages))
            return response_mock

        return mocked_fun

    def __init__(self, api_key, response: str = "mocked_response"):
        chat = SimpleObject()
        chat.__dict__["completions"] = SimpleObject()

        completions = chat.completions  # type: ignore
        completions.__dict__["create"] = self.mock_create_function(response)

        self.chat = chat

        self.mock_create_args = []
        self.mock_create_kwargs = []
        self.mock_create_messages = []

    def get_mock_create_args(self):
        return self.mock_create_args

    def get_mock_create_kwargs(self):
        return self.mock_create_kwargs

    def get_mock_create_messages(self):
        return self.mock_create_messages


class TestOpenAIConversation:
    def test_begin_transaction_throws_if_already_started(self):
        openai_mock = MockOpenAI("mock_key")
        conversation = OpenAIConversation(
            openai_mock,  # type: ignore
            model_name="mock_model",
            seed=3,
            max_tokens=15
        )

        conversation.begin_transaction(Role.USER)

        with pytest.raises(Exception):
            conversation.begin_transaction(Role.USER)

        with pytest.raises(Exception):
            conversation.begin_transaction(Role.ASSISTANT)

        conversation.commit_transaction()

        conversation.begin_transaction(Role.ASSISTANT)

        with pytest.raises(Exception):
            conversation.begin_transaction(Role.USER)

        with pytest.raises(Exception):
            conversation.begin_transaction(Role.ASSISTANT)

    def test_commit_transaction_throws_if_not_started(self):
        openai_mock = MockOpenAI("mock_key")

        conversation = OpenAIConversation(
            openai_mock,  # type: ignore
            model_name="mock_model",
            seed=3,
            max_tokens=15
        )

        with pytest.raises(Exception):
            conversation.commit_transaction()

    def test_commit_transaction_does_not_send_to_vlm_by_default(self):
        openai_mock = MockOpenAI("mock_key")

        conversation = OpenAIConversation(
            openai_mock,  # type: ignore
            model_name="mock_model",
            seed=3,
            max_tokens=15
        )

        img = Image.new("RGB", (100, 100))

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message("mock_message")
        conversation.add_image_message(img)
        conversation.commit_transaction()

        conversation.begin_transaction(Role.ASSISTANT)
        conversation.add_text_message("mock_response")
        conversation.add_image_message(img)
        conversation.commit_transaction()

        assert len(openai_mock.get_mock_create_args()) == 0
        assert len(openai_mock.get_mock_create_kwargs()) == 0

    def test_arguments_are_only_passed_via_kwargs(self):
        # To make testing simpler.
        openai_mock = MockOpenAI("mock_key")

        conversation = OpenAIConversation(
            openai_mock,  # type: ignore
            model_name="mock_model",
            seed=3,
            max_tokens=15
        )

        img = Image.new("RGB", (100, 100))

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message("mock_message")
        conversation.add_image_message(img)
        conversation.commit_transaction(send_to_vlm=True)

        args = openai_mock.get_mock_create_args()[0]
        kwargs = openai_mock.get_mock_create_kwargs()[0]

        assert len(args) == 0

    def test_commit_transaction_throws_if_assistant_message_sent_to_vlm(self):
        openai_mock = MockOpenAI("mock_key")

        conversation = OpenAIConversation(
            openai_mock,  # type: ignore
            model_name="mock_model",
            seed=3,
            max_tokens=15
        )

        conversation.begin_transaction(Role.ASSISTANT)

        with pytest.raises(Exception):
            conversation.commit_transaction(send_to_vlm=True)

    def test_commit_transaction_sends_to_vlm_if_specified(self):
        openai_mock = MockOpenAI("mock_key")

        conversation = OpenAIConversation(
            openai_mock,  # type: ignore
            model_name="mock_model",
            seed=3,
            max_tokens=15
        )

        img = Image.new("RGB", (100, 100))

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message("mock_message")
        conversation.add_image_message(img)
        conversation.commit_transaction(send_to_vlm=True)

        user_args = openai_mock.get_mock_create_args()[0]
        user_kwargs = openai_mock.get_mock_create_kwargs()[0]

        # Assuming all arguments are passed as kwargs
        user_message = user_kwargs["messages"][0]
        user_content = user_message["content"]

        assert len(openai_mock.get_mock_create_args()) == 1
        assert len(openai_mock.get_mock_create_kwargs()) == 1

        assert len(user_args) == 0

        assert user_kwargs["model"] == "mock_model"
        assert user_kwargs["max_tokens"] == 15
        assert user_kwargs["seed"] == 3
        assert user_message["role"] == "user"
        assert user_content[0]["type"] == "text"
        assert user_content[0]["text"] == "mock_message"
        assert user_content[1]["type"] == "image_url"

    def test_commit_transaction_keeps_history_of_conversation(self):
        openai_mock = MockOpenAI("mock_key")

        conversation = OpenAIConversation(
            openai_mock,  # type: ignore
            model_name="mock_model",
            seed=3,
            max_tokens=15
        )

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message("mock_message")
        conversation.commit_transaction(send_to_vlm=True)

        conversation.begin_transaction(Role.USER)
        conversation.add_text_message("mock_message2")
        conversation.commit_transaction(send_to_vlm=True)

        messages = openai_mock.get_mock_create_messages()

        assert len(messages) == 2

        messages_first = messages[0]
        messages_second = messages[1]

        assert messages_first[0] == messages_second[0]
        assert messages_second[0]["role"] == "user"
        assert messages_second[1]["role"] == "assistant"
        assert messages_second[2]["role"] == "user"

        assert messages_second[0]["content"][0]["text"] == "mock_message"
        assert messages_second[1]["content"][0]["text"] == "mocked_response"
        assert messages_second[2]["content"][0]["text"] == "mock_message2"
