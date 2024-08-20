from http.client import responses

import numpy as np
import cv2

from enum import Enum

from image_glimpse_generator import ImageGlimpseGenerator, BasicImageGlimpseGenerator
from visual_vstar_explorer import VisualVStarExplorer
from cv2_and_numpy import opencv_to_pil, pil_to_opencv
from prompt_generation import get_starting_prompt_for_vstar_explorer, get_classification_prompt_for_vstar_explorer
from abstract_conversation import Conversation, Role


class MockConversation(Conversation):
    def __init__(self, future_responses: list[str]):
        self.function_calls: list[tuple[callable, dict]] = []
        self.future_responses = future_responses
        self.responses = []

    def begin_transaction(self, role: Role):
        self.function_calls.append((self.begin_transaction, {"role": role}))

    def add_text_message(self, text: str):
        self.function_calls.append((self.add_text_message, {"text": text}))

    def add_image_message(self, image: np.ndarray):
        self.function_calls.append((self.add_image_message, {"image": image}))

    def commit_transaction(self, send_to_vlm: bool):
        if send_to_vlm:
            self.responses.append((Role.ASSISTANT, self.future_responses.pop(0)))
        self.function_calls.append((self.commit_transaction, {"send_to_vlm": send_to_vlm}))

    def rollback_transaction(self):
        self.function_calls.append((self.rollback_transaction, {}))

    def get_conversation(self):
        self.function_calls.append((self.get_conversation, {}))
        return self.responses

    def get_latest_message(self):
        self.function_calls.append((self.get_latest_message, {}))
        return self.responses[-1]

    def get_function_calls(self):
        return self.function_calls


class MockGlimpseGenerator(BasicImageGlimpseGenerator):
    def __init__(self, image: np.ndarray):
        self.raw_glimpse_requests: list[tuple[float, float, float, float]] = []
        super().__init__(image)

    def get_raw_glimpse(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        self.raw_glimpse_requests.append((x1, y1, x2, y2))

        return super().get_raw_glimpse(x1, y1, x2, y2)


class TestVisualVStarExplorer:
    def test_explorer_sends_prompt_question_and_first_glimpse_on_first_step(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation(["ANSWER: dog"])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=5
        )

        explorer.step(first=True)

        transaction_start = mock_conversation.get_function_calls()[0]
        prompt_message = mock_conversation.get_function_calls()[1]
        question_message = mock_conversation.get_function_calls()[2]
        glimpse_message = mock_conversation.get_function_calls()[3]
        transaction_commit = mock_conversation.get_function_calls()[4]

        assert transaction_start[0] == mock_conversation.begin_transaction
        assert transaction_start[1]["role"] == Role.USER

        assert prompt_message[0] == mock_conversation.add_text_message
        assert prompt_message[1]["text"] == get_starting_prompt_for_vstar_explorer(5)

        assert question_message[0] == mock_conversation.add_text_message
        assert question_message[1]["text"] == get_classification_prompt_for_vstar_explorer("What animal do you see?",
                                                                                           ["cat", "dog"])

        assert glimpse_message[0] == mock_conversation.add_image_message
        assert np.array_equal(image, pil_to_opencv(glimpse_message[1]["image"]))

        assert transaction_commit[0] == mock_conversation.commit_transaction
        assert transaction_commit[1]["send_to_vlm"] == True

    def test_explorer_properly_parses_answers_without_comments(self):
        mock_glimpse_generator = MockGlimpseGenerator(np.zeros((100, 100, 3), dtype=np.uint8))

        explorer = VisualVStarExplorer(
            conversation=MockConversation([
                "(0.5, 0.5) and (1.0, 1.0)",
                "ANSWER: cat"
            ]),
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator
        )

        explorer.answer()

        assert explorer.get_response() == "cat"
