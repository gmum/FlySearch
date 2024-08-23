import numpy as np
import cv2

from enum import Enum
from PIL import Image

from image_glimpse_generator import ImageGlimpseGenerator, BasicImageGlimpseGenerator
from visual_vstar_explorer import VisualVStarExplorer
from cv2_and_numpy import opencv_to_pil, pil_to_opencv
from prompt_generation import get_starting_prompt_for_vstar_explorer, get_classification_prompt_for_vstar_explorer
from abstract_conversation import Conversation, Role
from abstract_response_parser import AbstractResponseParser, SimpleResponseParser


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
        self.raw_glimpse_returns: list[np.ndarray] = []
        super().__init__(image)

    def get_raw_glimpse(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        self.raw_glimpse_requests.append((x1, y1, x2, y2))

        result = super().get_raw_glimpse(x1, y1, x2, y2)
        self.raw_glimpse_returns.append(result)

        return result


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
            number_glimpses=5,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
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

    def test_explorer_terminates_ave_process_on_incorrect_message(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation(
            ["<Haha, I will put my coordinates in the comment block. This will be invalid. (0.3, 0.3) and (0.5, 0.5)>"])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=5,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_response() == -1

    def test_explorer_terminates_ave_process_on_invalid_coordinates(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation(
            ["<Haha!> (3.0, 7.0) and (0.5, 0.5)"])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=5,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_response() == -1
        assert explorer.get_failed_coord_request() == "<Haha!> (3.0, 7.0) and (0.5, 0.5)"

    def test_explorer_terminates_ave_process_on_invalid_coordinates_2(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation(
            ["<Haha!> (0.3, 0.4) and (0.1, 0.5)"])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=5,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_response() == -1
        assert explorer.get_failed_coord_request() == "<Haha!> (0.3, 0.4) and (0.1, 0.5)"

    def test_explorer_properly_parses_answers_without_comments(self):
        mock_glimpse_generator = MockGlimpseGenerator(np.zeros((100, 100, 3), dtype=np.uint8))

        explorer = VisualVStarExplorer(
            conversation=MockConversation([
                "(0.5, 0.5) and (1.0, 1.0)",
                "ANSWER: cat"
            ]),
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_response() == "cat"

    def test_if_ave_succeeded_failed_coord_request_is_none(self):
        mock_glimpse_generator = MockGlimpseGenerator(np.zeros((100, 100, 3), dtype=np.uint8))

        explorer = VisualVStarExplorer(
            conversation=MockConversation([
                "(0.5, 0.5) and (1.0, 1.0)",
                "ANSWER: cat"
            ]),
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_failed_coord_request() is None

    def test_comments_are_ignored(self):
        mock_glimpse_generator = MockGlimpseGenerator(np.zeros((100, 100, 3), dtype=np.uint8))

        very_long_text_with_comment = """
            <Litwo! Ojczyzno moja!
            
            Ty jesteś jak zdrowie.
            
            Zdrowe jest również wstawianie newline'ów w komentarzach.
            
            Haha, look, newline! 
            
            ąęć 
            
            
            
            
            
            
            miłego dnia
            
            
            
            >
            
            
            
            (0.32, 0.18) and (0.76, 0.1)
        """

        explorer = VisualVStarExplorer(
            conversation=MockConversation([
                very_long_text_with_comment,
                "ANSWER: cat"
            ]),
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_failed_coord_request() is None
        assert explorer.get_response() == "cat"

    def test_correct_glimpses_are_requested(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation([
            "(0.12, 0.23) and (0.45, 0.67)",
            "(0.44, 0.17) AnD (0.89, 0.29)",
            "(0.3, 0.5) AND (0.7, 0.7)",
            "ANSWER: dog"
        ])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=5,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert mock_glimpse_generator.raw_glimpse_requests == [(0.0, 0.0, 1.0, 1.0), (0.12, 0.23, 0.45, 0.67),
                                                               (0.44, 0.17, 0.89, 0.29),
                                                               (0.3, 0.5, 0.7, 0.7)]

        assert len([func_call for func_call in mock_conversation.get_function_calls() if
                    func_call[0] == mock_conversation.add_image_message]) == 4

        assert explorer.get_failed_coord_request() is None
        assert explorer.get_response() == "dog"

    def test_ave_process_fails_if_glimpse_limit_is_exceeded(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation([
            "(0.1, 0.2) and (0.4, 0.6)",
            "ANSWER: dog"
        ])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=1,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_response() == -1
        assert explorer.get_failed_coord_request() is None

    def test_ave_process_fails_if_glimpse_limit_is_exceeded_2(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation([
            "(0.1, 0.2) and (0.4, 0.6)",
            "(0.3, 0.5) and (0.7, 0.7)",
            "(0.1, 0.2) and (0.4, 0.6)",
            "ANSWER: dog"
        ])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=3,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_response() == -1
        assert explorer.get_failed_coord_request() is None

    def test_ave_process_doesnt_fail_if_glimpse_limit_is_not_exceeded(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation([
            "(0.1, 0.2) and (0.4, 0.6)",
            "(0.3, 0.5) and (0.7, 0.7)",
            "(0.1, 0.2) and (0.4, 0.6)",
            "ANSWER: dog"
        ])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=4,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_response() == "dog"
        assert explorer.get_failed_coord_request() is None

    def test_visual_explorer_properly_returns_glimpse_coordinate_requests(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation([
            "(0.12, 0.23) and (0.45, 0.67)",
            "(0.44, 0.17) AnD (0.89, 0.29)",
            "(0.3, 0.5) AND (0.7, 0.7)",
            "(0.111, 0.222) aNd (0.333, 0.444)",
            "ANSWER: dog"
        ])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=5,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        assert explorer.get_glimpse_requests() == [(0.0, 0.0, 1.0, 1.0), (0.12, 0.23, 0.45, 0.67),
                                                   (0.44, 0.17, 0.89, 0.29), (0.3, 0.5, 0.7, 0.7),
                                                   (0.111, 0.222, 0.333, 0.444)]

        assert explorer.get_failed_coord_request() is None
        assert explorer.get_response() == "dog"

    def test_glimpses_passed_to_conversation_are_the_ones_returned_by_glimpse_generator(self):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        mock_glimpse_generator = MockGlimpseGenerator(image)
        mock_conversation = MockConversation([
            "(0.12, 0.23) and (0.45, 0.67)",
            "(0.44, 0.17) AnD (0.89, 0.29)",
            "(0.3, 0.5) AND (0.7, 0.7)",
            "(0.111, 0.222) aNd (0.333, 0.444)",
            "ANSWER: dog"
        ])

        explorer = VisualVStarExplorer(
            conversation=mock_conversation,
            question="What animal do you see?",
            options=["cat", "dog"],
            glimpse_generator=mock_glimpse_generator,
            number_glimpses=5,
            response_parser=SimpleResponseParser(),
            starting_prompt_generator=get_starting_prompt_for_vstar_explorer,
            classification_prompt_generator=get_classification_prompt_for_vstar_explorer,
        )

        explorer.answer()

        glimpses_from_generator = mock_glimpse_generator.raw_glimpse_returns  # type: list[np.ndarray]
        glimpses_passed_to_conversation = [call[1]["image"] for call in mock_conversation.get_function_calls() if
                                           call[0] == mock_conversation.add_image_message]  # type: list[Image]

        assert len(glimpses_from_generator) == len(glimpses_passed_to_conversation)

        for from_generator, passed_to_conv in zip(glimpses_from_generator, glimpses_passed_to_conversation):
            passed_to_conv_cv = pil_to_opencv(passed_to_conv)

            assert np.array_equal(from_generator, passed_to_conv_cv)
