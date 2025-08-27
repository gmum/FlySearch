import numpy as np
import pytest

from conversation import Conversation
from rl.agents.simple_llm_agent import SimpleLLMAgent


class ConversationMock(Conversation):
    def __init__(self):
        self.messages = []
        self.returned_message = None
        self.all_sent = True

    def begin_transaction(self, role):
        pass

    def commit_transaction(self, send_to_vlm):
        if not send_to_vlm:
            self.all_sent = False

    def add_text_message(self, text):
        self.messages.append(("text", text))

    def add_image_message(self, image):
        self.messages.append(("image", image))

    def get_latest_message(self):
        return None, self.returned_message

    def get_conversation(self, save_urls=True):
        return [(None, self.returned_message)]

    def set_returned_message(self, msg: str):
        self.returned_message = msg


class TestSimpleLLMAgent:
    def test_agent_passes_data_correctly_and_returns_action(self):
        conversation = ConversationMock()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 255

        agent = SimpleLLMAgent(conversation, "prompt")
        conversation.set_returned_message("<action>(32, 555, -8)</action>")

        action = agent.sample_action({"image": image, "altitude": np.array([4]), "collision": 0})
        messages = conversation.messages

        assert action == {"coordinate_change": (32, 555, -8), "found": 0}
        assert messages[0] == ("text", "prompt")
        assert messages[1] == ("text", "Your current altitude is 4 meters above ground level.")
        assert messages[2][0] == "image"
        assert np.array_equal(messages[2][1], image[:, :, ::-1])
        assert conversation.all_sent

    def test_prompt_is_not_sent_after_first_step(self):
        conversation = ConversationMock()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        agent = SimpleLLMAgent(conversation, "prompt123")
        conversation.set_returned_message("<action>(32, 555, -8)</action>")

        agent.sample_action({"image": image, "altitude": np.array([4]), "collision": 0})
        agent.sample_action({"image": image, "altitude": np.array([4]), "collision": 0})

        messages = conversation.messages

        message_second_parts = [msg[1] for msg in messages]
        assert "prompt123" not in message_second_parts[1:]
        assert message_second_parts[0] == "prompt123"
        assert conversation.all_sent

    def test_agent_throws_if_asked_to_correct_when_uninitialised(self):
        conversation = ConversationMock()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        agent = SimpleLLMAgent(conversation, "prompt")

        with pytest.raises(ValueError):
            agent.correct_previous_action({"reason": "too_high", "alt_before": 0, "alt_after": 0, "alt_max": 0})

    def test_agent_corrects_previous_action_for_altitude(self):
        conversation = ConversationMock()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        agent = SimpleLLMAgent(conversation, "prompt")
        conversation.set_returned_message("<action>(32, 555, -8)</action>")

        agent.sample_action({"image": image, "altitude": np.array([4]), "collision": 0})

        conversation.set_returned_message("<action>(3, -5, -10)</action>")
        corrected_action = agent.correct_previous_action(
            {"reason": "too_high", "alt_before": 4, "alt_after": 20, "alt_max": 10})

        messages = conversation.messages

        complaint = messages[-1][1]

        assert complaint == "This command would cause you to fly too high. You can't fly higher than 10 meters. Your current altitude is 4 meters, which means that you can only fly 6 meters higher."
        assert conversation.all_sent
        assert corrected_action == {"coordinate_change": (3, -5, -10), "found": 0}

    def test_agent_corrects_previous_action_for_recklessness(self):
        conversation = ConversationMock()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        agent = SimpleLLMAgent(conversation, "prompt")
        conversation.set_returned_message("<action>(32, 555, -8)</action>")

        agent.sample_action({"image": image, "altitude": np.array([4]), "collision": 0})
        conversation.set_returned_message("<action>(3, -5, -10)</action>")
        corrected_action = agent.correct_previous_action({"reason": "reckless"})

        messages = conversation.messages

        complaint = messages[-1][1]

        assert complaint == "This command would endanger the drone, as you would fly out of bounds of the last seen image, possibly flying into unknown territories, recklessly. Please adjust your command so that you don't fly out of bounds of the previous glimpse."

        assert conversation.all_sent
        assert corrected_action == {"coordinate_change": (3, -5, -10), "found": 0}

    def test_agent_corrects_previous_action_for_out_of_bounds(self):
        conversation = ConversationMock()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        agent = SimpleLLMAgent(conversation, "prompt")
        conversation.set_returned_message("<action>(32, 555, -8)</action>")

        agent.sample_action({"image": image, "altitude": np.array([4]), "collision": 0})
        conversation.set_returned_message("<action>(3, -5, -10)</action>")
        corrected_action = agent.correct_previous_action({"reason": "out_of_bounds", "xy_bound": 213})

        messages = conversation.messages

        complaint = messages[-1][1]

        assert complaint == "This command would cause you to fly out of the search area's bounds. You can't fly further than 213 meters from the starting point in any axis."

        assert conversation.all_sent
        assert corrected_action == {"coordinate_change": (3, -5, -10), "found": 0}

    def test_properly_returns_found(self):
        conversation = ConversationMock()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        agent = SimpleLLMAgent(conversation, "prompt")
        conversation.set_returned_message("asdfsdf<action>FOUND</action>")

        action = agent.sample_action({"image": image, "altitude": np.array([4]), "collision": 0})

        assert action == {"found": 1, "coordinate_change": (0, 0, 0)}
        assert conversation.all_sent

    def test_can_return_found_for_correction(self):
        conversation = ConversationMock()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        agent = SimpleLLMAgent(conversation, "prompt")

        conversation.set_returned_message("<action>(32, 555, -8)</action>")
        agent.sample_action({"image": image, "altitude": np.array([4]), "collision": 0})

        conversation.set_returned_message("found")
        action = agent.correct_previous_action({"reason": "too_high", "alt_before": 4, "alt_after": 20, "alt_max": 10})

        assert action == {"found": 1, "coordinate_change": (0, 0, 0)}
        assert conversation.all_sent
