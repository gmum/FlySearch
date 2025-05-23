from typing import Dict

import numpy as np

from conversation import BaseConversationFactory
from misc import opencv_to_pil
from rl.agents import SimpleLLMAgent
from rl.agents.semantic_units.action_space_specialist import ActionSpaceSpecialistFailure, ActionSpaceSpecialist
from rl.agents.semantic_units.decision_making_specialist import DecisionMakingSpecialist
from rl.agents.semantic_units.detection_specialist import GoalIdentifier, SimpleDetectionSpecialist
from rl.agents.semantic_units.execution_specialist import ExecutionSpecialist
from rl.agents.semantic_units.summary_specialist import SummarySpecialist


class GeneralistOne(SimpleLLMAgent):
    def __init__(self, conversation_factory: BaseConversationFactory, prompt: str):
        super().__init__(conversation_factory.get_conversation(), prompt)
        self.conversation_factory = conversation_factory
        self.uninitialised = True

        self.object_name = GoalIdentifier(conversation_factory=conversation_factory).get_goal({"prompt": prompt})
        self.detection_specialist = SimpleDetectionSpecialist(conversation_factory=conversation_factory)
        self.decision_maker = DecisionMakingSpecialist(conversation_factory=conversation_factory)
        self.action_space_specialist = ActionSpaceSpecialist(conversation_factory=conversation_factory)
        self.execution_specialist = ExecutionSpecialist(conversation_factory=conversation_factory)
        self.summary_specialist = SummarySpecialist(conversation_factory=conversation_factory)

        self.summary_archive = []
        self.action_space_archive = []
        self.decision_archive = []
        self.action_archive = []

        self.previous_actions = []
        self.previous_summary = "This is the first move; as such, there is no previous summary."

    def _act(self, image: np.ndarray, altitude: np.ndarray, collision: int, **kwargs):
        image = opencv_to_pil(image)
        collision = True if collision == 1 else False

        annotated_image, detection_coords = self.detection_specialist.get_detections(image, self.object_name)
        summary = self.summary_specialist.get_summary(
            {
                "prompt": self.prompt,
                "current_drone_view": image,
                "search_target": self.object_name,
                "detection_coords": detection_coords,
                "annotated_image": annotated_image,
                "previous_moves": self.previous_actions,
                "previous_summary": self.previous_summary,
                "collision_after_previous_move": collision,
                "current_altitude_very_important": altitude,
            }
        )

        self.previous_summary = summary

        action_space = self.action_space_specialist.get_actions(
            {
                "summary": summary,
                "annotated_image": annotated_image,
            }
        )

        decision = self.decision_maker.get_decision(
            {
                "actions": action_space,
            }
        )

        action = self.execution_specialist.get_formatted_action(
            {
                "instruction": self.prompt,
                "action": decision,
            }
        )

        self.summary_archive.append(summary)
        self.action_space_archive.append(action_space)
        self.decision_archive.append(decision)
        self.action_archive.append(action)

        return self._return_action_from_response(action)

    def sample_action(self, observation: Dict) -> Dict:
        action = super().sample_action(observation)
        self.previous_actions.append(action["coordinate_change"])

        return action

    def correct_previous_action(self, fail_reason: Dict):
        raise NotImplementedError("GeneralistOne does not support action correction.")

    def get_agent_info(self) -> Dict:
        return {
            "summary_archive": self.summary_archive,
            "action_space_archive": self.action_space_archive,
            "decision_archive": self.decision_archive,
            "action_archive": self.action_archive,
            "conversation_history": [],
            "object_name": self.object_name,
        }
