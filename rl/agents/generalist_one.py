from typing import Dict

import numpy as np

from conversation import BaseConversationFactory
from misc import opencv_to_pil
from rl.agents import SimpleLLMAgent
from rl.agents.semantic_units.action_space_specialist import ActionSpaceSpecialistFailure, ActionSpaceSpecialist
from rl.agents.semantic_units.decision_making_specialist import DecisionMakingSpecialist
from rl.agents.semantic_units.detection_specialist import GoalIdentifier, SimpleDetectionSpecialist, \
    SplittingDetectionSpecialist
from rl.agents.semantic_units.execution_specialist import ExecutionSpecialist
from rl.agents.semantic_units.summary_specialist import SummarySpecialist


class GeneralistOne(SimpleLLMAgent):
    def __init__(self, conversation_factory: BaseConversationFactory, prompt: str):
        super().__init__(conversation_factory.get_conversation(), prompt)
        self.conversation_factory = conversation_factory
        self.uninitialised = True

        self.decision_maker = DecisionMakingSpecialist(conversation_factory=conversation_factory)
        self.action_space_specialist = ActionSpaceSpecialist(conversation_factory=conversation_factory)
        self.execution_specialist = ExecutionSpecialist(conversation_factory=conversation_factory)
        self.summary_specialist = SummarySpecialist(conversation_factory=conversation_factory)

        self.summary_archive = []
        self.action_space_archive = []
        self.decision_archive = []
        self.action_archive = []
        self.prompt_summary_archive = []
        self.strategies = []

        self.previous_actions = []
        self.previous_summary = "This is the first move; as such, there is no previous summary."

    def _act(self, image: np.ndarray, altitude: np.ndarray, collision: int, **kwargs):
        image = opencv_to_pil(image)
        collision = True if collision == 1 else False

        if len(self.strategies) > 0:
            strategy = self.strategies[0]
        else:
            strategy = self.decision_maker.get_decision(
                {
                    "prompt_for_task": self.prompt,
                    "YOUR TASK": "based on that prompt, define a strategy for the drone to follow. IT SHOULD STICK TO IT FOR THE ENTIRE TASK."
                }
            )
            self.strategies.append(strategy)

        summary = self.summary_specialist.get_summary(
            {
                "prompt": self.prompt,
                "current_drone_view": image,
                # "detection_coords": detection_coords,
                "previous_moves": self.previous_actions,
                "previous_summary": self.previous_summary,
                "collision_after_previous_move": collision,
                "current_ABSOLUTE_altitude_very_important": altitude.item(),
                "decision_maker_decisions": self.decision_archive,
                "devised_strategy": strategy,
            }
        )

        if len(self.summary_archive) > 0:
            prompt_summary = self.summary_archive[0]
        else:
            prompt_summary = self.summary_specialist.get_summary(
                {
                    "your_goal": "briefly summarise the task prompt for the component that will decide the next action",
                    "task_prompt": self.prompt,
                }
            )
            self.prompt_summary_archive.append(prompt_summary)

        self.previous_summary = summary

        decision = self.decision_maker.get_decision(
            {
                "prompt_summary": prompt_summary,
                "history": summary,
                "current_drone_view": image,
                "current_altitude": altitude.item(),
                "goal": "MOVE OR CLAIM FOUND",
                "devised_strategy": strategy,
            }
        )

        action = self.execution_specialist.get_formatted_action(
            {
                "instruction": self.prompt,
                "action": decision,
                "context": {
                    "current_altitude": altitude.item(),
                    "image": image,
                }
            }
        )

        self.summary_archive.append(summary)
        # self.action_space_archive.append(action_space)
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
            "prompt_summary_archive": self.prompt_summary_archive,
            "devised_strategies": self.strategies,
        }
