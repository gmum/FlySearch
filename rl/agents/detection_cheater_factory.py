from rl.agents import BaseAgentFactory
from rl.agents.detection_cheater_agent import DetectionCheaterAgent


class DetectionCheaterFactory(BaseAgentFactory):
    def __init__(self, conversation_factory):
        self.conversation_factory = conversation_factory

    def create_agent(self, prompt: str, object_desc: str, **kwargs):
        return DetectionCheaterAgent(
            conversation=self.conversation_factory.get_conversation(),
            prompt=prompt,
            object_desc=object_desc,
            description_conversation_factory=self.conversation_factory
        )
