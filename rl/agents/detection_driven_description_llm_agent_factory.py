from rl.agents import BaseAgentFactory


class DetectionDrivenDescriptionLLMAgentFactory(BaseAgentFactory):
    def __init__(self, conversation_factory):
        self.conversation_factory = conversation_factory

    def create_agent(self, prompt: str, object_desc: str, **kwargs):
        from rl.agents.detection_driven_description_llm_agent import DetectionDrivenDescriptionLLMAgent
        return DetectionDrivenDescriptionLLMAgent(
            conversation=self.conversation_factory.get_conversation(),
            prompt=prompt,
            object_desc=object_desc,
            description_conversation_factory=self.conversation_factory
        )
