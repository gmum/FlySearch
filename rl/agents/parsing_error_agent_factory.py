from rl.agents import BaseAgentFactory
from rl.agents.parsing_error_agent import ParsingErrorAgent


class ParsingErrorAgentFactory(BaseAgentFactory):
    def __init__(self, conversation_factory):
        pass

    def create_agent(self, prompt: str, **kwargs) -> ParsingErrorAgent:
        return ParsingErrorAgent()
