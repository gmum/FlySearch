from rl.agents import BaseAgent


class BaseAgentFactory:
    def create_agent(self, prompt: str) -> BaseAgent:
        raise NotImplementedError
