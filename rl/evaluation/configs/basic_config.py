from prompts import xml_found_prompt
from rl.agents import SimpleLLMAgentFactory
from rl.environment import BaseFlySearchEnv, CityFlySearchEnv
from rl.evaluation.configs import ExperimentConfig
from scenarios import CityScenarioMapper, BaseScenarioMapper


class BasicConfig(ExperimentConfig):
    def __init__(self, conversation_factory, environment: BaseFlySearchEnv, scenario_mapper: BaseScenarioMapper):
        super().__init__(
            agent_factory=SimpleLLMAgentFactory(conversation_factory),
            environment=environment,
            scenario_mapper=scenario_mapper,
            loggers=[],
            validators=[],
            forgiveness=5,
            number_of_runs=3,
            number_of_glimpses=10,
            prompt_factory=xml_found_prompt
        )
