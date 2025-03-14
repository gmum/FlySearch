from prompts import xml_found_prompt
from rl.agents import SimpleLLMAgentFactory
from rl.environment import BaseFlySearchEnv, CityFlySearchEnv
from rl.evaluation.configs import ExperimentConfig
from rl.evaluation.loggers import WandbLoggerFactory
from rl.evaluation.loggers.wandb_logger import WandbLogger
from scenarios import CityScenarioMapper, BaseScenarioMapper


class BasicConfig(ExperimentConfig):
    def __init__(self, conversation_factory, environment: BaseFlySearchEnv, scenario_mapper: BaseScenarioMapper,
                 number_of_runs: int):
        super().__init__(
            agent_factory=SimpleLLMAgentFactory(conversation_factory),
            environment=environment,
            scenario_mapper=scenario_mapper,
            # loggers=[],
            logger_factories=[WandbLoggerFactory(project_name="WTLN-RL-T")],
            validator_factories=[],
            forgiveness=5,
            number_of_runs=number_of_runs,
            number_of_glimpses=10,
            prompt_factory=xml_found_prompt
        )
