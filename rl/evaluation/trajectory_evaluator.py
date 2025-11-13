from typing import Dict, List, Optional

from glimpse_generators.unreal_client_wrapper import UnrealDiedException
from response_parsers.xml_response_parser import ParsingError
from rl.agents.base_agent_factory import BaseAgentFactory
from rl.environment.base_fly_search_env import BaseFlySearchEnv, InvalidScenarioException
from rl.evaluation.evaluation_state import EvaluationState
from rl.evaluation.loggers.base_logger import BaseLogger
from rl.evaluation.validators.base_validator import BaseValidator
from scenarios.base_scenario_mapper import BaseScenarioMapper, EpisodeCollectionMapper, EpisodeIteratorMapper


class TrajectoryEvaluator:
    """
    Class tasked with performing a single evaluation of the agent in the environment.
    """

    def __init__(
            self,
            agent_factory: BaseAgentFactory,
            environment: BaseFlySearchEnv,
            max_glimpses: int,
            scenario_mapper: BaseScenarioMapper,
            validators: List[BaseValidator],
            seed: int,
            forgiveness: int,
            prompt_factory,
            first_observation,
            first_info,
            scenario,
    ):
        self.agent_factory = agent_factory
        self.environment = environment
        self.max_glimpses = max_glimpses
        self.scenario_mapper = scenario_mapper
        self.validators = validators
        self.seed = seed
        self.forgiveness = forgiveness
        self.prompt_factory = prompt_factory

        self.first_observation = first_observation
        self.first_info = first_info
        self.scenario = scenario

        self.agent = None

        if not self.environment.resources_initialized:
            raise ValueError(
                "Environment resources not initialized. Use `with` statement before using the evaluator."
            )

        self._prepare_agent()

    @classmethod
    def prepare_simulator(cls,
                          agent_factory: BaseAgentFactory,
                          environment: BaseFlySearchEnv,
                          max_glimpses: int,
                          scenario_mapper: BaseScenarioMapper,
                          validators: List[BaseValidator],
                          seed: int,
                          forgiveness: int,
                          prompt_factory,
                          scenario_idx: Optional[int]) -> Optional['TrajectoryEvaluator']:

        if isinstance(scenario_mapper, EpisodeCollectionMapper):
            assert scenario_idx is not None, "scenario_idx must be provided when using EpisodeCollectionMapper"
            scenario = scenario_mapper[scenario_idx]
        elif isinstance(scenario_mapper, EpisodeIteratorMapper):
            scenario = next(scenario_mapper)
        else:
            raise ValueError("agent_factory must be an EpisodeIteratorMapper or EpisodeCollectionMapper")

        throws = 0
        max_retries = 30

        for _attempt in range(max_retries):
            try:
                first_observation, first_info = environment.reset(options=scenario)
                scenario["throws"] = throws
                return cls(
                    agent_factory=agent_factory,
                    environment=environment,
                    max_glimpses=max_glimpses,
                    scenario_mapper=scenario_mapper,
                    validators=validators,
                    seed=seed,
                    forgiveness=forgiveness,
                    prompt_factory=prompt_factory,
                    first_observation=first_observation,
                    first_info=first_info,
                    scenario=scenario
                )
            except UnrealDiedException:
                continue
            except InvalidScenarioException as ex:
                if isinstance(scenario_mapper, EpisodeIteratorMapper):
                    print(f"Invalid scenario {scenario} due to {str(ex)}, re-generating scenario.")
                    scenario = next(scenario_mapper)
                    throws += 1
                    continue
                else:
                    print(f"Invalid scenario {scenario} due to {str(ex)}, skipping scenario.")
                    return None

        raise RuntimeError(f"Failed to prepare simulator after {max_retries} attempts.")

    def _prepare_agent(self):
        object_type = self.scenario["object_type"]
        object_desc = self.scenario_mapper.get_description(object_type)

        self.scenario["passed_object_name"] = object_desc

        search_area_rectangle_length = 400
        self.agent = self.agent_factory.create_agent(
            self.prompt_factory(
                self.max_glimpses, object_desc, search_area_rectangle_length=search_area_rectangle_length
            ),
            max_glimpses=self.max_glimpses,
            object_desc=object_desc,
            search_area_rectangle_length=search_area_rectangle_length,
        )

    @staticmethod
    def tell_loggers_about_termination(loggers, termination_info: Dict):
        for logger in loggers:
            logger.log_termination(termination_info)

    @staticmethod
    def tell_loggers(loggers, evaluation_state: EvaluationState):
        for logger in loggers:
            logger.log(evaluation_state)

    def tell_validators(self, evaluation_state: EvaluationState):
        """
        If all validators return True, the action is valid. Otherwise, return info from the first failing validator.
        """

        for validator in self.validators:
            valid, info = validator.validate(evaluation_state)

            if not valid:
                return False, info

        return True, {}

    @staticmethod
    def nuke_loggers(loggers):
        for logger in loggers:
            logger.nuke()

    def nuke_validators(self):
        for validator in self.validators:
            validator.nuke()

    def tell_validators_about_starting_altitude(self, starting_altitude: int):
        for validator in self.validators:
            validator.inform_about_starting_altitude(starting_altitude)

    def evaluate(self, loggers: List[BaseLogger]):
        success = False
        while not success:
            try:
                self._evaluate_unsafe(loggers)
                success = True
            except UnrealDiedException:
                if self.scenario is None:
                    raise ValueError(
                        "Sanity check failed: Scenario is not initialised, but it should be set during initialisation."
                    )
                inner_success = False

                self.nuke_loggers(loggers)
                self.nuke_validators()

                while not inner_success:
                    try:
                        self.first_observation, self.first_info = (
                            self.environment.reset(options=self.scenario)
                        )
                        inner_success = True
                    except UnrealDiedException:
                        inner_success = False

                self._prepare_agent()

    def _evaluate_unsafe(self, loggers):
        observation = self.first_observation
        info = self.first_info

        starting_altitude = self.scenario["drone_rel_coords"][2]
        self.tell_validators_about_starting_altitude(starting_altitude)

        for glimpse_number in range(self.max_glimpses):
            observation["cheats"] = info  # Use this field ONLY for ablation purposes.
            try:
                action = self.agent.sample_action(observation)
            except ParsingError:
                # If agent gives some nonsense, still log initial position
                evaluation_state = EvaluationState(
                    observation=observation,
                    action={"ERROR": "invalid"},
                    info=info,
                    observation_number=glimpse_number,
                    correction_number=0,
                    agent_info=self.agent.get_agent_info(),
                    scenario=self.scenario,
                )
                self.tell_loggers(loggers, evaluation_state)
                self.tell_loggers_about_termination(loggers, {"reason": "parsing error"})
                return
            except:
                self.tell_loggers_about_termination(loggers,
                                                    {"reason": "unknown error during agent action sampling"}
                                                    )
                raise

            for fails in range(self.forgiveness):
                evaluation_state = EvaluationState(
                    observation=observation,
                    action=action,
                    info=info,
                    observation_number=glimpse_number,
                    correction_number=fails,
                    agent_info=self.agent.get_agent_info(),
                    scenario=self.scenario,
                )

                self.tell_loggers(loggers, evaluation_state)
                valid, fail_info = self.tell_validators(evaluation_state)

                if valid:
                    break

                try:
                    action = self.agent.correct_previous_action(fail_reason=fail_info)
                except:
                    self.tell_loggers_about_termination(loggers, {"reason": "environment error"})
                    raise
            else:
                self.tell_loggers_about_termination(loggers,
                                                    {"reason": "validation forgiveness ran out"}
                                                    )
                break

            observation, _, terminated, _, info = self.environment.step(action)

            if terminated:
                self.tell_loggers_about_termination(loggers, {"reason": "found claimed"})
                break
        else:
            self.tell_loggers_about_termination(loggers, {"reason": "glimpses ran out"})
