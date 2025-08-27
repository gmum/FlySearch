from typing import Callable, List, Dict

from conversation import Conversation
from glimpse_generators.unreal_client_wrapper import UnrealDiedException
from response_parsers import ParsingError
from rl.evaluation import EvaluationState
from rl.evaluation.validators import BaseValidator
from rl.evaluation.loggers import BaseLogger
from rl.agents import BaseAgent, SimpleLLMAgentFactory, BaseAgentFactory
from rl.environment import BaseFlySearchEnv, DroneCannotSeeTargetException
from scenarios import BaseScenarioMapper

class TrajectoryEvaluator:
    """
    Class tasked with performing a single evaluation of the agent in the environment.
    """

    def __init__(self, agent_factory: BaseAgentFactory, environment: BaseFlySearchEnv, max_glimpses: int,
                 scenario_mapper: BaseScenarioMapper,
                 loggers: List[BaseLogger], validators: List[BaseValidator], seed: int, forgiveness: int,
                 prompt_factory: Callable[[int, str, int], str]):
        self.agent_factory = agent_factory
        self.environment = environment
        self.max_glimpses = max_glimpses
        self.scenario_mapper = scenario_mapper
        self.loggers = loggers
        self.validators = validators
        self.seed = seed
        self.forgiveness = forgiveness
        self.prompt_factory = prompt_factory

        self.first_observation = None
        self.first_info = None

        self.agent = None
        self.scenario = None

        if not self.environment.resources_initialized:
            raise ValueError("Environment resources not initialized. Use `with` statement before using the evaluator.")

        self._prepare_environment()
        self._prepare_agent()

    def _prepare_environment(self):
        throws = 0

        generate_new_scenario: bool = True

        while True:
            if generate_new_scenario:
                scenario = self.scenario_mapper.create_random_scenario(seed=self.seed)
            try:
                self.first_observation, self.first_info = self.environment.reset(options=scenario)
            except UnrealDiedException:
                generate_new_scenario = False # It was likely not scenarios fault
                continue
            except DroneCannotSeeTargetException:
                generate_new_scenario = True # Scenario is broken
                throws += 1
            else:
                break

        scenario["throws"] = throws

        self.scenario = scenario

    def _prepare_agent(self):
        object_type = self.scenario["object_type"]
        object_desc = self.scenario_mapper.get_description(object_type)

        self.scenario["passed_object_name"] = object_desc

        search_area_rectangle_length = 400
        self.agent = self.agent_factory.create_agent(
            self.prompt_factory(self.max_glimpses, object_desc, search_area_rectangle_length),
            max_glimpses=self.max_glimpses, object_desc=object_desc,
            search_area_rectangle_length=search_area_rectangle_length)

    def tell_loggers_about_termination(self, termination_info: Dict):
        for logger in self.loggers:
            logger.log_termination(termination_info)

    def tell_loggers(self, evaluation_state: EvaluationState):
        for logger in self.loggers:
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

    def nuke_loggers(self):
        for logger in self.loggers:
            logger.nuke()

    def nuke_validators(self):
        for validator in self.validators:
            validator.nuke()

    def tell_validators_about_starting_altitude(self, starting_altitude: int):
        for validator in self.validators:
            validator.inform_about_starting_altitude(starting_altitude)

    def evaluate(self):
        success = False
        while not success:
            try:
                self._evaluate_unsafe()
                success = True
            except UnrealDiedException:
                if self.scenario is None:
                    raise ValueError("Sanity check failed: Scenario is not initialised, but it should be set during initialisation.")
                inner_success = False

                self.nuke_loggers()
                self.nuke_validators()

                while not inner_success:
                    try:
                        self.first_observation, self.first_info = self.environment.reset(options=self.scenario)
                        inner_success = True
                    except UnrealDiedException:
                        inner_success = False

                self._prepare_agent()


    def _evaluate_unsafe(self):
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
                    scenario=self.scenario
                )
                self.tell_loggers(evaluation_state)
                self.tell_loggers_about_termination({"reason": "parsing error"})
                return
            except:
                self.tell_loggers_about_termination({"reason": "unknown error during agent action sampling"})
                raise

            for fails in range(self.forgiveness):
                evaluation_state = EvaluationState(
                    observation=observation,
                    action=action,
                    info=info,
                    observation_number=glimpse_number,
                    correction_number=fails,
                    agent_info=self.agent.get_agent_info(),
                    scenario=self.scenario
                )

                self.tell_loggers(evaluation_state)
                valid, fail_info = self.tell_validators(evaluation_state)

                if valid:
                    break

                try:
                    action = self.agent.correct_previous_action(fail_reason=fail_info)
                except:
                    self.tell_loggers_about_termination({"reason": "environment error"})
                    raise
            else:
                self.tell_loggers_about_termination({"reason": "validation forgiveness ran out"})
                break

            observation, _, terminated, _, info = self.environment.step(action)

            if terminated:
                self.tell_loggers_about_termination({"reason": "found claimed"})
                break
        else:
            self.tell_loggers_about_termination({"reason": "glimpses ran out"})
