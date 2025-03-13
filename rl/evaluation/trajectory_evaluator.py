from typing import Callable, List, Dict

from rl.evaluation import EvaluationState
from rl.evaluation.validators import BaseValidator
from rl.evaluation.loggers import BaseLogger
from rl.agents import BaseAgent
from rl.environment import BaseFlySearchEnv, DroneCannotSeeTargetException
from scenarios import BaseScenarioMapper


class TrajectoryEvaluator:
    """
    Class tasked with performing a single evaluation of the agent in the environment.
    """

    def __init__(self, agent: BaseAgent, environment: BaseFlySearchEnv, max_glimpses: int,
                 scenario_mapper: BaseScenarioMapper,
                 loggers: List[BaseLogger], validators: List[BaseValidator], seed: int, forgiveness: int):
        self.agent = agent
        self.environment = environment
        self.max_glimpses = max_glimpses
        self.scenario_mapper = scenario_mapper
        self.loggers = loggers
        self.validators = validators
        self.seed = seed
        self.forgiveness = forgiveness

        self.first_observation = None
        self.first_info = None

        if not self.environment.resources_initialized:
            raise ValueError("Environment resources not initialized. Use `with` statement before using the evaluator.")

        self._prepare_environment()

    def _prepare_environment(self):
        throws = 0

        while True:
            scenario = self.scenario_mapper.create_random_scenario(seed=self.seed)
            try:
                self.first_observation, self.first_info = self.environment.reset(seed=self.seed, options=scenario)
            except DroneCannotSeeTargetException:
                throws += 1
            else:
                break

        scenario["throws"] = throws

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

    def evaluate(self):
        observation = self.first_observation
        info = self.first_info

        for glimpse_number in range(self.max_glimpses):

            action = self.agent.sample_action(observation)

            for fails in range(self.forgiveness):
                evaluation_state = EvaluationState(
                    observation=observation,
                    action=action,
                    info=info,
                    observation_number=glimpse_number,
                    correction_number=fails,
                )

                self.tell_loggers(evaluation_state)
                valid, info = self.tell_validators(evaluation_state)

                if valid:
                    break

                action = self.agent.correct_previous_action(fail_reason=info)
            else:
                self.tell_loggers_about_termination({"reason": "validation forgiveness ran out"})
                break

            # Execute these lines at the end
            observation, _, terminated, _, info = self.environment.step(action)

            if terminated:
                self.tell_loggers_about_termination({"reason": "found claimed"})
                break
        else:
            self.tell_loggers_about_termination({"reason": "glimpses ran out"})
