from typing import Dict

from rl.agents import BaseAgent, BaseAgentFactory
from rl.evaluation import EvaluationState, TrajectoryEvaluator
from rl.evaluation.validators import BaseValidator
from rl.evaluation.loggers import BaseLogger
from rl.environment import BaseFlySearchEnv, DroneCannotSeeTargetException
from scenarios import BaseScenarioMapper, CityScenarioMapper, ForestScenarioMapper


class ScenarioMapperMock:
    def __init__(self):
        self.dict_with_returns = {}

    def set_dict_with_returns(self, dict_with_returns):
        self.dict_with_returns = dict_with_returns

    def create_random_scenario(self, seed):
        scenario = {"seed": seed, "object_type": "car"}
        scenario.update(self.dict_with_returns)

        return scenario

    def get_description(self, object_type):
        return f"a {object_type}"


class EnvironmentMock:
    def __init__(self):
        self.throws_on_reset = False
        self.configs_passed = []
        self.seeds_passed = []
        self.actions_passed = []
        self.obs_returned = None
        self.info_returned = None
        self.return_terminated = False
        self.resources_initialized = True

    def reset(self, seed=None, options=None):
        self.configs_passed.append(options)
        if self.throws_on_reset != 0:
            self.throws_on_reset -= 1
            raise DroneCannotSeeTargetException()
        return self.obs_returned, self.info_returned

    def step(self, action):
        self.actions_passed.append(action)
        return self.obs_returned, 0.0, self.return_terminated, False, self.info_returned

    def set_throws_on_reset(self, throws_on_reset):
        self.throws_on_reset = throws_on_reset

    def set_obs_returned(self, obs_returned):
        self.obs_returned = obs_returned

    def set_info_returned(self, info_returned):
        self.info_returned = info_returned

    def set_return_terminated(self, return_terminated):
        self.return_terminated = return_terminated

    def set_resources_initialized(self, resources_initialized):
        self.resources_initialized = resources_initialized


class AgentMock(BaseAgent):
    def __init__(self):
        self.action_to_return = None
        self.observations = []
        self.fail_reasons = []

    def sample_action(self, observation):
        self.observations.append(observation)
        return self.action_to_return

    def set_action_to_return(self, action_to_return):
        self.action_to_return = action_to_return

    def correct_previous_action(self, fail_reason: Dict) -> Dict:
        self.fail_reasons.append(fail_reason)
        return self.action_to_return

    def get_agent_info(self) -> Dict:
        return {"eee": 42}


class ValidatorMock(BaseValidator):
    def __init__(self):
        self.fail = False
        self.fail_modulo = 1
        self.fail_reason_str = "haha"

    def set_fail(self, fail):
        self.fail = fail

    def set_fail_modulo(self, fail_modulo):
        self.fail_modulo = fail_modulo

    def set_fail_reason_str(self, fail_reason_str):
        self.fail_reason_str = fail_reason_str

    def validate(self, evaluation_state: EvaluationState):
        if not self.fail:
            return True, {}

        if evaluation_state.correction_number % self.fail_modulo == 0:
            return False, {"reason": self.fail_reason_str}
        else:
            return True, {}


class LoggerMock(BaseLogger):
    def __init__(self):
        self.termination_info = []
        self.evaluation_states = []

    def log(self, evaluation_state: EvaluationState):
        self.evaluation_states.append(evaluation_state)

    def log_termination(self, termination_info: Dict):
        self.termination_info.append(termination_info)


class AgentFactoryMock(BaseAgentFactory):
    def __init__(self, agent):
        self.agent = agent
        self.prompt = None

    def create_agent(self, prompt, **kwargs):
        self.prompt = prompt
        return self.agent


def prompt_func(max_glimpses, object_desc, search_area_rectangle_length):
    return f"Prompt: {max_glimpses}, {object_desc}, {search_area_rectangle_length}"


class TestTrajectoryEvaluator:
    def test_trajectory_evaluator_passes_config_to_env(self):
        env_mock = EnvironmentMock()
        agent_mock = AgentMock()
        agent_factory = AgentFactoryMock(agent_mock)
        mapper = ScenarioMapperMock()
        mapper.set_dict_with_returns({"abc": 43})

        env_mock.set_throws_on_reset(0)
        evaluator = TrajectoryEvaluator(agent_factory, env_mock, 10, mapper, [], [], 155, 10, prompt_func)

        assert len(env_mock.configs_passed) == 1

        def get_subset(d):
            return {k: d[k] for k in ("seed", "abc", "object_type", "throws")}

        assert get_subset(env_mock.configs_passed[0]) == {"seed": 155, "abc": 43, "object_type": "car",
                                                          "throws": 0}  # Should add "throws" key

        assert agent_factory.prompt == "Prompt: 10, a car, 400"

    def test_trajectory_evaluator_attempts_new_scenario_creation_after_throw(self):
        env_mock = EnvironmentMock()
        agent_mock = AgentMock()
        agent_factory = AgentFactoryMock(agent_mock)
        mapper = ScenarioMapperMock()
        mapper.set_dict_with_returns({"abc": 43})

        env_mock.set_throws_on_reset(3)
        evaluator = TrajectoryEvaluator(agent_factory, env_mock, 6, mapper, [], [], 155, 10, prompt_func)

        def get_subset(d):
            return {k: d[k] for k in ("seed", "abc", "object_type")}

        assert len(env_mock.configs_passed) == 4
        assert get_subset(env_mock.configs_passed[0]) == {"seed": 155, "abc": 43, "object_type": "car"}
        assert get_subset(env_mock.configs_passed[1]) == {"seed": 155, "abc": 43, "object_type": "car"}
        assert get_subset(env_mock.configs_passed[2]) == {"seed": 155, "abc": 43, "object_type": "car"}
        assert get_subset(env_mock.configs_passed[3]) == {"seed": 155, "abc": 43, "object_type": "car"}

        assert agent_factory.prompt == "Prompt: 6, a car, 400"

    def test_trajectory_evaluator_passes_observation_to_agent(self):
        env_mock = EnvironmentMock()
        agent_mock = AgentMock()
        agent_factory = AgentFactoryMock(agent_mock)
        mapper = ScenarioMapperMock()
        mapper.set_dict_with_returns({"abc": 43})

        env_mock.set_obs_returned({"image": "image", "altitude": 10, "collision": 0})
        evaluator = TrajectoryEvaluator(agent_factory, env_mock, 1, mapper, [], [], 155, 10, prompt_func)
        evaluator.evaluate()

        assert len(agent_mock.observations) == 1
        assert agent_mock.observations[0] == {"image": "image", "altitude": 10, "collision": 0}

    def test_trajectory_evaluator_passes_many_actions_to_agent(self):
        env_mock = EnvironmentMock()
        agent_mock = AgentMock()
        agent_factory = AgentFactoryMock(agent_mock)
        mapper = ScenarioMapperMock()
        mapper.set_dict_with_returns({"abc": 43})

        env_mock.set_obs_returned({"image": "image", "altitude": 10, "collision": 0})

        agent_mock.set_action_to_return({"found": 0, "coordinate_change": (1, 2, 3)})
        evaluator = TrajectoryEvaluator(agent_factory, env_mock, 15, mapper, [], [], 155, 10, prompt_func)
        evaluator.evaluate()

        assert len(agent_mock.observations) == 15
        assert agent_mock.observations[0] == {"image": "image", "altitude": 10, "collision": 0}
        assert agent_mock.observations[1] == {"image": "image", "altitude": 10, "collision": 0}
        assert len(env_mock.actions_passed) == 15

    def test_tells_agent_to_redo_if_validator_has_objections(self):
        validator_1 = ValidatorMock()
        validator_2 = ValidatorMock()
        validator_3 = ValidatorMock()

        logger = LoggerMock()

        validator_1.set_fail(False)
        validator_2.set_fail(True)
        validator_3.set_fail(True)

        validator_2.set_fail_modulo(2)
        validator_3.set_fail_modulo(2)

        validator_2.set_fail_reason_str("reason 2")
        validator_3.set_fail_reason_str("reason 3")

        env_mock = EnvironmentMock()
        agent_mock = AgentMock()
        agent_factory = AgentFactoryMock(agent_mock)
        mapper = ScenarioMapperMock()

        env_mock.set_obs_returned({"image": "image", "altitude": 10, "collision": 0})

        agent_mock.set_action_to_return({"found": 0, "coordinate_change": (1, 2, 3)})
        evaluator = TrajectoryEvaluator(agent_factory, env_mock, 3, mapper, [logger],
                                        [validator_1, validator_2, validator_3],
                                        155, 10, prompt_func)

        evaluator.evaluate()

        assert len(agent_mock.fail_reasons) == 3
        assert agent_mock.fail_reasons[0] == {"reason": "reason 2"}
        assert agent_mock.fail_reasons[1] == {"reason": "reason 2"}
        assert agent_mock.fail_reasons[2] == {"reason": "reason 2"}

        assert len(env_mock.actions_passed) == 3

        assert len(logger.evaluation_states) == 6  # There were always 2 action attempts per evaluation
        assert len(logger.termination_info) == 1
        assert logger.termination_info[0] == {"reason": "glimpses ran out"}

    def test_terminates_if_out_of_forgiveness(self):
        validator_1 = ValidatorMock()
        validator_2 = ValidatorMock()
        validator_3 = ValidatorMock()

        logger = LoggerMock()

        validator_1.set_fail(False)
        validator_2.set_fail(True)
        validator_3.set_fail(True)

        validator_2.set_fail_modulo(1)  # fail always, haha
        validator_3.set_fail_modulo(1)

        validator_2.set_fail_reason_str("reason 2")
        validator_3.set_fail_reason_str("reason 3")

        env_mock = EnvironmentMock()
        agent_mock = AgentMock()
        agent_factory = AgentFactoryMock(agent_mock)
        mapper = ScenarioMapperMock()

        env_mock.set_obs_returned({"image": "image", "altitude": 10, "collision": 0})

        agent_mock.set_action_to_return({"found": 0, "coordinate_change": (1, 2, 3)})
        evaluator = TrajectoryEvaluator(agent_factory, env_mock, 3, mapper, [logger],
                                        [validator_1, validator_2, validator_3],
                                        155, 3, prompt_func)

        evaluator.evaluate()

        assert len(agent_mock.fail_reasons) == 3
        assert agent_mock.fail_reasons[0] == {"reason": "reason 2"}
        assert agent_mock.fail_reasons[1] == {"reason": "reason 2"}
        assert agent_mock.fail_reasons[2] == {"reason": "reason 2"}

        assert len(env_mock.actions_passed) == 0

        assert len(logger.evaluation_states) == 3  # 3 times informed
        assert len(logger.termination_info) == 1
        assert logger.termination_info[0] == {"reason": "validation forgiveness ran out"}

    def test_terminates_if_environment_told_it_so(self):
        env_mock = EnvironmentMock()
        agent_mock = AgentMock()
        agent_factory = AgentFactoryMock(agent_mock)
        mapper = ScenarioMapperMock()
        logger = LoggerMock()

        env_mock.set_return_terminated(True)
        agent_mock.set_action_to_return({"found": 0, "coordinate_change": (1, 2, 3)})

        evaluator = TrajectoryEvaluator(agent_factory, env_mock, 3, mapper, [logger], [], 155, 3, prompt_func)

        evaluator.evaluate()

        assert len(env_mock.actions_passed) == 1
        assert len(logger.termination_info) == 1
        assert logger.evaluation_states[0].agent_info == {"eee": 42}
        assert logger.termination_info[0] == {"reason": "found claimed"}
