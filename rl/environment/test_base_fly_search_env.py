import pytest
import numpy as np

from rl.environment import BaseFlySearchEnv, UnitialisedEnvironmentException


class GlimpseGeneratorMock:
    def __init__(self, relative_from_start_return, camera_image_return):
        self.relative_from_start_return = relative_from_start_return
        self.camera_image_return = camera_image_return

        self.force_move_history = []

    def get_relative_from_start(self):
        return self.relative_from_start_return

    def get_camera_image(self, rel_position_m=(0, 0, 0), force_move=False):
        self.force_move_history.append(force_move)
        return self.camera_image_return


class ClientMock:
    def __init__(self):
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True

    def request(self, *args, **kwargs):
        return {"object_id": "asda"}


class TrivialExtension(BaseFlySearchEnv):
    def __init__(self):
        super().__init__()

        self.configure_called = False

    def get_client(self):
        return ClientMock()

    def _configure(self, options) -> None:
        self.configure_called = True
        self.relative_position = np.array([-10, -10, 100])

    @staticmethod
    def get_glimpse_generator(client):
        return GlimpseGeneratorMock((42, 42, 42), (np.random.randn(50, 50, 3)))


class TestBaseFlySearchEnv:
    def test_reset_calls_configure(self):
        env = TrivialExtension()
        with env:
            env.reset(None, {"seed": 1})

        assert env.configure_called

    def test_reset_returns_observation(self):
        env = TrivialExtension()
        with env:
            obs, info = env.reset(None, {"seed": 1})

        assert "image" in obs
        assert "altitude" in obs
        assert "collision" in obs

        assert obs["altitude"][0] == 42

    def test_reset_throws_error_if_no_seed(self):
        env = TrivialExtension()

        with pytest.raises(ValueError):
            with env:
                env.reset(None, {})

    def test_step_returns_observation(self):
        env = TrivialExtension()

        with env:
            env.reset(None, {"seed": 123})
            obs, reward, done, truncated, info = env.step({"found": 0, "coordinate_change": (1, 1, 1)})

        glimpse_generator = env.glimpse_generator

        assert "image" in obs
        assert "altitude" in obs
        assert "collision" in obs

        assert obs["altitude"][0] == 42
        assert np.allclose(obs["image"], glimpse_generator.camera_image_return[:, :, ::-1])  # PIL -> OpenCV

    def test_step_saves_trajectory_overridden_by_glimpse_gen(self):
        env = TrivialExtension()

        with env:
            env.reset(None, {"seed": 123})
            observation_1, *_ = env.step({"found": 0, "coordinate_change": (1, 1, 1)})
            observation_2, *_ = env.step({"found": 0, "coordinate_change": (1, 5, 8)})
            observation_3, *_ = env.step({"found": 0, "coordinate_change": (5, 5235, 123)})

        assert np.allclose(env.trajectory[0], np.array([42, 42, 42]))
        assert np.allclose(env.trajectory[1], np.array([42, 42, 42]))
        assert np.allclose(env.trajectory[2], np.array([42, 42, 42]))
        assert np.allclose(env.trajectory[3], np.array([42, 42, 42]))

        assert observation_1["collision"] == 0
        assert observation_2["collision"] == 1
        assert observation_3["collision"] == 1

        assert len(env.trajectory) == 4

    def test_returns_empty_observation_if_found(self):
        env = TrivialExtension()

        with env:
            env.reset(None, {"seed": 123})
            obs, reward, done, truncated, info = env.step({"found": 1})

        assert obs == {}
        assert reward == 0.0
        assert done
        assert not truncated

    def test_reset_forces_move(self):
        env = TrivialExtension()

        with env:
            env.reset(None, {"seed": 123})

        assert env.glimpse_generator.force_move_history == [True]

    def test_step_does_not_force_move(self):
        env = TrivialExtension()

        with env:
            env.reset(None, {"seed": 123})
            env.step({"found": 0, "coordinate_change": (1, 1, 1)})
            env.reset(None, {"seed": 125})
            env.step({"found": 0, "coordinate_change": (5, 5, 5)})

        assert env.glimpse_generator.force_move_history == [True, False, True, False]

    def test_throws_error_if_step_called_before_reset(self):
        env = TrivialExtension()

        with pytest.raises(UnitialisedEnvironmentException):
            with env:
                env.step({"found": 0, "coordinate_change": (1, 1, 1)})

    def test_calls_disconnect_on_client(self):
        env = TrivialExtension()

        with env:
            env.reset(None, {"seed": 123})
            env.step({"found": 1})

        assert env.client.disconnected
