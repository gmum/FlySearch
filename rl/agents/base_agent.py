from typing import Dict


class BaseAgent:
    """
    Base class for agents in FlySearch environments.
    """

    def sample_action(self, observation: Dict) -> Dict:
        """
        Samples an action given an observation.

        Args:
            observation: The observation presented to the agent.

        Returns:
            The sampled action.
        """
        raise NotImplementedError()

    def correct_previous_action(self, fail_reason: Dict) -> Dict:
        """
        In FlySearch, some actions can fail if they are absurd, but syntactically correct. For example, flying lots of meters up when the drone is already at the maximum altitude or flying out of the search area. Agents should implement this method to handle such cases.

        Note that it is possible to correct previous correction if it failed as well. Classes using this method should probably set a limit on the maximum number of corrections to avoid infinite loops.

        Args:
            fail_reason: The reason why the previous action failed.

        Returns:
            The corrected action. It overrides the previous action.
        """
        raise NotImplementedError()

    def get_agent_info(self) -> Dict:
        """
        Returns additional information about the agent.

        Returns:
            Information about the agent.
        """
        return {}
