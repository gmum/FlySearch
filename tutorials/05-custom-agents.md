# Custom Agents

FlySearch also allows you to test multiple agentic methods (as long as you implement them yourself ;)). 

## Overview

We provide an interface for agents that allows you to implement new agentic approaches. The interface is represented by the `BaseAgent` class located in `rl/agents/base_agent.py` file. Let's have a look:

```python 
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

```

In general, agent will receive an observation from the environment and should return a valid action (see `01-environment.md`). For ablation purposes, alongside the normal observation we also provide a `cheats` field with `info` from the environment -- for ablation purposes.

We recommend looking into the implementation of a naive agent to see how this interface should be implemented in practice. This agent is called `SimpleLLMAgent`, located in `rl/agents/simple_llm_agent.py`. Note that our `LocalFSLogger` requires `get_agent_info` to return a dict where there exists a key called `conversation_history` (which makes sense, as we want to retain conversation history with the model).

Similarly to conversations, we rely on agent factories to provide us with new agents. Below is the code of `SimpleLLMAgentFactory`:

```python
from conversation import BaseConversationFactory
from rl.agents import SimpleLLMAgent, BaseAgentFactory


class SimpleLLMAgentFactory(BaseAgentFactory):
    def __init__(self, conversation_factory: BaseConversationFactory):
        self.conversation_factory = conversation_factory

    def create_agent(self, prompt: str, **kwargs) -> SimpleLLMAgent:
        return SimpleLLMAgent(
            conversation=self.conversation_factory.get_conversation(),
            prompt=prompt
        )
```

We also provide examples of different, less trivial agents (that unfortunately don't really work well) -- classes like `DescriptionLLMAgent`, `DetectionDrivenDescriptionLLMAgent`, `GeneralistOne` with their respective factories. Furthermore, there exists `DetectionCheaterAgent`, which can be used for ablation testing -- it uses `cheats` field from the observation to overlay a red rectangle on image (in place of the searched object).

## Contributing

If you've implemented your own agent and its respective factory, you can test it in our environment by adding it to `arg_resolvers/agent_factory_resolver.py`:

```python
class AgentFactoryResolver(BaseArgResolver):
    def register_args(self, parser):
        parser.add_argument("--agent", type=str, required=True,
                            choices=["YOUR_AGENT_NAME_GOES_HERE", "simple_llm", "description_llm", "generalist_one",
                                     "detection_driven_description_llm", "detection_cheater_factory", "parsing_error"], )

    def resolve_args(self, args, accumulator):
        if args.agent == "simple_llm":
            agent_factory = SimpleLLMAgentFactory(
                conversation_factory=accumulator["conversation_factory"],
            )
        elif args.agent == "YOUR_AGENT_NAME_GOES_HERE":
            agent_factory = YOUR_AGENT_FACTORY(
                conversation_factory=accumulator["conversation_factory"],
            )
        elif args.agent == "description_llm":
            agent_factory = DescriptionLLMAgentFactory(
                conversation_factory=accumulator["conversation_factory"],
            )
        ...
```

(_Note on the accumulator_: The resolve_args function receives an _accumulator_ dictionary, which holds other components that have already been constructed based on the script's arguments, such as the `conversation_factory` -- this is the design pattern for argument resolving I've came up with while working on this repo, but I think it works pretty well. This knowledge may be useful to you while tinkering with different arguments as well.)

Happy contributing!