# Environment usage 

## Basics

Here is a minimal example on how to use our environment directly, without using our evaluation scripts:

```python
import json

from dotenv import load_dotenv

from rl.environment import CityFlySearchEnv # (or ForestFlySearchEnv)
from scenarios import parse_scenario
from misc import opencv_to_pil

load_dotenv()

env = CityFlySearchEnv()

with open('run_templates/city-template/42_r0/scenario_params.json', 'r') as json_file:
    scenario_params = json.load(json_file)

scenario_params = parse_scenario(scenario_params)

with env:
    obs, _ = env.reset(options=scenario_params)
    image = obs['image']
    image = opencv_to_pil(image)
    image.show()
```

Example observation in our environment follows this format:

```python
{
            "image": opencv_image,
            "altitude": np.array(z),
            "collision": 1 if crash else 0,
}
```

If `found` is reported, the observation is an empty dict (`{}`).

Furthermore, additional info field follows this format:

```python
{
            "real_position": np.array([x, y, z]),
            "object_bbox": str
}
```

Example action follows this format:

```python 
{
            "found": 0,
            "coordinate_change": np.array([10, 10, 10])
}
```

Here is another example code where we perform a few actions and use `real_position` field of the `info` dict.

```python
import json
import numpy as np

from dotenv import load_dotenv

from rl.environment import CityFlySearchEnv # (or ForestFlySearchEnv)
from scenarios import parse_scenario

load_dotenv()

env = CityFlySearchEnv()

with open('run_templates/city-template/42_r0/scenario_params.json', 'r') as json_file:
    scenario_params = json.load(json_file)

scenario_params = parse_scenario(scenario_params)

real_positions = []

with env:
    _, info = env.reset(options=scenario_params)
    real_position = info["real_position"]
    real_positions.append(real_position)


    _, _, _, _, info = env.step(
        {
            "found": 0,
            "coordinate_change": np.array([10, 10, 10])
        }
    )


    real_position = info["real_position"]
    real_positions.append(real_position)

    _, _, _, _, info = env.step(
        {
            "found": 0,
            "coordinate_change": np.array([-10, -20, 10])
        }
    )

    real_position = info["real_position"]
    real_positions.append(real_position)

print(real_positions[0]) # [-1 -3 60]
print(real_positions[1]) # [9 7 70]
print(real_positions[2]) # [-1 -13 80]
```

## How to obtain scenarios?

TODO 

## Important info 

If you are using our environment directly you need to handle the `UnrealDiedException` during the requests to the environment. You do not need to restart it _but_ this exception is meant to communicate to you that the UE5's internal state during your experiment was _lost_, and you need to restore it appropriately (probably by calling the `reset` method and then restarting a trajectory evaluation).

If you are using our testing script, you don't need to worry about that because handling of those situations is implemented by us (inside the `TrajectoryEvaluator` class).
