import base64
import pathlib
import os
import random
import sys
import json

from typing import Optional

import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, '../')

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from rl.environment import CityFlySearchEnv, MockFlySearchEnv, DroneCannotSeeTargetException
from scenarios import DefaultCityScenarioMapper

app = FastAPI()
env = MockFlySearchEnv()
csm = DefaultCityScenarioMapper(drone_alt_max=250, drone_alt_min=200)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# yes, I know
env.__enter__()


@app.get("/")
async def root():
    return {"message": "Hello World"}


last_observation = None
move_counter = 0
coordinates = []
current_scenario = None
log_path = pathlib.Path(__file__).parent / "trajectories"
log_path.mkdir(parents=True, exist_ok=True)


def log_info_at_finish(object_bbox_str: str):
    global move_counter
    global coordinates
    global log_path
    global current_scenario

    total_trajectory_count = len(os.listdir(log_path))
    trajectory_name = f"{total_trajectory_count}_r0"
    trajectory_path = log_path / trajectory_name

    trajectory_path.mkdir(parents=False, exist_ok=False)

    for i, coordinates in enumerate(coordinates):
        with open(trajectory_path / f"{i}_coords.txt", "w") as f:
            f.write(str(tuple(coordinates.tolist())))

        # Compliance with the previous standard for doing things
        with open(trajectory_path / f"{i}.txt", "w") as f:
            f.write("")

    with open(trajectory_path / "scenario_params.json", "w") as f:
        def denumpifier(obj):
            if isinstance(obj, tuple):
                return tuple([float(x) for x in obj])
            else:
                return obj

        current_scenario = {k: str(denumpifier(v)) for k, v in current_scenario.items()}

        json.dump(current_scenario, f, indent=4)

    with open(trajectory_path / "object_bbox.txt", "w") as f:
        f.write(object_bbox_str)

    with open(trajectory_path / "conversation.json", "w") as f:
        json.dump([], f, indent=4)


class Observation(BaseModel):
    image_b64: str
    altitude: int
    collision: bool


class Action(BaseModel):
    found: bool
    coordinate_change: tuple[int, int, int]


@app.get("/get_observation", status_code=200)
async def get_observation(response: Response) -> Optional[Observation]:
    global last_observation

    if last_observation is None:
        response.status_code = status.HTTP_409_CONFLICT
        return None

    opencv_image = last_observation["image"]
    altitude = last_observation["altitude"]
    collision = last_observation["collision"]

    base64_image = cv2.imencode('.jpeg', opencv_image)[1].tobytes()
    base64_image = base64.b64encode(base64_image).decode('utf-8')
    collision = collision == 1

    return Observation(image_b64=base64_image, altitude=altitude, collision=collision)


@app.post("/move", status_code=200)
async def move(action: Action, response: Response):
    global last_observation
    global move_counter

    if last_observation is None:
        response.status_code = status.HTTP_409_CONFLICT
        return

    move_counter += 1
    moves_left = 10 - move_counter

    if moves_left < 0:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return

    found = action.found
    coordinate_change = action.coordinate_change

    action_dict = {
        "found": found,
        "coordinate_change": coordinate_change
    }

    obs, _, _, _, info = env.step(action_dict)

    last_observation = obs

    real_position = info["real_position"]
    coordinates.append(real_position)

    object_bbox_str = info["object_bbox"]
    object_bbox = object_bbox_str.split()

    user_alt = real_position[2]

    # FIXME
    try:
        object_max_z = int(object_bbox[5]) // 100
        alt_diff_ok = abs(user_alt - object_max_z) <= 10
    except:
        print("Parsing object bbox failed; assuming max object altitude is 0")
        print("Object bbox:", object_bbox)
        alt_diff_ok = user_alt <= 10

    x, y, z = real_position

    max_x = x + z
    min_x = x - z

    max_y = y + z
    min_y = y - z

    object_visible = min_x < 0 and max_x > 0 and min_y < 0 and max_y > 0

    if found:
        log_info_at_finish(object_bbox_str)

        if alt_diff_ok and object_visible:
            return {"success": True}
        else:
            return {"success": False}

    if moves_left == 0:
        log_info_at_finish(object_bbox_str)

    return {
        "moves_left": moves_left,
    }


@app.post("/generate_new", status_code=201)
async def generate_new(response: Response):
    global last_observation
    global move_counter
    global coordinates
    global current_scenario

    failed = True

    while failed:
        try:
            seed = random.randint(0, int(1e12))
            scenario = csm.create_random_scenario(seed)
            current_scenario = scenario
            last_observation, info = env.reset(options=scenario)
            failed = False
        except DroneCannotSeeTargetException as e:
            pass

    move_counter = 0
    real_coords = info["real_position"]
    coordinates = [real_coords]

    return {
        'target': csm.get_description(scenario["object_type"]),
        'moves_left': 10
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
