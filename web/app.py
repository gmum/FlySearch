import base64
import json
import os
import pathlib
import random
import sys
import time
from typing import Optional

import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, '../')

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from rl.environment import MockFlySearchEnv, DroneCannotSeeTargetException, CityFlySearchEnv
from scenarios import DefaultCityScenarioMapper

app = FastAPI()
env = CityFlySearchEnv(throw_if_hard_config=False, max_altitude=250)
csm = DefaultCityScenarioMapper(drone_alt_max=250, drone_alt_min=200)
fs1 = False

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


object_bbox_str = ""
consecutive_failures = 0
last_observation = None
move_counter = 0
coordinates = []
opencv_images = []
actions = []
current_scenario = None
log_path = pathlib.Path(__file__).parent / "trajectories"
log_path.mkdir(parents=True, exist_ok=True)
current_client_uuid = None
current_client_name = None
last_ping = None


def log_info_at_finish():
    global move_counter
    global coordinates
    global log_path
    global current_scenario
    global actions
    global object_bbox_str

    total_trajectory_count = len(os.listdir(log_path))
    trajectory_name = f"{total_trajectory_count}_r0"
    trajectory_path = log_path / trajectory_name

    trajectory_path.mkdir(parents=False, exist_ok=False)

    for i, coordinates in enumerate(coordinates):
        with open(trajectory_path / f"{i}_coords.txt", "w") as f:
            f.write(str(tuple(coordinates.tolist())))

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

    with open(trajectory_path / "user.txt", "w") as f:
        f.write(str(current_client_name))

    # with open(trajectory_path / "conversation.json", "w") as f:
    #    json.dump([], f, indent=4)

    simple_conversation = []

    for (found, coord_change) in actions:
        simple_conversation.append(["user", "image"])
        simple_conversation.append(["assistant", f"FOUND: {found}, COORD_CHANGE: {coord_change}"])

    with open(trajectory_path / "simple_conversation.json", "w") as f:
        json.dump(simple_conversation, f, indent=4)

    for i, opencv_image in enumerate(opencv_images):
        cv2.imwrite(str(trajectory_path / f"{i}.png"), opencv_image)


class Observation(BaseModel):
    image_b64: str
    altitude: int
    collision: bool


class Action(BaseModel):
    found: bool
    coordinate_change: tuple[int, int, int]


class GenerateNewRequest(BaseModel):
    is_fs1: bool


@app.get("/get_observation", status_code=200)
async def get_observation(client_uuid: str, response: Response) -> Optional[Observation]:
    global last_observation
    global current_client_uuid

    if client_uuid != current_client_uuid:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return None

    if last_observation is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return None

    opencv_image = last_observation["image"]
    altitude = last_observation["altitude"]
    collision = last_observation["collision"]

    base64_image = cv2.imencode('.jpeg', opencv_image)[1].tobytes()
    base64_image = base64.b64encode(base64_image).decode('utf-8')
    collision = collision == 1

    return Observation(image_b64=base64_image, altitude=altitude, collision=collision)


@app.post("/move", status_code=200)
async def move(client_uuid: str, action: Action, response: Response):
    global last_observation
    global move_counter
    global actions
    global coordinates
    global consecutive_failures
    global current_client_uuid

    if client_uuid != current_client_uuid:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return

    if last_observation is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return

    # consecutive_failures = 0

    move_counter += 1
    moves_left = 10 - move_counter

    found = action.found

    if not found:
        current_coords = coordinates[-1]
        current_coords = np.array(current_coords)
        coordinate_change = action.coordinate_change
        flipped_coordinate_change = (coordinate_change[0], -coordinate_change[1], coordinate_change[
            2])  # We invert the y axis here. Normally the environment does that for us, but we perform the calculation alongside it, hence requiring us to do this... stuff.
        new_coords = current_coords + np.array(flipped_coordinate_change)
        dronecentric_coords = new_coords - coordinates[0]
        abs_coords = np.abs(dronecentric_coords)

        x, y = abs_coords[0], abs_coords[1]
        alt = new_coords[2]

        if x > 200 or y > 200 or alt > 300:
            response.status_code = status.HTTP_400_BAD_REQUEST

            if x > 200 or y > 200:
                return {"user_error": "Coordinates out of bounds"}

            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"user_error": "Altitude out of bounds"}

    if moves_left < 0:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return

    coordinate_change = action.coordinate_change

    action_dict = {
        "found": found,
        "coordinate_change": coordinate_change
    }

    obs, _, _, _, info = env.step(action_dict)
    actions.append((found, coordinate_change))

    if moves_left == 0 and not found:
        log_info_at_finish()
        response.status_code = status.HTTP_400_BAD_REQUEST
        return

    last_observation = obs

    real_position = info["real_position"]
    coordinates.append(real_position)

    if obs and "image" in obs:
        opencv_image = obs["image"]
        opencv_images.append(opencv_image)

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

    object_visible = min_x < 0 < max_x and min_y < 0 < max_y

    if found:
        log_info_at_finish()

        if alt_diff_ok and object_visible:
            return {"success": True}
        else:
            return {"success": False}

    if moves_left == 0:
        log_info_at_finish()

    return {
        "moves_left": moves_left,
    }


@app.post("/generate_new", status_code=201)
async def generate_new(client_uuid: str, request: GenerateNewRequest, response: Response):
    global last_observation
    global object_bbox_str
    global move_counter
    global coordinates
    global current_scenario
    global opencv_images
    global actions
    global fs1
    global csm
    global current_client_uuid

    if client_uuid != current_client_uuid:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return

    fs1 = request.is_fs1

    if fs1:
        csm = DefaultCityScenarioMapper(drone_alt_min=30, drone_alt_max=100)
        env.set_throw_if_hard_config(True)
    else:
        csm = DefaultCityScenarioMapper(drone_alt_min=200, drone_alt_max=250)
        env.set_throw_if_hard_config(False)

    retry = 25

    while retry > 0:
        try:
            seed = random.randint(0, int(1e12))
            scenario = csm.create_random_scenario(seed)
            current_scenario = scenario
            last_observation, info = env.reset(options=scenario)
            break
        except DroneCannotSeeTargetException as e:
            retry -= 1
            if retry <= 0:
                raise e

    move_counter = 0
    real_coords = info["real_position"]
    object_bbox_str = info["object_bbox"]
    coordinates = [real_coords]
    opencv_images = [last_observation["image"]]
    actions = []

    return {
        'target': csm.get_description(scenario["object_type"]),
        'moves_left': 10
    }


class PingRequest(BaseModel):
    client_uuid: str
    client_name: str


@app.post("/ping", status_code=200)
async def ping(request: PingRequest, response: Response):
    global current_client_uuid
    global current_client_name
    global last_ping

    current_time = time.time()

    if request.client_uuid == current_client_uuid:
        last_ping = current_time
        print(f"Keep-alive ping from {request.client_name} ({request.client_uuid})", file=sys.stderr)
        return

    if last_ping is not None and current_time - last_ping < 5:
        response.status_code = status.HTTP_409_CONFLICT
        print(f"Client rejected - server busy: {request.client_name}", file=sys.stderr)
        return

    current_client_uuid = request.client_uuid
    current_client_name = request.client_name
    last_ping = current_time
    print(f"New client connected: {request.client_name} ({request.client_uuid})", file=sys.stderr)
    return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
