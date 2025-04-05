import base64
import sys
from typing import Optional

import cv2
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, '../')

from fastapi import FastAPI, Response, status
from pydantic import BaseModel

from rl.environment import CityFlySearchEnv, MockFlySearchEnv
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

    if last_observation is None:
        response.status_code = status.HTTP_409_CONFLICT
        return

    found = action.found
    coordinate_change = action.coordinate_change

    action_dict = {
        "found": found,
        "coordinate_change": coordinate_change
    }

    obs, *_ = env.step(action_dict)

    last_observation = obs


@app.post("/generate_new", status_code=201)
async def generate_new(response: Response):
    global last_observation

    last_observation, *_ = env.reset(options=csm.create_random_scenario(seed=0))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
