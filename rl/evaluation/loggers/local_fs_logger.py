import pathlib
import json

from misc import opencv_to_pil
from rl.evaluation import EvaluationState
from rl.evaluation.loggers import BaseLogger


class LocalFSLogger(BaseLogger):
    def __init__(self, log_dir: pathlib.Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=False)

    def log(self, evaluation_state: EvaluationState):
        class_image = evaluation_state.observation.get("class_image", None)
        modifier = 0

        if class_image is not None:
            class_image = opencv_to_pil(class_image)
            class_image.save(self.log_dir / f"0.png")
            modifier = 1

        image = evaluation_state.observation["image"]
        image = opencv_to_pil(image)
        image.save(self.log_dir / f"{evaluation_state.observation_number + modifier}.png")

        writable_config = {k: str(v) for k, v in evaluation_state.scenario.items()}

        with open(self.log_dir / "scenario_params.json", "w") as f:
            json.dump(writable_config, f)

        with open(self.log_dir / "conversation.json", "w") as f:
            json.dump(evaluation_state.agent_info["conversation_history"], f)

        def speech_entry_converter(entry):
            if type(entry[1]) is dict:
                return entry[0], "image"
            else:
                return entry

        simple_conversation = [speech_entry_converter(entry) for entry in
                               evaluation_state.agent_info["conversation_history"]]

        with open(self.log_dir / "simple_conversation.json", "w") as f:
            json.dump(simple_conversation, f, indent=4)

        current_coords = evaluation_state.info["real_position"]
        glimpse_number = evaluation_state.observation_number

        with open(self.log_dir / f"{glimpse_number}_coords.txt", "w") as f:
            f.write(f"({current_coords[0]}, {current_coords[1]}, {current_coords[2]})")

        with open(self.log_dir / f"object_bbox.txt", "w") as f:
            f.write(f"{evaluation_state.info['object_bbox']}")

    def log_termination(self, termination_info):
        with open(self.log_dir / "termination.txt", "w") as f:
            f.write(termination_info["reason"])
