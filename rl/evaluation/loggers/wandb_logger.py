import numpy as np
import wandb

from typing import Dict, Optional

from conversation import Role
from misc import opencv_to_pil
from rl.evaluation import EvaluationState
from rl.evaluation.loggers import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(self, project_name: str, run_name: Optional[str] = None):
        self.project_name = project_name
        self.run_name = run_name

        wandb.finish()

        if run_name is not None:
            wandb.init(project=project_name, name=run_name)
        else:
            wandb.init(project=project_name)

        self.images = []
        self.latest_history = []

    def log(self, evaluation_state: EvaluationState):
        object_class_image: Optional[np.ndarray] = evaluation_state.observation.get("class_image", None)

        if object_class_image is not None and len(self.images) == 0:
            converted = opencv_to_pil(object_class_image)
            self.images.append(converted)

        image = evaluation_state.observation["image"]
        image = opencv_to_pil(image)

        self.images.append(image)
        self.latest_history = evaluation_state.agent_info["conversation_history"]

        # self.table.add_data(wandb.Image(image), evaluation_state.agent_info["conversation_history"][-1][1],
        #                    evaluation_state.correction_number)

    def log_termination(self, termination_info: Dict):
        wandb.log(
            {
                "termination_reason": wandb.Html(termination_info["reason"])
            }
        )

        images = list(self.images)

        def speech_entry_converter(entry):
            if entry[0] == Role.USER:
                content = entry[1]

                if isinstance(content, dict):
                    image = images.pop(0)
                    return wandb.Image(image), None, None
                else:
                    return None, None, content
            else:
                return None, entry[1], None

        history = [speech_entry_converter(entry) for entry in self.latest_history]

        table = wandb.Table(columns=["image", "model_speech", "benchmark_speech"], data=history)
        wandb.log({"model_speeches": table})

    def nuke(self):
        self.images = []
        self.latest_history = []
