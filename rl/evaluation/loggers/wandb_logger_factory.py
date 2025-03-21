from typing import Optional

from rl.evaluation.loggers import BaseLoggerFactory, BaseLogger, WandbLogger


class WandbLoggerFactory(BaseLoggerFactory):
    def __init__(self, project_name: str, run_name: Optional[str] = None):
        self.project_name = project_name
        self.run_name = run_name

    def get_logger(self) -> WandbLogger:
        return WandbLogger(project_name=self.project_name, run_name=self.run_name)
