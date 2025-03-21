from rl.evaluation.loggers import BaseLogger


class BaseLoggerFactory:

    def get_logger(self) -> BaseLogger:
        raise NotImplementedError()
