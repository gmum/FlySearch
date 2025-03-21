from rl.evaluation.validators import BaseValidator


class BaseValidatorFactory:
    def get_validator(self) -> BaseValidator:
        raise NotImplementedError()
