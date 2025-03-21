from rl.evaluation.validators import BaseValidatorFactory
from rl.evaluation.validators.reckless_flying_validator import RecklessFlyingValidator


class RecklessFlyingValidatorFactory(BaseValidatorFactory):
    def __init__(self):
        super().__init__()

    def get_validator(self):
        return RecklessFlyingValidator()
