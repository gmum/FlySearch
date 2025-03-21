from rl.evaluation.validators import BaseValidatorFactory, BaseValidator, AltitudeValidator


class AltitudeValidatorFactory(BaseValidatorFactory):
    def __init__(self, max_altitude: int):
        self.max_altitude = max_altitude

    def get_validator(self) -> BaseValidator:
        return AltitudeValidator(self.max_altitude)
