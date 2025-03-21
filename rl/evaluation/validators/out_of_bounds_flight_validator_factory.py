from rl.evaluation.validators import BaseValidatorFactory, OutOfBoundsFlightValidator


class OutOfBoundsFlightValidatorFactory(BaseValidatorFactory):
    def __init__(self, search_diameter: int = 200):
        self.search_diameter = search_diameter

    def get_validator(self):
        return OutOfBoundsFlightValidator(self.search_diameter)
