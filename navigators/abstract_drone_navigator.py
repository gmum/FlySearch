class AbstractDroneNavigator:
    def get_new_position(self, current_position: tuple[float, float, float], response: str) -> tuple[
        float, float, float]:
        pass
