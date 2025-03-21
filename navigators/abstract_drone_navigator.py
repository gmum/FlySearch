class RecklessFlyingException(Exception):
    pass


class AbstractDroneNavigator:
    def get_new_position(self, current_position: tuple[float, float, float], response: str, throw_if_reckless=False) -> \
            tuple[
                float, float, float]:
        pass

    def get_diffs(self, response: str) -> tuple[int, int, int]:
        pass

    def get_claim(self, response: str) -> bool:
        pass
