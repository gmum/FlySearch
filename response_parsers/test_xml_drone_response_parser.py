from response_parsers import XMLDroneResponseParser


class TestXMLDroneResponseParser:
    def test_if_there_are_2_actions_first_is_picked(self):
        response = "<Action> (0, 0, 0) </Action> <Action> (1, 1, 1) </Action>"
        parser = XMLDroneResponseParser()
        assert parser.get_basic_string(response) == "(0, 0, 0)"
