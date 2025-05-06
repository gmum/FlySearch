import pytest

from response_parsers.xml_response_parser import parse_xml_response, ParsingError


def test_parses_valid_xml_with_found_action():
    response = "<reasoning>Object detected</reasoning><action>found</action>"
    result = parse_xml_response(response)
    assert result.found is True
    assert result.move == (0, 0, 0)


def test_parses_valid_xml_with_move_action():
    response = """<Reasoning>I don't see the crowd in the current view. I'll start by moving towards the center of the grid and decreasing altitude to get a better view.</Reasoning>
<Action>(0, 0, -30)</Action>
"""
    result = parse_xml_response(response)
    assert result.found is False
    assert result.move == (0, 0, -30)


def test_parses_valid_xml_with_move_action2():
    response = """xml
<Reasoning>I need to start searching for the trash pile. I'll start by moving to the right and down a bit to get a better view of the area. I will also descend to get a better look.</Reasoning>
<Action>(36, 0, -20)</Action>
"""

    result = parse_xml_response(response)
    assert result.found is False
    assert result.move == (36, 0, -20)


def test_raises_error_for_invalid_xml_format():
    response = "<reasoning>Invalid</reasoning><action>"
    with pytest.raises(ParsingError, match="Invalid XML response:"):
        parse_xml_response(response)


def test_raises_error_for_invalid_action_format():
    response = "<reasoning>Invalid action</reasoning><action>(1.5, invalid, 0.0)</action>"
    with pytest.raises(ParsingError, match="Invalid action format:"):
        parse_xml_response(response)


def test_handles_empty_response_gracefully():
    response = ""
    with pytest.raises(ParsingError, match="Invalid XML response:"):
        parse_xml_response(response)


def test_ignores_case_in_xml_tags():
    response = "<Reasoning>Case insensitive</Reasoning><Action>found</Action>"
    result = parse_xml_response(response)
    assert result.found is True
    assert result.move == (0, 0, 0)
