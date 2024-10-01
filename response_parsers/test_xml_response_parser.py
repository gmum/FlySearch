from response_parsers.xml_response_parser import XMLResponseParser


class TestXmlResponseParser:
    def test_get_coordinates_fails_on_invalid_response(self):
        response = "<Request>(0.5) and (0.7, 0.8)</Request>"
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) is None
        assert parser.is_invalid(response)

    def test_get_coordinates_fails_on_invalid_response_2(self):
        response = "<Request>(0.5, 0.5, 0.6) and (0.7, 0.3)</Request>"
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) is None
        assert parser.is_invalid(response)

    def test_get_coordinates_fails_on_invalid_response_3(self):
        response = "<Request>(0.5, 0.5 0.6) and (0.7, 0.3 0.6)</Request>"
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) is None
        assert parser.is_invalid(response)

    def test_get_coordinates_fails_on_invalid_response_4(self):
        response = "<Request>0.3 0.6 0.0 0.7</Request>"
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) is None
        assert parser.is_invalid(response)

    def test_get_coordinates_fails_on_invalid_response_5(self):
        response = "<Request>0.3 0.6 and 0.0 0.7</Request>"
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) is None

    def test_requests_are_case_invariant(self):
        response = "<Request>(0.5, 0.5) and (0.7, 0.8)</Request>"
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) == (0.5, 0.5, 0.7, 0.8)
        assert parser.is_coordinate_request(response)
        assert not parser.is_invalid(response)
        assert not parser.is_answer(response)

    def test_requests_are_case_invariant_2(self):
        response = "<REQUEST>(0.5, 0.5) aNd (0.7, 0.8)</RequesT>"
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) == (0.5, 0.5, 0.7, 0.8)
        assert not parser.is_invalid(response)
        assert not parser.is_answer(response)

    def test_answer_is_lower_case(self):
        response = "<AnsWeR>THIS SHOULD BE LOWER CASE</ANSweR>"
        parser = XMLResponseParser()
        assert parser.get_answer(response) == "this should be lower case"
        assert parser.is_answer(response)
        assert not parser.is_invalid(response)
        assert not parser.is_coordinate_request(response)

    def test_if_something_is_request_and_answer_then_it_is_coordinates(self):
        response = "<ANSWER>HAHAHAHA<RequEsT>(0.3, 0.5) and (0.87, 0.6)</REQUEST></anSwEr>"
        parser = XMLResponseParser()

        assert parser.get_coordinates(response) == (0.3, 0.5, 0.87, 0.6)

        assert parser.is_answer(response) is False
        assert parser.is_coordinate_request(response) is True
        assert parser.is_invalid(response) is False

    def test_request_is_properly_picked_from_string(self):
        response = (
            "asdaasdafasdgasrfweerkjbngkjlANEDJKASD\n\n\n\n\nNASJKDNASJKLnkjqwr\nnqkljwrqnweq<Request>(0.3, 0.5) and (0.87, 0.6)</Request>SAKLDASDKLMASD\n\n\n\n\n\nsdafasaggaf")
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) == (0.3, 0.5, 0.87, 0.6)
        assert parser.is_coordinate_request(response)
        assert not parser.is_invalid(response)
        assert not parser.is_answer(response)

    def test_request_tag_can_be_in_multiple_lines(self):
        response = "<R\ne\nq\nuest>\n(0.5, 0.5) and (0.7, 0.8)\n</R\n\n\n\n\nEQuest>"
        parser = XMLResponseParser()
        assert parser.get_coordinates(response) == (0.5, 0.5, 0.7, 0.8)
        assert parser.is_coordinate_request(response)
        assert not parser.is_invalid(response)
        assert not parser.is_answer(response)

    def test_answer_tag_can_be_in_multiple_lines(self):
        response = ("<A\nn\ns\nw\ner>\nABC\n\n\n\n\n\n\n\nD\nE</A\n\n\n\n\nn\ns\nw\nER>")
        parser = XMLResponseParser()
        assert parser.get_answer(response) == "abcde"
        assert parser.is_answer(response)
        assert not parser.is_invalid(response)
        assert not parser.is_coordinate_request(response)
