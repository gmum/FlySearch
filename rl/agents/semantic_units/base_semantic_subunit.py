from typing import Dict


class BaseSemanticSubunit:
    def process_information(self, information: Dict) -> Dict:
        """
        Process the information received from the main agent and return a dictionary with the processed information.
        :param information: Dictionary with the information to be processed.
        :return: Dictionary with the processed information.
        """
        raise NotImplementedError
