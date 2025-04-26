from typing import List, Dict

from rl.agents.semantic_units import BaseSemanticSubunit


class SemanticUnit:
    def __init__(self, subunit_list: List[BaseSemanticSubunit]):
        self.subunit_list = subunit_list

    def process_information(self, information: Dict) -> Dict:
        """
        Process the information received from the main agent and return a dictionary with the processed information.
        :param information: Dictionary with the information to be processed.
        :return: Dictionary with the processed information.
        """
        for subunit in self.subunit_list:
            information = subunit.process_information(information)
        return information
