
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from loan_status_mapping import map_loan_status

class TestMapLoanStatus:

    # returns False for 'Did not default'
    def test_returns_false_for_did_not_default(self):
        assert map_loan_status('Did not default') == False

    # returns True for 'Defaulted'
    def test_returns_true_for_defaulted(self):
        assert map_loan_status('Defaulted') == True

    # returns None for an empty string input
    def test_returns_none_for_empty_string(self):
        assert map_loan_status('') is None

    # Add more tests as needed for other cases
    def test_returns_none_for_invalid_input(self):
        assert map_loan_status('Invalid status') is None

    def test_handles_uppercase(self):
        assert map_loan_status('DID NOT DEFAULT') == False

    def test_handles_lowercase(self):
        assert map_loan_status('defaulted') == True
