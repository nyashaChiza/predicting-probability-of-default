
def map_loan_status(status):
    status_map = {
        'Did not default': False,
        'Defaulted': True
    }
    return status_map.get(status, None)