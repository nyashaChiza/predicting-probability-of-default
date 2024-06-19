
def map_loan_status(status):
    status_map = {
        'did not default': False,
        'defaulted': True
    }
    return status_map.get(str(status).lower(), None)