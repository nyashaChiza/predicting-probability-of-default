import pandas as pd

def map_loan_status(status):
    status_map = {
        'did not default': False,
        'defaulted': True
    }
    return status_map.get(str(status).lower(), None)

# Binning functions
def bin_age(age):
    bins = [0, 18, 30, 40, 50, float('inf')]
    labels = ['0-18', '19-30', '31-40', '41-50', '51+']
    return pd.cut(age, bins=bins, labels=labels, right=False).astype(str)

def bin_salary(salary):
    salary_bins = [0, 2273.93, 2665.44, 3146.58, 10000]
    salary_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    return pd.cut(salary, bins=salary_bins, labels=salary_labels, right=False).astype(str)

# Custom transformer for age binning
def age_bin_transformer(X):
    return pd.DataFrame(X).apply(lambda col: bin_age(col)).values

# Custom transformer for salary binning
def salary_bin_transformer(X):
    return pd.DataFrame(X).apply(lambda col: bin_salary(col)).values
