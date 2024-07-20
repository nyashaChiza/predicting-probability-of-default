
def bin_age(age):
    import pandas as pd
    
    bins = [0, 18, 30, 40, 50, float('inf')]
    labels = ['0-18', '19-30', '31-40', '41-50', '51+']
    return pd.cut(age, bins=bins, labels=labels, right=False).astype(str)

def bin_salary(salary):
    import pandas as pd
    salary_bins = [0, 2273.93, 2665.44, 3146.58, 10000]
    salary_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    return pd.cut(salary, bins=salary_bins, labels=salary_labels, right=False).astype(str)

def age_bin_transformer(X):
    import pandas as pd    
    return pd.DataFrame(X).apply(lambda col: bin_age(col)).values

def salary_bin_transformer(X):
    import pandas as pd
    return pd.DataFrame(X).apply(lambda col: bin_salary(col)).values