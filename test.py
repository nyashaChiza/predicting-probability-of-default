import requests
import json

url = 'http://127.0.0.1:8000/predict/'

input_str = '{"gender":"female", "disbursement_date":"2022-05-15", "currency":"USD", "country": "Zimbabwe", "is_employed": true, "job": "Software Engineer", "location": "Harare", "loan_amount": 5000, "number_of_defaults": 0, "outstanding_balance": 0, "interest_rate": 5, "age": 30, "remaining_term": 12, "salary": 80000, "marital_status": "single", "loan_status": "paid"}'
data = json.loads(input_str)
# Convert the JSON object to a string

payload = json.dumps(data)

# Print the payload

response = requests.post(url, json=data)
print(response.content)