"""Send one student to the served model and print the prediction."""
import json
import requests

URL = "http://127.0.0.1:5001/invocations"

student = {
    "age": 18,
    "gender": "male",
    "category": "general",
    "state": "gujarat",
    "preferred_stream": "engineering",
    "entrance_exam": "jee",
    "entrance_score": 250,
    "board_percentage": 92.5,
    "extracurricular_score": 8,
}

payload = {"dataframe_split": {
    "columns": list(student.keys()),
    "data": [list(student.values())],
}}

response = requests.post(URL, json=payload)
print("status:", response.status_code)
print("response:", json.dumps(response.json(), indent=2))