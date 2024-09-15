import requests

url = 'http://127.0.0.1:5000/upload'
files = {'file': open('your_csv_file.csv', 'rb')}

try:
    response = requests.post(url, files=files)
    response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")