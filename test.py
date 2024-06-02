import requests

resp = requests.post("http://localhost:5000/", files={'file': open('sample.JPG', 'rb')})

print(resp.json())