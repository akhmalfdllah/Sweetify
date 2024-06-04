import requests
import os
print (os.getcwd())
resp = requests.post("https://sweetify2-3sj26kulua-as.a.run.app", files={'file': open('test/sample.JPG', 'rb')})


print(resp)