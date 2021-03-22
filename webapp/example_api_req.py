import requests 

# api-endpoint
URL = "http://127.0.0.1:5000/api/"

# location given here
query = "hi"
  
# defining a params dict for the parameters to be sent to the API
PARAMS = {'q': query} 

# sending get request and saving the response as response object
r = requests.get(url = URL, params = PARAMS) 

# extracting data in json format
data = r.json()['response']

print(data)

