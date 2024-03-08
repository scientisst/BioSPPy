import json
from bs4 import BeautifulSoup

f = open(r"C:\Users\sofia\Documents\GitHub\BioSPPy\docs\notebooks\A.Setting up Your Tools and Workspace\A003 Arduino - Getting started\test2.ipynb","r")

data = f.read()
jsonObj = json.loads(data)   


cleantext = BeautifulSoup(str(jsonObj), "lxml").text

print(cleantext)

with open(r"C:\Users\sofia\Documents\GitHub\BioSPPy\docs\notebooks\A.Setting up Your Tools and Workspace\A003 Arduino - Getting started\test2.ipynb", 'w') as json_data:
            json.dump(cleantext, json_data)
