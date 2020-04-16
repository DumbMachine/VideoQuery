import requests
from bs4 import BeautifulSoup as soup


index_url = "http://lab.nqnwebs.com/descargas/Friends/"
res = requests.get(index_url)

for a in soup(res.text, 'lxml').find_all("a"):
    if "Friends" in a['href']:
        url = "http://lab.nqnwebs.com/descargas/Friends/"+a['href']
        print(url)
        break



r = requests.get(image_url) # create HTTP response object 
  
# send a HTTP request to the server and save 
# the HTTP response in a response object called r 
with open("python_logo.png",'wb') as f: 
  
    # Saving received content as a png file in 
    # binary format 
  
    # write the contents of the response (r.content) 
    # to a new file in binary mode. 
    f.write(r.content) 