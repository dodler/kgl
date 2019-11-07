from bs4 import BeautifulSoup
import requests

url = 'https://badoo.com/dating/taiwan/new-taipei-city/taipei/'
resp = requests.get(url)
html_soup = BeautifulSoup(resp.text, 'html.parser')
print(html_soup.findAll('div', 'folder__content'))
#
