import numpy as np

from bs4 import BeautifulSoup as bs
from requests_html import HTMLSession
from urllib.request import urlopen

url_string = 'https://www.offtrackbetting.com/results/73/remington-park-{yyyymmdd}.html'
base_date = 20200523

# Open HTML session to grab webpage, render the javascript, and create the 
# BeautifulSoup object for later parsing
session = HTMLSession()
html_response = session.get(url_string.format(yyyymmdd = base_date))
html_response.html.render(timeout=32)
html_string = bs(html_response.html.html, 'html.parser')
session.close()

races = html_string.find_all('div', id='finishers')

rows = [['date', 'track', 'race', 'number', 'name', 'jockey', 'win', 'place', 'show']]

# Loop through all races on the page, find the finishers table, parse the data,
# and save to the data table
for i in range(len(races)):
    finishers_rows = races[i].find('table').find_all('tr')
    for row in finishers_rows[1:]:
        data = row.find_all('td')
        if len(data) == 0:
            continue

        number = data[0].getText()
        name = data[1].getText()
        jockey = data[2].getText()
        win = data[-3].getText().replace('$','')
        if win == '': win = '0.00'
        place = data[-2].getText().replace('$', '')
        if place == '': place = '0.00'
        show = data[-1].getText().replace('$', '')
        
        rows.append([base_date, 'Remington Park', i, number, name, jockey, win, place, show])

rows = np.array(rows)
with open('data/remingtonParkData.npy', 'wb') as outfile:
    np.save(outfile, rows)
