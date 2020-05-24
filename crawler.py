import argparse
import asyncio
import pyppeteer
import numpy as np

from bs4 import BeautifulSoup as bs
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Gather horse race data from offtrackbetting.com')
parser.add_argument('-o', '--outfile', type=str, dest='outfile', help='Name of the date output file', default='')

args = parser.parse_args()
outfile = args.outfile

url_string = 'https://www.offtrackbetting.com/results/73/remington-park-{yyyymmdd}.html'
base_date = 20200522

async def get_page(url, selector):
    browser = await pyppeteer.launch()
    page = await browser.newPage()
    await page.goto(url)
    try:
        await page.waitForSelector(selector, timeout=15000)
    except pyppeteer.errors.TimeoutError:
        print('Could not find tables')
    retval = await page.content()
    await browser.close()
    return retval

page_data = asyncio.get_event_loop().run_until_complete(get_page(url_string.format(yyyymmdd = base_date), 'td.postposition'))

html_string = bs(page_data, 'html.parser')

races = html_string.find_all('div', id='finishers')

data = [['date', 'track', 'race', 'number', 'name', 'jockey', 'win', 'place', 'show']]

# Loop through all races on the page, find the finishers table, parse the data,
# and save to the data table
for i in range(len(races)):
    finishers_rows = races[i].find('table').find_all('tr')
    for row in finishers_rows[1:]:
        row_data = row.find_all('td')
        if len(row_data) == 0:
            continue

        number = row_data[0].getText()
        name = row_data[1].getText()
        jockey = row_data[2].getText()
        win = row_data[-3].getText().replace('$','')
        if win == '': win = '0.00'
        place = row_data[-2].getText().replace('$', '')
        if place == '': place = '0.00'
        show = row_data[-1].getText().replace('$', '')
        
        data.append([base_date, 'Remington Park', i, number, name, jockey, win, place, show])

data = np.array(data)
with open(outfile if outfile != '' else 'data/output.npy', 'wb') as output:
    np.save(output, data)
