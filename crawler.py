import argparse
import asyncio
import json
import pyppeteer
import re
import sys
import numpy as np

from bs4 import BeautifulSoup as bs
from tabulate import tabulate

parser = argparse.ArgumentParser(description='Gather horse race data from offtrackbetting.com')
parser.add_argument('-b', '--base-date', type=int, dest='base_date', help='Base date to start search for all enabled dates, YYYYMMDD format', default=20200522)
parser.add_argument('-s', '--start-date', type=int, dest='start_date', help='Start date for data crawl window, YYYYMMDD format', default=0)
parser.add_argument('-e', '--end-date', type=int, dest='end_date', help='End date for data crawl window, YYYYMMDD format', default=30000000)
parser.add_argument('-o', '--outfile', type=str, dest='outfile', help='Name of the date output file', default='')

args = parser.parse_args()
base_date = args.base_date
start_date = args.start_date
end_date = args.end_date
outfile = args.outfile

if start_date > end_date:
    print('Start date cannot come after end date')
    sys.exit()

url_string = 'https://www.offtrackbetting.com/results/73/remington-park-{yyyymmdd}.html'
track = 'remington_park'

async def get_page(url, selector):
    browser = await pyppeteer.launch()
    page = await browser.newPage()
    await page.goto(url)
    try:
        await page.waitForSelector(selector, timeout=10000)
    except pyppeteer.errors.TimeoutError:
        print('Could not find tables on current date')
    retval = await page.content()
    await browser.close()
    return retval

page_data = asyncio.get_event_loop().run_until_complete(get_page(url_string.format(yyyymmdd = base_date), 'script'))

html_string = bs(page_data, 'html.parser')

scripts = html_string.find_all('script')
regex = re.compile('var enableDays = (.*?);')
for script in scripts:
    array_check = re.search(regex, ' '.join(script.contents)) if len(script.contents) > 0 else None
    if array_check != None:
        enabled_days = json.loads(array_check.groups()[0])
        break

if enabled_days == None:
    print('Could not find enabled days, quitting')
    sys.exit()

data = [['date', 'track', 'race', 'number', 'name', 'jockey', 'win', 'place', 'show']]
enabled_days = [day for day in enabled_days if start_date <= day and day <= end_date]

for day in enabled_days:
    print('Starting day', day)
    page_data = asyncio.get_event_loop().run_until_complete(get_page(url_string.format(yyyymmdd = day), 'td.postposition'))

    html_string = bs(page_data, 'html.parser')
    races = html_string.find_all('div', id='finishers')

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
            
            data.append([day, track, i, number, name, jockey, win, place, show])

data = np.array(data)
with open(outfile if outfile != '' else 'data/' + track + ('_s' + str(start_date) if start_date > 19700101 else '') + ('_e' + str(end_date) if end_date < 29991231 else '') + '.npy', 'wb') as output:
    print('Written to', output.name)
    np.save(output, data)
