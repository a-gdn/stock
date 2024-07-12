import aiohttp
import asyncio
import nest_asyncio

from bs4 import BeautifulSoup
import pandas as pd


# Apply nest_asyncio to allow nesting of the event loop
nest_asyncio.apply()

async def fetch_theoretical_opening_price(session, ticker):
    url = f'https://bourse.fortuneo.fr/actions/{ticker}'
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                span = soup.find('span', class_='header-devise')
                if span:
                    price_str = span.text.strip().replace(',', '.').replace(' ', '').replace('EUR', '')
                    theoretical_opening_price = float(price_str)
                else:
                    theoretical_opening_price = None
                return ticker, theoretical_opening_price
            else:
                return ticker, None
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return ticker, None

# Function to fetch opening prices for a list of tickers
async def fetch_theoretical_opening_prices(tickers):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_theoretical_opening_price(session, ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        return results

# Function to be called from another file
def get_theoretical_opening_prices(tickers):
    ticker_data = asyncio.run(fetch_theoretical_opening_prices(tickers))

    df = pd.DataFrame(ticker_data, columns=['ticker', 'theoretical_opening_price'])

    return df




file_path = './db/tickers_euronext_regulated_euro_500k€.xlsx'

df = pd.read_excel(file_path)
tickers = df['fortuneo'].iloc[1:872].values.tolist()

print(tickers)

df_prices = get_theoretical_opening_prices(tickers)

pd.set_option('display.max_rows', None)
print(df_prices)