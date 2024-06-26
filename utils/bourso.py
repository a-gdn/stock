import aiohttp
import asyncio
import nest_asyncio

from bs4 import BeautifulSoup
import pandas as pd


# Apply nest_asyncio to allow nesting of the event loop
nest_asyncio.apply()

# Function to fetch and convert the opening price for a given ticker
async def fetch_opening_price(session, ticker):
    url = f'https://www.boursorama.com/cours/{ticker}/'
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                span = soup.find('span', class_='c-instrument c-instrument--open')
                if span:
                    price_str = span.text.strip().replace(',', '.').replace(' ', '')
                    opening_price = float(price_str)
                else:
                    opening_price = None
                return ticker, opening_price
            else:
                return ticker, None
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return ticker, None

# Function to fetch opening prices for a list of tickers
async def fetch_opening_prices(tickers):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_opening_price(session, ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks)
        return results

# Function to be called from another file
def get_opening_prices(tickers):
    # Run the asynchronous fetching function
    ticker_data = asyncio.run(fetch_opening_prices(tickers))

    # Create a DataFrame from the fetched data
    df = pd.DataFrame(ticker_data, columns=['Ticker', 'Opening Price'])

    return df

def convert_yahoo_to_bourso_tickers(tickers):
    def convert_yahoo_to_bourso_ticker(ticker):
        parts = ticker.rsplit('.', 1)
        
        if len(parts) == 2:
            symbol, suffix = parts
        else:
            return None
        
        if suffix == 'BR':
            return f'FF11-{symbol}'
        elif suffix == 'MC':
            return f'FF55-{symbol}'
        elif suffix == 'MI':
            return f'1g{symbol}'
        elif suffix == 'AS':
            return f'1rA{symbol}'
        elif suffix == 'PA':
            return f'1rP{symbol}'
        else:
            return f'{ticker} not converted'
    
    return [convert_yahoo_to_bourso_ticker(ticker) for ticker in tickers]
