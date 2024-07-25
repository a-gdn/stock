import aiohttp
import asyncio
import nest_asyncio

from bs4 import BeautifulSoup
import pandas as pd


# Apply nest_asyncio to allow nesting of the event loop
nest_asyncio.apply()

async def fetch_theoretical_opening_price(session, ticker):
    url = f'https://www.boursorama.com/cours/{ticker}'
    try:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                opening_label = soup.find('span', string='Ouverture théorique ')

                if opening_label:
                    span = opening_label.find_next_sibling('span', class_='c-faceplate__indicative-value')
                    
                    if span:
                        price_str = span.text.strip().replace(',', '.').replace(' ', '')
                        theoretical_opening_price = float(price_str)
                    else:
                        theoretical_opening_price = None
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

def clean_tickers(tickers):
    # Remove empty or NaN values from the tickers list
    return [ticker for ticker in tickers if ticker]

# Function to be called from another file
def get_theoretical_opening_prices(tickers):
    tickers = clean_tickers(tickers)

    ticker_data = asyncio.run(fetch_theoretical_opening_prices(tickers))

    df = pd.DataFrame(ticker_data, columns=['ticker', 'bourso_theor_open_price'])

    return df

if __name__ == "__main__":
    file_path = './db/tickers_euronext_regulated_euro_500k€.xlsx'

    df = pd.read_excel(file_path)
    tickers = df['bourso'].iloc[1:len(df)+1].values.tolist()

    print(tickers)

    df_prices = get_theoretical_opening_prices(tickers)

    pd.set_option('display.max_rows', None)
    print(df_prices)