import asyncio
import nest_asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

# Apply nest_asyncio to allow nesting of the event loop
nest_asyncio.apply()

# Initialize Selenium WebDriver
def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
    chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    return driver

async def fetch_theoretical_opening_price(ticker):
    driver = init_driver()
    url = f'https://live.euronext.com/en/product/equities/{ticker}'
    try:
        driver.get(url)
        try:
            # Wait for the necessary element to load
            span = WebDriverWait(driver, 10).until(
                # EC.presence_of_element_located((By.ID, 'header-instrument-price'))
                EC.presence_of_element_located((By.XPATH, "//span[contains(text(), \"Cours Théorique d'Ouverture\")]/following-sibling::span"))
            )
            price_str = span.text.strip().replace(',', '.')
            theoretical_opening_price = float(price_str)
        except Exception as e:
            print(f"Error fetching price for {ticker}: {e}")
            theoretical_opening_price = None
        return ticker, theoretical_opening_price
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return ticker, None
    finally:
        driver.quit()

async def semaphore_fetch(semaphore, ticker):
    async with semaphore:
        return await fetch_theoretical_opening_price(ticker)

# Function to fetch opening prices for a list of tickers with progress tracking
async def fetch_theoretical_opening_prices(tickers, max_concurrent_tasks=10):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    tasks = [semaphore_fetch(semaphore, ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    return results

# Function to be called from another file
def get_theoretical_opening_prices(tickers, max_concurrent_tasks=10):
    ticker_data = asyncio.run(fetch_theoretical_opening_prices(tickers, max_concurrent_tasks))

    df = pd.DataFrame(ticker_data, columns=['ticker', 'euronext_theor_open_price'])

    return df

if __name__ == "__main__":
    file_path = './db/tickers_euronext_regulated_euro_500k€.xlsx'

    df = pd.read_excel(file_path)
    tickers = df['euronext'].iloc[1:6].values.tolist()

    print(tickers)

    df_prices = get_theoretical_opening_prices(tickers)

    pd.set_option('display.max_rows', None)
    print(df_prices)