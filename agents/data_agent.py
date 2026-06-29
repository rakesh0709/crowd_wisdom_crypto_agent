import os
import sys
# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from config import APIFY_TOKEN

logger = logging.getLogger("crypto_agent.data")

class HistoricalDataAgent:
    """
    Data agent that fetches historical price data (OHLCV bars) for crypto assets.
    Supports Apify (via crawlerbros/binance-price-scraper) and falls back to
    direct Binance REST API for real-time and resilient operations.
    """

    def __init__(self, apify_token: str = APIFY_TOKEN):
        self.apify_token = apify_token
        self.binance_klines_url = "https://api.binance.com/api/v3/klines"

    def fetch_via_apify(self, symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Attempts to fetch historical price data using the Apify actor 'crawlerbros/binance-price-scraper'.
        """
        if not self.apify_token or "your_apify_token" in self.apify_token:
            logger.warning("APIFY_TOKEN is not configured. Skipping Apify method.")
            return None

        logger.info(f"Fetching {symbol} ({interval}) data via Apify...")
        try:
            from apify_client import ApifyClient
            client = ApifyClient(self.apify_token)

            # Map interval to what the scraper supports
            # Map symbol (e.g. BTC) to Binance pair (BTCUSDT)
            binance_pair = f"{symbol}USDT" if "USDT" not in symbol else symbol
            
            # For 1000 bars, calculate start date
            # e.g., 1000 bars of 5m is ~3.5 days. Let's start 5 days ago to be safe.
            now = datetime.utcnow()
            if "m" in interval:
                minutes = int(interval.replace("m", ""))
                delta = timedelta(minutes=minutes * limit)
            elif "h" in interval:
                hours = int(interval.replace("h", ""))
                delta = timedelta(hours=hours * limit)
            else:
                delta = timedelta(days=limit)

            # Binance historical archives are only uploaded for completed days.
            # Make sure we start at least 5 days ago to fetch completed archives.
            lookback_days = max(5, delta.days if hasattr(delta, "days") else 5)
            start_date = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            run_input = {
                "symbols": [binance_pair],
                "interval": interval,
                "startDate": start_date,
                "marketType": "spot"
            }

            logger.info(f"Running Apify actor crawlerbros/binance-price-scraper with input: {run_input}")
            # Call the actor
            run = client.actor("crawlerbros/binance-price-scraper").call(run_input=run_input)
            
            # Fetch results from dataset
            run_id = run.get("id") if isinstance(run, dict) else getattr(run, "id", "Unknown")
            logger.info(f"Apify actor run completed. Run ID: {run_id}. Fetching dataset items...")
            dataset_items = client.dataset(run["defaultDatasetId"] if isinstance(run, dict) else getattr(run, "default_dataset_id", "")).list_items().items
            
            if not dataset_items:
                logger.warning("Apify dataset returned no items.")
                return None

            # Parse results
            # The crawlerbros output format typically returns objects with:
            # symbol, interval, open, high, low, close, volume, openTime, closeTime, etc.
            df = pd.DataFrame(dataset_items)
            if df.empty:
                return None

            # Clean and format columns
            df['timestamps'] = pd.to_datetime(df['openTime'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['amount'] = df.get('quoteAssetVolume', df['open'] * df['volume']).astype(float)
            
            # Sort and select required columns
            df = df.sort_values('timestamps').tail(limit).reset_index(drop=True)
            return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]

        except ImportError:
            logger.warning("apify-client package not installed. Install via pip install apify-client.")
        except Exception as e:
            logger.error(f"Apify scraping failed: {str(e)}")
            
        return None

    def fetch_via_binance_rest(self, symbol: str, interval: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Direct REST API call to Binance public endpoint for instant, real-time candles.
        """
        logger.info(f"Fetching {symbol} ({interval}) data directly from Binance REST API...")
        binance_pair = f"{symbol}USDT" if "USDT" not in symbol else symbol
        
        # Binance REST limits klines to 1000 per request
        params = {
            "symbol": binance_pair,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        try:
            response = requests.get(self.binance_klines_url, params=params, timeout=10)
            if response.status_code != 200:
                logger.error(f"Binance API returned HTTP {response.status_code}")
                return None
                
            data = response.json()
            # Format according to Kronos requirements
            # 0: Open time (ms), 1: Open, 2: High, 3: Low, 4: Close, 5: Volume, 7: Quote asset volume (amount)
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['amount'] = df['quote_asset_volume'].astype(float)
            
            return df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            
        except Exception as e:
            logger.error(f"Binance REST API fetch failed: {str(e)}")
            
        return None

    def get_data(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Main interface: Tries Apify first, then falls back to direct Binance REST API.
        """
        # Try Apify
        df = self.fetch_via_apify(symbol, interval, limit)
        if df is not None and not df.empty:
            logger.info("Successfully fetched data via Apify.")
            return df
            
        # Fall back to direct Binance REST
        logger.info("Falling back to direct Binance REST API...")
        df = self.fetch_via_binance_rest(symbol, interval, limit)
        if df is not None and not df.empty:
            logger.info("Successfully fetched data via Binance REST API.")
            return df
            
        raise ValueError(f"Failed to fetch historical data for {symbol} ({interval}) via all methods.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = HistoricalDataAgent()
    try:
        df = agent.get_data("BTC", "5m", 5)
        print("Sample data fetched:")
        print(df)
    except Exception as e:
        print(f"Error: {e}")
