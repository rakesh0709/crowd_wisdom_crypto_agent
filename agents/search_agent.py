import logging
import requests
from typing import List, Dict, Any

logger = logging.getLogger("crypto_agent.search")

class CryptoSearchAgent:
    """
    Search agent that discovers active crypto prediction markets on Polymarket and Kalshi
    for Bitcoin (BTC) and Ethereum (ETH).
    """

    def __init__(self):
        self.polymarket_gamma_url = "https://gamma-api.polymarket.com/events"
        self.kalshi_api_url = "https://external-api.kalshi.com/trade-api/v2/markets"

    def fetch_polymarket_markets(self) -> List[Dict[str, Any]]:
        """
        Fetches active markets from Polymarket and filters for BTC/ETH.
        """
        logger.info("Fetching markets from Polymarket Gamma API...")
        results = []
        try:
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100
            }
            response = requests.get(self.polymarket_gamma_url, params=params, timeout=10)
            if response.status_code == 200:
                events = response.json()
                for event in events:
                    title = event.get("title", "").lower()
                    description = event.get("description", "").lower()
                    
                    # Filter for Bitcoin and Ethereum
                    is_btc = "bitcoin" in title or "btc" in title or "bitcoin" in description
                    is_eth = "ethereum" in title or "eth" in title or "ethereum" in description
                    
                    if not (is_btc or is_eth):
                        continue
                        
                    asset = "BTC" if is_btc else "ETH"
                    markets = event.get("markets", [])
                    for m in markets:
                        prices = m.get("outcomePrices", ["0.5", "0.5"])
                        # Parse Yes price as implied probability
                        try:
                            yes_prob = float(prices[0]) if prices else 0.5
                        except Exception:
                            yes_prob = 0.5
                            
                        results.append({
                            "platform": "Polymarket",
                            "asset": asset,
                            "market_id": m.get("id"),
                            "ticker": m.get("slug") or m.get("id"),
                            "title": m.get("question") or event.get("title"),
                            "description": event.get("description", ""),
                            "yes_price": yes_prob,
                            "no_price": 1.0 - yes_prob,
                            "end_time": m.get("endDate"),
                            "original_data": m
                        })
                logger.info(f"Successfully fetched {len(results)} active BTC/ETH markets from Polymarket.")
            else:
                logger.error(f"Failed to fetch Polymarket events: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching Polymarket events: {str(e)}")
            
        return results

    def fetch_kalshi_markets(self) -> List[Dict[str, Any]]:
        """
        Fetches active markets from Kalshi and filters for BTC/ETH.
        """
        logger.info("Fetching markets from Kalshi Trade API...")
        results = []
        try:
            # Get open markets, up to 100
            params = {
                "status": "open",
                "limit": 100
            }
            response = requests.get(self.kalshi_api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                markets = data.get("markets", [])
                for m in markets:
                    ticker = m.get("ticker", "")
                    title = m.get("title", "").lower()
                    
                    # Kalshi crypto markets typically start with KXBTC or KXETH or have Bitcoin/Ethereum in title
                    is_btc = "btc" in ticker.lower() or "bitcoin" in title
                    is_eth = "eth" in ticker.lower() or "ethereum" in title
                    
                    if not (is_btc or is_eth):
                        continue
                        
                    asset = "BTC" if is_btc else "ETH"
                    
                    # Kalshi prices are in cents (0 to 100)
                    yes_ask = m.get("yes_ask", 50)
                    yes_bid = m.get("yes_bid", 50)
                    
                    # Implied probability is mid price or ask price
                    yes_price = ((yes_ask + yes_bid) / 2.0) / 100.0 if (yes_ask and yes_bid) else (yes_ask or 50) / 100.0
                    
                    results.append({
                        "platform": "Kalshi",
                        "asset": asset,
                        "market_id": m.get("ticker"),
                        "ticker": m.get("ticker"),
                        "title": m.get("title"),
                        "description": m.get("subtitle", ""),
                        "yes_price": yes_price,
                        "no_price": 1.0 - yes_price,
                        "end_time": m.get("expiration_time"),
                        "original_data": m
                    })
                logger.info(f"Successfully fetched {len(results)} active BTC/ETH markets from Kalshi.")
            else:
                logger.error(f"Failed to fetch Kalshi markets: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching Kalshi markets: {str(e)}")
            
        return results

    def get_all_markets(self) -> List[Dict[str, Any]]:
        """
        Gathers active markets from both Polymarket and Kalshi.
        """
        polymarket = self.fetch_polymarket_markets()
        kalshi = self.fetch_kalshi_markets()
        return polymarket + kalshi

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    searcher = CryptoSearchAgent()
    markets = searcher.get_all_markets()
    print(f"Total markets found: {len(markets)}")
    for m in markets[:5]:
        print(f"Platform: {m['platform']} | Asset: {m['asset']} | Ticker: {m['ticker']} | Price: {m['yes_price']:.2f}")
