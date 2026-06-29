import os
import sys
# Add parent directory to path to allow importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from typing import Dict, Any, List
from config import DEFAULT_FRACTIONAL_KELLY

logger = logging.getLogger("crypto_agent.risk")

class KellyRiskAgent:
    """
    Risk management agent that uses the Kelly Criterion to size positions
    for prediction markets (Polymarket and Kalshi).
    """

    def __init__(self, default_fractional_kelly: float = DEFAULT_FRACTIONAL_KELLY):
        self.default_fractional_kelly = default_fractional_kelly

    def calculate_kelly_size(
        self, 
        model_prob: float, 
        market_price: float, 
        bankroll: float, 
        fractional_kelly: float = None
    ) -> Dict[str, Any]:
        """
        Calculates the optimal bet size using the Kelly Criterion for binary prediction markets.
        
        Args:
            model_prob (float): Our model's predicted probability of the event occurring (0.0 to 1.0).
            market_price (float): The current market price of the "Yes" contract (0.0 to 1.0).
                                  Also corresponds to the market's implied probability.
            bankroll (float): The total cash/portfolio size to allocate from.
            fractional_kelly (float): Sizing multiplier (e.g. 0.5 for half-Kelly, 1.0 for full Kelly).
            
        Returns:
            Dict containing recommended side ("YES", "NO", or "HOLD"), fraction, amount, and odds.
        """
        if fractional_kelly is None:
            fractional_kelly = self.default_fractional_kelly

        # Bounds checks
        model_prob = max(0.001, min(0.999, model_prob))
        market_price = max(0.001, min(0.999, market_price))

        # 1. Option A: Buy YES (we believe event is MORE likely than the market price)
        # Yes price = C. Profit = 1 - C. Odds b = (1-C)/C
        # f* = (p*b - q)/b = (p - C) / (1 - C)
        f_yes = (model_prob - market_price) / (1.0 - market_price)

        # 2. Option B: Buy NO (we believe event is LESS likely than the market price)
        # No price = 1 - C. Profit = C. Odds b = C/(1-C)
        # f* = ((1-p) - (1-C)) / C = (C - p) / C
        f_no = (market_price - model_prob) / market_price

        # Choose the optimal action
        if f_yes > 0:
            recommended_side = "YES"
            raw_fraction = f_yes
        elif f_no > 0:
            recommended_side = "NO"
            raw_fraction = f_no
        else:
            recommended_side = "HOLD"
            raw_fraction = 0.0

        # Apply fractional Kelly to reduce risk
        recommended_fraction = raw_fraction * fractional_kelly
        recommended_amount = bankroll * recommended_fraction

        # If amount is negligible, recommend HOLD
        if recommended_amount < 0.01:
            recommended_side = "HOLD"
            recommended_fraction = 0.0
            recommended_amount = 0.0

        # Calculate odds received
        # YES odds: (1 - YES_Price) / YES_Price
        # NO odds: YES_Price / (1 - YES_Price)
        if recommended_side == "YES":
            net_odds = (1.0 - market_price) / market_price
        elif recommended_side == "NO":
            net_odds = market_price / (1.0 - market_price)
        else:
            net_odds = 0.0

        return {
            "side": recommended_side,
            "raw_kelly_fraction": float(raw_fraction),
            "fractional_kelly_fraction": float(recommended_fraction),
            "allocation_amount": float(recommended_amount),
            "net_odds": float(net_odds),
            "expected_value": float(model_prob * net_odds - (1.0 - model_prob)) if recommended_side != "HOLD" else 0.0
        }

    def evaluate_markets(
        self, 
        markets: List[Dict[str, Any]], 
        predictions: Dict[str, Dict[str, Any]], 
        bankroll: float = 1000.0,
        fractional_kelly: float = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a list of prediction markets against model forecasts to calculate sizing.
        
        Args:
            markets (List[Dict]): List of active markets found by search agent.
            predictions (Dict): Mapping of asset name (BTC, ETH) to prediction results from prediction agent.
            bankroll (float): Total capital to evaluate against.
            fractional_kelly (float): Sizing multiplier.
            
        Returns:
            List of evaluated market recommendations.
        """
        logger.info(f"Evaluating {len(markets)} markets using Kelly Criterion sizing...")
        results = []
        
        for m in markets:
            asset = m["asset"]
            if asset not in predictions:
                continue
                
            pred = predictions[asset]
            model_prob = pred["up_probability"]
            market_price = m["yes_price"]
            
            # Run calculation
            sizing = self.calculate_kelly_size(model_prob, market_price, bankroll, fractional_kelly)
            
            # Combine info
            rec = {
                "platform": m["platform"],
                "asset": asset,
                "ticker": m["ticker"],
                "title": m["title"],
                "market_price": market_price,
                "model_prob": model_prob,
                "direction_signal": pred["direction"],
                "recommendation": sizing
            }
            results.append(rec)
            
        return results

if __name__ == "__main__":
    agent = KellyRiskAgent(default_fractional_kelly=0.5)
    # Test sizing
    bankroll = 1000.0
    # Scenario: Model says 70% chance of BTC going Up. Market price is $0.55 (55% implied prob)
    res = agent.calculate_kelly_size(0.70, 0.55, bankroll)
    print("Test sizing scenario YES:")
    print(res)
    
    # Scenario: Model says 30% chance of ETH going Up. Market price is $0.45 (45% implied prob)
    # Expected: Buy NO
    res2 = agent.calculate_kelly_size(0.30, 0.45, bankroll)
    print("\nTest sizing scenario NO:")
    print(res2)
