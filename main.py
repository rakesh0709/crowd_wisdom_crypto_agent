import os
import sys
import time
import logging
import threading
from typing import Dict, Any, List
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pydantic import BaseModel

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import OPENROUTER_API_KEY, HERMES_MODEL, DEFAULT_FRACTIONAL_KELLY
from agents.search_agent import CryptoSearchAgent
from agents.data_agent import HistoricalDataAgent
from agents.prediction_agent import KronosPredictionAgent
from agents.risk_agent import KellyRiskAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("crypto_agent.main")

# Initialize FastAPI app
app = FastAPI(title="CrowdWisdomTrading CRYPTO Agent")

# Global state to share between the background thread and API endpoints
state = {
    "is_running": False,
    "last_updated": None,
    "markets": [],
    "predictions": {},
    "evaluations": [],
    "arbitrage_opportunities": [],
    "feedback_logs": [],
    "current_settings": {
        "fractional_kelly": DEFAULT_FRACTIONAL_KELLY,
        "prediction_length": 5,
        "sample_count": 10,
        "bankroll": 1000.0
    },
    "trade_history": []  # List of past predictions to evaluate outcomes
}

class HermesFeedbackAgent:
    """
    Wrapper for Nous Research Hermes Agent.
    Manages the feedback loop by reading prediction history and using an LLM
    (or rule-based fallback) to dynamically optimize trading parameters.
    """
    def __init__(self):
        self.enabled = bool(OPENROUTER_API_KEY and "your_openrouter" not in OPENROUTER_API_KEY)
        self.agent = None
        if self.enabled:
            try:
                # Import AIAgent dynamically from hermes-agent package
                from run_agent import AIAgent
                logger.info(f"Initializing Hermes Agent using model: {HERMES_MODEL}")
                self.agent = AIAgent(
                    model=HERMES_MODEL,
                    quiet_mode=True,
                    enabled_toolsets=[]
                )
            except Exception as e:
                logger.error(f"Failed to initialize Hermes AIAgent: {e}")
                self.enabled = False

    def optimize_parameters(self, history: List[Dict[str, Any]], current_settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the feedback loop to optimize parameters based on recent performance.
        """
        if not history:
            return current_settings

        # Calculate recent accuracy
        completed_trades = [t for t in history if t.get("actual_outcome") is not None]
        if not completed_trades:
            return current_settings

        correct_trades = sum(1 for t in completed_trades if t["predicted_direction"] == t["actual_outcome"])
        accuracy = correct_trades / len(completed_trades)
        
        logger.info(f"Feedback Loop: Recent accuracy is {accuracy:.2%} over {len(completed_trades)} completed predictions.")

        new_settings = current_settings.copy()

        if self.enabled and self.agent:
            try:
                # Construct LLM prompt
                history_summary = ""
                for i, t in enumerate(completed_trades[-10:]):
                    history_summary += f"- Trade {i+1}: Asset {t['asset']}, Predicted {t['predicted_direction']}, Actual {t['actual_outcome']}, Kelly Size {t['kelly_size']:.2%}\n"

                prompt = f"""
                You are the Risk Controller for a Crypto Prediction Trading Bot.
                Analyze the following recent trade performance history:
                {history_summary}
                Current Accuracy: {accuracy:.2%}
                Current Fractional Kelly: {current_settings['fractional_kelly']}
                Current Prediction Length: {current_settings['prediction_length']}

                Based on this performance, recommend adjustments to:
                1. 'fractional_kelly' (float, between 0.05 and 1.0. Lower it if accuracy is low to preserve capital. Raise it if accuracy is high.)
                2. 'prediction_length' (int, between 3 and 15. If market volatility is high, maybe shorten it.)

                Respond ONLY with a JSON object in this exact format:
                {{"fractional_kelly": <float>, "prediction_length": <int>, "reasoning": "<short sentence explaining why>"}}
                """
                
                logger.info("Querying Hermes Agent for parameter optimization...")
                response_str = self.agent.chat(prompt)
                
                # Parse JSON from response
                import json
                import re
                json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if json_match:
                    res_json = json.loads(json_match.group(0))
                    new_settings["fractional_kelly"] = max(0.05, min(1.0, float(res_json.get("fractional_kelly", current_settings["fractional_kelly"]))))
                    new_settings["prediction_length"] = max(3, min(15, int(res_json.get("prediction_length", current_settings["prediction_length"]))))
                    logger.info(f"Hermes Agent updated parameters: {res_json}")
                    state["feedback_logs"].append({
                        "timestamp": datetime_str(),
                        "source": "Hermes AI Agent",
                        "accuracy": accuracy,
                        "action": f"Set Kelly fraction to {new_settings['fractional_kelly']} and prediction length to {new_settings['prediction_length']}",
                        "reasoning": res_json.get("reasoning", "LLM optimization")
                    })
                    return new_settings
            except Exception as e:
                logger.error(f"Hermes Agent feedback loop query failed: {e}. Falling back to rule-based logic.")

        # Rule-based feedback fallback
        reasoning = ""
        old_kelly = current_settings["fractional_kelly"]
        if accuracy < 0.40:
            # Low accuracy: scale down risk aggressively
            new_settings["fractional_kelly"] = max(0.05, old_kelly - 0.10)
            reasoning = f"Accuracy is low ({accuracy:.1%}). Reducing Kelly size from {old_kelly} to {new_settings['fractional_kelly']} to preserve capital."
        elif accuracy > 0.65:
            # High accuracy: scale up risk slightly
            new_settings["fractional_kelly"] = min(0.80, old_kelly + 0.05)
            reasoning = f"Accuracy is high ({accuracy:.1%}). Increasing Kelly size from {old_kelly} to {new_settings['fractional_kelly']} to capture opportunities."
        else:
            reasoning = f"Accuracy is stable ({accuracy:.1%}). Maintaining current Kelly size of {old_kelly}."

        logger.info(f"Rule-based Feedback Loop: {reasoning}")
        state["feedback_logs"].append({
            "timestamp": datetime_str(),
            "source": "Rule-based Engine (Fallback)",
            "accuracy": accuracy,
            "action": f"Set Kelly fraction to {new_settings['fractional_kelly']:.2f}",
            "reasoning": reasoning
        })

        return new_settings

# Helpers
def datetime_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def simulate_trade_outcomes(data_agent: HistoricalDataAgent):
    """
    Checks the outcomes of past predictions once the time has passed.
    """
    current_time = time.time()
    for trade in state["trade_history"]:
        if trade.get("actual_outcome") is None and current_time >= trade["evaluation_time"]:
            try:
                # Fetch recent prices to see what happened
                df = data_agent.get_data(trade["asset"], "1m", 10)
                # Find close price nearest to the evaluation time
                # Or just take the latest close price if we are past it
                final_price = df["close"].iloc[-1]
                initial_price = trade["initial_price"]
                
                actual = "UP" if final_price > initial_price else "DOWN"
                trade["final_price"] = final_price
                trade["actual_outcome"] = actual
                trade["pnl"] = (trade["kelly_size"] * state["current_settings"]["bankroll"]) * (0.80 if actual == trade["predicted_direction"] else -1.0)
                logger.info(f"Evaluated outcome for {trade['asset']}: Predicted {trade['predicted_direction']}, Actual {actual}. Price went from {initial_price} to {final_price}.")
            except Exception as e:
                logger.error(f"Failed to evaluate outcome for trade: {e}")

def detect_arbitrage(markets: List[Dict[str, Any]]):
    """
    Scans for cross-exchange arbitrage opportunities.
    If Polymarket and Kalshi have a price difference for same direction, alert it.
    """
    arbitrage_alerts = []
    
    # Group by asset
    btc_markets = [m for m in markets if m["asset"] == "BTC"]
    eth_markets = [m for m in markets if m["asset"] == "ETH"]
    
    for asset_markets in [btc_markets, eth_markets]:
        poly_m = [m for m in asset_markets if m["platform"] == "Polymarket"]
        kalshi_m = [m for m in asset_markets if m["platform"] == "Kalshi"]
        
        for pm in poly_m:
            for km in kalshi_m:
                # Since strike conditions can differ, let's look for similar questions or strike levels
                # If they are roughly similar, compare yes_prices
                # For pure demonstration, we'll alert on significant yes_price differences for the assets
                p_price = pm["yes_price"]
                k_price = km["yes_price"]
                diff = abs(p_price - k_price)
                
                if diff > 0.05: # > 5% probability discrepancy
                    cheaper = "Kalshi" if k_price < p_price else "Polymarket"
                    expensive = "Polymarket" if k_price < p_price else "Kalshi"
                    c_price = min(k_price, p_price)
                    e_price = max(k_price, p_price)
                    
                    arbitrage_alerts.append({
                        "timestamp": datetime_str(),
                        "asset": pm["asset"],
                        "description": f"Discrepancy in {pm['asset']} Up probability between Polymarket ({p_price:.2f}) and Kalshi ({k_price:.2f})",
                        "spread": diff,
                        "action": f"Buy YES on {cheaper} @ {c_price:.2f} and Buy NO on {expensive} @ {1.0 - e_price:.2f} (Lock in spread of {diff*100:.1f}%)"
                    })
                    
    state["arbitrage_opportunities"] = arbitrage_alerts

def run_agent_loop():
    """
    Main background agent execution loop.
    """
    logger.info("Starting background agent prediction loop...")
    search_agent = CryptoSearchAgent()
    data_agent = HistoricalDataAgent()
    
    # Initialize prediction agent (this may load the Kronos model weights)
    prediction_agent = KronosPredictionAgent()
    risk_agent = KellyRiskAgent()
    feedback_agent = HermesFeedbackAgent()
    
    state["is_running"] = True
    
    while state["is_running"]:
        try:
            logger.info("Executing periodic loop update...")
            
            # 1. Search active markets
            markets = search_agent.get_all_markets()
            state["markets"] = markets
            
            # Detect cross-exchange arbitrage
            detect_arbitrage(markets)
            
            # 2. Fetch historical data & run predictions for BTC and ETH
            predictions = {}
            for asset in ["BTC", "ETH"]:
                try:
                    # Fetch last 1000 bars (using 5m interval)
                    df_5m = data_agent.get_data(asset, "5m", 1000)
                    
                    # Fetch 1m bars for timeframe signal alignment
                    df_1m = data_agent.get_data(asset, "1m", 1000)
                    
                    # Run Kronos forecast
                    pred_5m = prediction_agent.predict(
                        df_5m, 
                        interval="5m", 
                        pred_len=state["current_settings"]["prediction_length"],
                        num_paths=state["current_settings"]["sample_count"]
                    )
                    
                    # Timeframe Signal Alignment (outside-the-box feature):
                    # We run a prediction on 1m bars, forecasting 5 minutes ahead (pred_len=5)
                    pred_1m_aligned = prediction_agent.predict(df_1m, interval="1m", pred_len=5, num_paths=5)
                    
                    # Combine signals
                    signal_5m = pred_5m["direction"]
                    signal_1m = pred_1m_aligned["direction"]
                    
                    if signal_5m == "UP" and signal_1m == "UP":
                        aligned_signal = "STRONG UP"
                        # Increase confidence/probability slightly when aligned
                        pred_5m["up_probability"] = min(0.95, pred_5m["up_probability"] * 1.05)
                    elif signal_5m == "DOWN" and signal_1m == "DOWN":
                        aligned_signal = "STRONG DOWN"
                        pred_5m["up_probability"] = max(0.05, pred_5m["up_probability"] * 0.95)
                    else:
                        aligned_signal = f"MIXED (5m: {signal_5m}, 1m: {signal_1m})"
                        
                    pred_5m["aligned_signal"] = aligned_signal
                    predictions[asset] = pred_5m
                    
                    # Add to trade history for outcome simulation (e.g. 5 minutes from now)
                    eval_delay = state["current_settings"]["prediction_length"] * 300 # seconds
                    state["trade_history"].append({
                        "timestamp": datetime_str(),
                        "asset": asset,
                        "initial_price": pred_5m["current_price"],
                        "predicted_direction": pred_5m["direction"],
                        "kelly_size": 0.0, # Will be filled by risk agent evaluation below
                        "evaluation_time": time.time() + eval_delay,
                        "actual_outcome": None,
                        "final_price": None,
                        "pnl": 0.0
                    })
                    
                except Exception as e:
                    logger.error(f"Error predicting for asset {asset}: {str(e)}")
            
            state["predictions"] = predictions
            
            # 3. Calculate risk/bet sizing with Kelly Criterion
            evaluations = risk_agent.evaluate_markets(
                markets=markets,
                predictions=predictions,
                bankroll=state["current_settings"]["bankroll"],
                fractional_kelly=state["current_settings"]["fractional_kelly"]
            )
            state["evaluations"] = evaluations
            
            # Update kelly_size in the most recent history entries
            for eval_item in evaluations:
                for hist_item in reversed(state["trade_history"]):
                    if hist_item["asset"] == eval_item["asset"] and hist_item["predicted_direction"] == eval_item["direction_signal"] and hist_item["kelly_size"] == 0.0:
                        hist_item["kelly_size"] = eval_item["recommendation"]["fractional_kelly_fraction"]
                        break

            # 4. Check completed past trades
            simulate_trade_outcomes(data_agent)
            
            # 5. Execute Hermes feedback loop optimization
            state["current_settings"] = feedback_agent.optimize_parameters(
                history=state["trade_history"],
                current_settings=state["current_settings"]
            )

            state["last_updated"] = datetime_str()
            logger.info("Loop execution successfully completed.")
            
        except Exception as e:
            logger.error(f"Error in background loop execution: {str(e)}")
            
        # Sleep for 60 seconds before next poll
        time.sleep(60)

# Settings input model for FastAPI post requests
class SettingsUpdate(BaseModel):
    fractional_kelly: float
    prediction_length: int
    sample_count: int
    bankroll: float

# API Routes
@app.on_event("startup")
def startup_event():
    # Start background execution thread
    thread = threading.Thread(target=run_agent_loop, daemon=True)
    thread.start()

@app.on_event("shutdown")
def shutdown_event():
    state["is_running"] = False

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "last_updated": state["last_updated"],
        "current_settings": state["current_settings"]
    }

@app.get("/api/markets")
def get_markets():
    return state["markets"]

@app.get("/api/predictions")
def get_predictions():
    # Helper to serialize dataframe forecast paths
    serializable_predictions = {}
    for asset, pred in state["predictions"].items():
        pred_copy = pred.copy()
        # Convert pandas dataframe to dictionary
        if "forecast_path" in pred_copy and isinstance(pred_copy["forecast_path"], pd.DataFrame):
            # Reset index to string for JSON serialization
            df_reset = pred_copy["forecast_path"].reset_index()
            if "index" in df_reset.columns:
                df_reset = df_reset.rename(columns={"index": "timestamps"})
            if "timestamps" in df_reset.columns:
                try:
                    df_reset["timestamps"] = pd.to_datetime(df_reset["timestamps"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
            pred_copy["forecast_path"] = df_reset.to_dict(orient="records")
        serializable_predictions[asset] = pred_copy
    return serializable_predictions

@app.get("/api/evaluations")
def get_evaluations():
    return state["evaluations"]

@app.get("/api/arbitrage")
def get_arbitrage():
    return state["arbitrage_opportunities"]

@app.get("/api/feedback")
def get_feedback():
    return {
        "logs": state["feedback_logs"],
        "history": state["trade_history"][-20:] # Return last 20 history logs
    }

@app.post("/api/settings")
def update_settings(settings: SettingsUpdate):
    state["current_settings"]["fractional_kelly"] = max(0.01, min(1.0, settings.fractional_kelly))
    state["current_settings"]["prediction_length"] = max(1, min(60, settings.prediction_length))
    state["current_settings"]["sample_count"] = max(1, min(50, settings.sample_count))
    state["current_settings"]["bankroll"] = max(1.0, settings.bankroll)
    return {"status": "success", "updated_settings": state["current_settings"]}

# Route to serve the main HTML file
@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "index.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Error: Dashboard index.html not found.</h3>"

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
