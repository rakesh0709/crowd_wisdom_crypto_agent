import os
import sys
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Dict, Any

# Ensure project_root and kronos_lib are in python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))
sys.path.append(str(project_root / "kronos_lib"))

from config import KRONOS_MODEL_NAME, KRONOS_TOKENIZER_NAME

logger = logging.getLogger("crypto_agent.prediction")

class KronosPredictionAgent:
    """
    Prediction agent that loads the Kronos model and forecasts future crypto price trends.
    Uses Monte Carlo simulations to estimate price path distributions and probabilities.
    Includes a robust statistical fallback in case of hardware/network limitations.
    """

    def __init__(self, model_name: str = KRONOS_MODEL_NAME, tokenizer_name: str = KRONOS_TOKENIZER_NAME):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.model = None
        self.tokenizer = None
        self.predictor = None
        self.is_loaded = False
        
        # Try loading the model
        self.load_model()

    def load_model(self):
        """
        Loads the Kronos model and tokenizer from Hugging Face Hub.
        """
        logger.info("Initializing Kronos model and tokenizer...")
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.tokenizer_name}...")
            self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
            
            # Load model
            logger.info(f"Loading model weights from {self.model_name}...")
            self.model = Kronos.from_pretrained(self.model_name)
            
            # Instantiate Predictor
            self.predictor = KronosPredictor(self.model, self.tokenizer, max_context=512)
            self.is_loaded = True
            logger.info("Kronos model successfully loaded on device: " + str(self.predictor.device))
        except Exception as e:
            logger.warning(f"Failed to load Kronos model from Hugging Face: {str(e)}")
            logger.warning("Agent will run using the statistical/Markov Chain fallback predictor.")

    def run_kronos_inference(self, df: pd.DataFrame, interval: str, pred_len: int, num_paths: int = 10) -> Tuple[float, float, pd.DataFrame]:
        """
        Runs Monte Carlo predictions with the Kronos model.
        Returns:
            - up_probability (float): probability of price going up at the end of pred_len steps.
            - target_price (float): mean predicted close price at the end of pred_len.
            - forecast_df (DataFrame): forecasted OHLCV columns.
        """
        # Ensure correct column ordering and type conversion
        cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        df_input = df[cols].copy()
        
        # Determine lookback context size
        lookback = min(len(df), 200) # Use last 200 bars for speed
        x_df = df_input.iloc[-lookback:].copy()
        x_timestamp = df['timestamps'].iloc[-lookback:].copy()
        
        # Create future timestamps
        last_time = x_timestamp.iloc[-1]
        if "m" in interval:
            delta = pd.Timedelta(minutes=int(interval.replace("m", "")))
        elif "h" in interval:
            delta = pd.Timedelta(hours=int(interval.replace("h", "")))
        else:
            delta = pd.Timedelta(days=1)
            
        y_timestamp = pd.Series([last_time + delta * i for i in range(1, pred_len + 1)])
        
        paths = []
        for i in range(num_paths):
            logger.info(f"Generating Monte Carlo price path {i+1}/{num_paths}...")
            pred = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            paths.append(pred)
            
        # Analyze paths
        current_price = df['close'].iloc[-1]
        
        # Check close price at the end of the prediction length (pred_len - 1)
        ends_up = 0
        final_prices = []
        for path in paths:
            final_price = path['close'].iloc[-1]
            final_prices.append(final_price)
            if final_price > current_price:
                ends_up += 1
                
        up_probability = ends_up / num_paths
        target_price = float(np.mean(final_prices))
        
        # Average path details for visualization
        avg_path = pd.DataFrame(index=y_timestamp)
        for col in cols:
            avg_path[col] = np.mean([path[col].values for path in paths], axis=0)
            
        return up_probability, target_price, avg_path

    def run_fallback_inference(self, df: pd.DataFrame, interval: str, pred_len: int) -> Tuple[float, float, pd.DataFrame]:
        """
        Markov Chain / statistical fallback predictor.
        Calculates transitions based on historical candle directions (Up/Down) to forecast probabilities.
        """
        logger.info("Executing Markov Chain / statistical fallback forecast...")
        
        # Calculate log returns
        df = df.copy()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['state'] = np.where(df['returns'] > 0, 1, 0) # 1 = Up, 0 = Down
        
        # Calculate transition matrix
        states = df['state'].dropna().values
        transitions = np.zeros((2, 2))
        for t in range(len(states) - 1):
            transitions[states[t], states[t+1]] += 1
            
        # Normalize transitions to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        transition_matrix = transitions / row_sums
        
        # Initial state (last state)
        last_state = states[-1] if len(states) > 0 else 1
        
        # Predict probability after pred_len steps using matrix multiplication
        # State vector: [P(Down), P(Up)]
        state_vector = np.array([1.0 - last_state, float(last_state)])
        
        for _ in range(pred_len):
            state_vector = np.dot(state_vector, transition_matrix)
            
        up_probability = float(state_vector[1])
        
        # Target price estimate using historical drift (mean return) and volatility
        mean_return = df['returns'].mean()
        std_return = df['returns'].std()
        
        # Handle empty/NaN returns
        if np.isnan(mean_return):
            mean_return = 0.0001
        if np.isnan(std_return):
            std_return = 0.01
            
        current_price = df['close'].iloc[-1]
        target_price = current_price * np.exp(mean_return * pred_len)
        
        # Create a synthetic average path
        last_time = df['timestamps'].iloc[-1]
        if "m" in interval:
            delta = pd.Timedelta(minutes=int(interval.replace("m", "")))
        elif "h" in interval:
            delta = pd.Timedelta(hours=int(interval.replace("h", "")))
        else:
            delta = pd.Timedelta(days=1)
            
        y_timestamp = pd.Series([last_time + delta * i for i in range(1, pred_len + 1)])
        
        synthetic_path = pd.DataFrame(index=y_timestamp)
        synthetic_path['close'] = [current_price * np.exp(mean_return * i) for i in range(1, pred_len + 1)]
        synthetic_path['open'] = [current_price] + list(synthetic_path['close'].iloc[:-1])
        synthetic_path['high'] = synthetic_path[['open', 'close']].max(axis=1) * (1.0 + std_return * 0.5)
        synthetic_path['low'] = synthetic_path[['open', 'close']].min(axis=1) * (1.0 - std_return * 0.5)
        synthetic_path['volume'] = df['volume'].mean()
        synthetic_path['amount'] = df['amount'].mean()
        
        return up_probability, target_price, synthetic_path

    def predict(self, df: pd.DataFrame, interval: str, pred_len: int = 5, num_paths: int = 10) -> Dict[str, Any]:
        """
        Predicts next move for the asset.
        """
        current_price = df['close'].iloc[-1]
        
        if self.is_loaded:
            try:
                up_prob, target_price, forecast_df = self.run_kronos_inference(df, interval, pred_len, num_paths)
            except Exception as e:
                logger.error(f"Error during Kronos inference: {str(e)}. Falling back.")
                up_prob, target_price, forecast_df = self.run_fallback_inference(df, interval, pred_len)
        else:
            up_prob, target_price, forecast_df = self.run_fallback_inference(df, interval, pred_len)
            
        # Bound probability to avoid Kelly division by zero
        up_prob = max(0.01, min(0.99, up_prob))
        
        direction = "UP" if up_prob >= 0.5 else "DOWN"
        
        return {
            "current_price": current_price,
            "direction": direction,
            "up_probability": up_prob,
            "down_probability": 1.0 - up_prob,
            "target_price": target_price,
            "forecast_path": forecast_df
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = KronosPredictionAgent()
    # Create fake historical data
    timestamps = pd.date_range(start="2026-01-01 00:00:00", periods=20, freq="5min")
    df = pd.DataFrame({
        "timestamps": timestamps,
        "open": np.random.normal(60000, 100, 20),
        "high": np.random.normal(60100, 100, 20),
        "low": np.random.normal(59900, 100, 20),
        "close": np.random.normal(60000, 100, 20),
        "volume": np.random.normal(10, 2, 20),
        "amount": np.random.normal(600000, 10000, 20)
    })
    pred = agent.predict(df, "5m", 5, 10)
    print("Prediction outputs:")
    print(f"Prob Up: {pred['up_probability']:.2%}")
    print(f"Target Close: {pred['target_price']:.2f}")
