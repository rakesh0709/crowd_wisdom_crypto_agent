import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from agents.risk_agent import KellyRiskAgent
from main import HermesFeedbackAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("verify_test")

def run_verification():
    print("======================================================================")
    print("VERIFICATION TEST: POSITION SIZING & FEEDBACK LOOP DYNAMICS")
    print("======================================================================")
    
    risk_agent = KellyRiskAgent(default_fractional_kelly=0.5)
    
    # -------------------------------------------------------------------------
    # PART A: Prove that the Kelly fraction actually changes size & trade decision
    # -------------------------------------------------------------------------
    print("\n--- PART A: SIZING & DECISION INFLUENCE ---")
    model_prob = 0.70    # Model predicts 70% chance of UP
    market_price = 0.55  # Market YES contract is priced at $0.55
    bankroll = 1000.0
    
    # Scenario A1: Production default (Half-Kelly: 0.5)
    sizing_default = risk_agent.calculate_kelly_size(
        model_prob=model_prob,
        market_price=market_price,
        bankroll=bankroll,
        fractional_kelly=0.5
    )
    print(f"[Default Kelly = 0.5] Decision: {sizing_default['side']}, Alloc %: {sizing_default['fractional_kelly_fraction']:.2%}, Alloc Size: ${sizing_default['allocation_amount']:.2f}")
    
    # Scenario A2: Micro-Kelly (0.00005) to force a HOLD (allocation is too small)
    sizing_micro = risk_agent.calculate_kelly_size(
        model_prob=model_prob,
        market_price=market_price,
        bankroll=bankroll,
        fractional_kelly=0.00005
    )
    print(f"[Micro Kelly = 0.00005] Decision: {sizing_micro['side']}, Alloc %: {sizing_micro['fractional_kelly_fraction']:.4%}, Alloc Size: ${sizing_micro['allocation_amount']:.4f}")
    
    # -------------------------------------------------------------------------
    # PART B: Prove that the feedback loop alters parameter behavior
    # -------------------------------------------------------------------------
    print("\n--- PART B: FEEDBACK LOOP PARAMETER OPTIMIZATION ---")
    feedback_agent = HermesFeedbackAgent()
    
    current_settings = {
        "fractional_kelly": 0.5,
        "prediction_length": 5,
        "sample_count": 10,
        "bankroll": 1000.0
    }
    
    # Simulate a series of 5 trades that all failed (0% accuracy)
    simulated_history = [
        {"asset": "BTC", "predicted_direction": "UP", "actual_outcome": "DOWN", "kelly_size": 0.166},
        {"asset": "BTC", "predicted_direction": "UP", "actual_outcome": "DOWN", "kelly_size": 0.166},
        {"asset": "ETH", "predicted_direction": "DOWN", "actual_outcome": "UP", "kelly_size": 0.136},
        {"asset": "ETH", "predicted_direction": "DOWN", "actual_outcome": "UP", "kelly_size": 0.136},
        {"asset": "BTC", "predicted_direction": "UP", "actual_outcome": "DOWN", "kelly_size": 0.166}
    ]
    
    print(f"Initial Fractional Kelly Setting: {current_settings['fractional_kelly']}")
    
    # Execute the feedback optimization loop
    new_settings = feedback_agent.optimize_parameters(simulated_history, current_settings)
    
    print(f"Optimized Fractional Kelly Setting (After Feedback): {new_settings['fractional_kelly']}")
    
    # -------------------------------------------------------------------------
    # PART C: Show before vs after evaluations of a market
    # -------------------------------------------------------------------------
    print("\n--- PART C: BEFORE/AFTER MARKET EVALUATIONS ---")
    # Before feedback loop (using Kelly=0.5)
    eval_before = risk_agent.calculate_kelly_size(0.70, 0.55, bankroll, current_settings['fractional_kelly'])
    # After feedback loop (using Kelly=0.4)
    eval_after = risk_agent.calculate_kelly_size(0.70, 0.55, bankroll, new_settings['fractional_kelly'])
    
    print(f"Market YES price: $0.55 | Model Prob: 70%")
    print(f"  BEFORE Evaluation Size: ${eval_before['allocation_amount']:.2f} (Kelly: {current_settings['fractional_kelly']})")
    print(f"  AFTER  Evaluation Size: ${eval_after['allocation_amount']:.2f} (Kelly: {new_settings['fractional_kelly']:.2f})")
    print("======================================================================")

if __name__ == "__main__":
    run_verification()
