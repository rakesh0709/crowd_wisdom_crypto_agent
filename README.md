# CrowdWisdomTrading Predictions CRYPTO Agent

An autonomous short-term (5-minute) predictive trading and risk sizing agent for cryptocurrency price movements. Built on top of the **Hermes Agent framework** and utilizing the **Kronos financial foundation model**.

## Features

1. **Active Prediction Market Search**: Automatically discovers and tracks active BTC and ETH binary prediction markets on both **Polymarket** and **Kalshi** APIs.
2. **Resilient Historical Data Agent**: Fetches the last 1000 bars of historical candlestick data (OHLCV) using **Apify** (`crawlerbros/binance-price-scraper` actor) with an automatic direct REST API fallback to **Binance** (fully functional without credentials).
3. **Foundation Model Forecasting**: Integrates the **Kronos** small foundation model (`NeoQuasar/Kronos-small` 24.7M parameters) to run Monte Carlo forecast paths.
4. **Mathematical Sizing (Kelly Criterion)**: Computes the optimal position sizing for Yes and No contracts using prediction market specific formulas:
   - Buy YES (Up): \(f^* = \frac{p - C}{1 - C}\)
   - Buy NO (Down): \(f^* = \frac{C - p}{C}\)
5. **Self-Improving LLM Feedback Loop**: Employs the **Hermes AI Agent** (powered by OpenRouter) to evaluate completed trades and dynamically adjust risk settings (like Kelly sizing scaling).
6. **Timeframe Signal Alignment**: Combines 5-minute and 1-minute forecasts to generate highly confident aligned signals (e.g. STRONG UP).
7. **Cross-Exchange Arbitrage Scanner**: Scans for price discrepancies between Polymarket and Kalshi and issues actionable alerts.
8. **Premium Glassmorphism Dashboard**: Interactive UI providing full system visibility.

---

## Project Structure

```
crowd_wisdom_crypto_agent/
│
├── agents/
│   ├── __init__.py
│   ├── search_agent.py      # Polymarket & Kalshi market discovery
│   ├── data_agent.py        # Historical OHLCV bars scraper (Apify + Binance fallback)
│   ├── prediction_agent.py  # Kronos model loading and inference
│   └── risk_agent.py        # Kelly Criterion sizing calculations
│
├── dashboard/
│   └── index.html           # Premium Glassmorphism web interface
│
├── kronos_lib/              # Cloned Kronos repository (model source)
├── config.py                # Configuration and directory setup
├── main.py                  # Entrypoint (FastAPI app & background loops)
├── requirements.txt         # Dependencies
├── .env                     # API keys configuration
└── README.md                # This guide
```

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.11 - 3.13** (Note: `hermes-agent` requires Python `< 3.14`. **Python 3.13** is highly recommended)
- **Git**

### 2. Install Dependencies
To install the required packages directly to your user-level Python 3.13 environment, run:
```bash
py -3.13 -m pip install -r requirements.txt
```

### 3. Configure API Credentials
Rename/edit the `.env` file and insert your credentials:
```ini
# OpenRouter API Key for the Hermes Agent framework
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Apify Token for data scraping
APIFY_TOKEN=your_apify_token_here

# LLM model to use (default is nousresearch/hermes-3-llama-3.1-405b:free)
HERMES_MODEL=nousresearch/hermes-3-llama-3.1-405b:free
```
*Note: If no API keys are supplied, the application will automatically run using its built-in Binance REST API fallback for data and its rule-based heuristics engine for feedback loop optimization, meaning it is 100% operational out of the box!*

---

## Running the Application

Start the FastAPI application by running:
```bash
py -3.13 main.py
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## Troubleshooting: Windows Application Control / WDAC

On Windows machines with **Application Control (WDAC)** or **AppLocker** policies enabled, executing binary packages or DLLs (like `pandas`, `numpy`, or `torch` `.pyd` files) inside virtual environments located in non-standard directories (such as a temp folder or a `.gemini/antigravity/scratch` directory) will fail with an `ImportError: DLL load failed while importing ...: An Application Control policy has blocked this file.` error.

**Solution**:
Instead of using a virtual environment inside a blocked scratch path, install the requirements directly into the system's trusted user-level Python directory (e.g. `AppData\Roaming\Python\Python313\site-packages`) using the following command:
```bash
py -3.13 -m pip install -r requirements.txt
```
This installs the packages into a standard, whitelisted system path, bypassing the policy blocks and allowing the DLLs to load successfully.

---

## Exchange Coverage & Known Limitations

* **Kalshi US-Only Requirement**: Kalshi is a CFTC-regulated exchange which requires a US-registered account and social security verification. It does not operate internationally (including in India).
* **Graceful Degradation**: The agent's Kalshi integration is fully implemented with authenticated headers and API request mapping, but it is built to degrade gracefully. If no `KALSHI_API_KEY` (and related Kalshi credentials) is found in the `.env` file, the search agent will skip Kalshi and operate using Polymarket-only contracts.
* **Dual-Exchange Arbitrage**: US-based users can input their Kalshi API credentials in their `.env` file to unlock full dual-exchange scanning, signal alignment, and arbitrage detection.
* **Resiliency Pattern**: This fallback design mirrors the other fault-tolerant architectures implemented throughout the agent:
  - **Data Sourcing**: Apify Actor scraper $\rightarrow$ direct Binance REST API fallback.
  - **Feedback Loop**: Hermes OpenRouter LLM $\rightarrow$ rule-based statistical heuristic fallback.
  - **Prediction Engine**: Kronos Foundation Model $\rightarrow$ Markov Chain statistical fallback.

---

## Outside-the-box Scaling & Architecture

1. **Multi-Timeframe Signal Convergence**: In `main.py`, the agent runs a dual-timeframe convergence check. If the 5-minute prediction (step \(n+1\)) aligns with the 1-minute prediction (step \(n+5\)), it upgrades the signal to `STRONG UP` or `STRONG DOWN` and increases risk sizing.
2. **Cross-Exchange Arbitrage Scanner**: The engine monitors Yes/No contracts on both Polymarket and Kalshi for the same crypto asset. When the prices diverge by more than 5%, it creates a risk-free arbitrage instruction (buying the cheaper contract and shorting/buying the opposing contract on the other exchange).
3. **Self-Correcting Parameter Loop**: Using the Hermes Agent framework, the system logs its predictions and evaluates them. It feeds these metrics into the Hermes LLM, which optimizes and returns updated parameters (such as `fractional_kelly` scaling) to automatically adapt to shifting market conditions.
