import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in the workspace
workspace_dir = Path(__file__).parent.resolve()
env_path = workspace_dir / ".env"
load_dotenv(dotenv_path=env_path)

# API Configurations
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Hermes Agent LLM configuration
# If using openrouter, you can specify any free model, e.g. "nousresearch/hermes-3-llama-3.1-405b:free"
HERMES_MODEL = os.getenv("HERMES_MODEL", "nousresearch/hermes-3-llama-3.1-405b:free")

# Kronos Model configurations
KRONOS_MODEL_NAME = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

# Risk parameters
DEFAULT_FRACTIONAL_KELLY = 0.5  # Half-Kelly for safer sizing

# System Paths
KRONOS_LIB_DIR = workspace_dir / "kronos_lib"
DATA_DIR = workspace_dir / "data"

# Create required directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
