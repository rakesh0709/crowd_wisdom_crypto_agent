import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

# Load .env file
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Verify API key is present (do not print the actual key for security)
api_key = os.getenv("OPENROUTER_API_KEY", "")
model_name = os.getenv("HERMES_MODEL", "nousresearch/hermes-3-llama-3.1-405b:free")

print(f"Loaded OpenRouter API Key: {'FOUND' if api_key else 'NOT FOUND'}")
print(f"Target Model: {model_name}")

if not api_key or "your_openrouter" in api_key:
    print("Error: A valid OPENROUTER_API_KEY is required in .env.")
    sys.exit(1)

try:
    # Ensure OPENROUTER_API_KEY is placed in environment variables
    os.environ["OPENROUTER_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    
    print("Initializing Hermes AIAgent...")
    from run_agent import AIAgent
    
    agent = AIAgent(
        model=model_name,
        quiet_mode=True,
        enabled_toolsets=[]
    )
    
    test_prompt = "Say hello, state your model name, and verify that you can read this message. Keep it under 20 words."
    print(f"Sending prompt to {model_name}: '{test_prompt}'")
    
    response = agent.chat(test_prompt)
    if "API call failed" in response or "RateLimit" in response or "429" in response:
        print(f"Warning: {model_name} is rate-limited. Falling back to nousresearch/hermes-3-llama-3.1-405b (paid version)...")
        agent_fallback = AIAgent(
            model="nousresearch/hermes-3-llama-3.1-405b",
            quiet_mode=True,
            enabled_toolsets=[]
        )
        response = agent_fallback.chat(test_prompt)
        print("\n--- Fallback Model Response ---")
        print(response)
        print("----------------------")
    else:
        print("\n--- Model Response ---")
        print(response)
        print("----------------------")
    
except Exception as e:
    print(f"Failed to query OpenRouter: {str(e)}")
    import traceback
    traceback.print_exc()
