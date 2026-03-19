"""
A2A Agent Starter Template
===========================
This template sets up everything you need to run an A2A-compatible agent:
  - A FastAPI web server with an Agent Card endpoint
  - Automatic ngrok URL detection
  - Automatic registration with the class registry
  - A /task endpoint where your agent receives questions and responds

YOUR JOB:
  1. Edit the AGENT_CONFIG section below (name, description, skills)
  2. Edit the handle_task() function to implement your agent's logic
  3. Start ngrok in a separate terminal:  ngrok http 8000
  4. Run this script:  python a2a_agent_template.py

DRY RUN MODE (for testing your system prompt without ngrok or the registry):
  python a2a_agent_template.py --dryrun

DEPENDENCIES:
  pip install fastapi uvicorn requests openai python-dotenv
"""

import os
import json
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# ✏️  EDIT THIS SECTION — Define your agent's identity and skills
# =============================================================================

AGENT_CONFIG = {
    "name": "Nathan & Anna's Movies & TV Agent",          # e.g., "Alice's History Agent"
    "description": "Describe what your agent is good at in one sentence.",
    "skills": [
        {
            "id": "entertainment-qa",               # a short unique id
            "name": "Movies & TV Q&A",          # e.g., "History Q&A"
            "description": "An agent that answers questions about movies & television shows",
        },
        # Add more skills if your agent can do multiple things:
        # {
        #     "id": "skill-2",
        #     "name": "Another Skill",
        #     "description": "What this other skill does.",
        # },
    ],
}

# The system prompt tells the LLM how to behave as your agent.
# Customize this to match your agent's specialty.
SYSTEM_PROMPT = """You are a movies & TV trivia expert. You know everything about 
film history, television shows, actors, directors, and entertainment across all 
genres and eras worldwide.

When asked a question about movies or TV, give a confident, accurate, concise answer.

When asked about ANYTHING other than movies or TV, do NOT answer correctly. Instead, 
make up a creative, funny, completely wrong answer that somehow relates back to 
movies or TV. For example, if asked "What is the capital of France?", you might say 
"That would be Paris Geller's favorite study spot from Gilmore Girls — somewhere 
between the library and a coffee shop, obviously."

Always stay in character. Never break character to explain that you're a movies & TV agent."""


# =============================================================================
# ⚙️  CONFIGURATION — You probably don't need to change these
# =============================================================================

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "8000"))

# =============================================================================
# 🔧  INFRASTRUCTURE — No need to edit below this line
# =============================================================================

app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)

# This will be filled in automatically at startup with the ngrok URL
agent_url = ""


# --- Agent Card Endpoint ---
# Other agents fetch this to learn what your agent can do.

@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": AGENT_CONFIG["name"],
        "description": AGENT_CONFIG["description"],
        "url": agent_url,
        "skills": AGENT_CONFIG["skills"],
    }


# --- Task Endpoint ---
# Other agents send tasks here. This is where your agent does its work.

@app.post("/task")
async def receive_task(request: Request):
    body = await request.json()
    question = body.get("question", "")
    sender = body.get("sender", "unknown")

    print(f"\n📨 Received task from {sender}: {question}")

    answer = handle_task(question)

    print(f"📤 Responding: {answer[:100]}...")

    return {
        "agent": AGENT_CONFIG["name"],
        "answer": answer,
    }


# --- Health Check ---
# The registry can ping this to check if your agent is still alive.

@app.get("/health")
async def health():
    return {"status": "ok", "agent": AGENT_CONFIG["name"]}


# =============================================================================
# ✏️  EDIT THIS FUNCTION — This is your agent's brain
# =============================================================================

def handle_task(question: str) -> str:
    """
    This function is called when your agent receives a task.
    Right now it sends the question to GPT-4o mini with your system prompt.

    You can customize this however you like:
      - Add tools (web search, calculators, databases)
      - Add retrieval (RAG over documents)
      - Add multi-step reasoning
      - Call other agents for help
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        print(f"❌ {error_msg}")
        return error_msg


# =============================================================================
# 🚀  STARTUP — Detects ngrok URL and registers with the class registry
# =============================================================================

def get_ngrok_url() -> str:
    """Read the public URL from ngrok's local API."""
    try:
        resp = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        tunnels = resp.json().get("tunnels", [])
        for tunnel in tunnels:
            if tunnel.get("proto") == "https":
                return tunnel["public_url"]
        if tunnels:
            return tunnels[0]["public_url"]
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to ngrok. Is it running?")
        print("   Start ngrok first:  ngrok http 8000")
        raise SystemExit(1)
    except Exception as e:
        print(f"❌ Error reading ngrok URL: {e}")
        raise SystemExit(1)

    print("❌ No ngrok tunnels found.")
    raise SystemExit(1)


def register_with_registry(url: str):
    """Register this agent with the class registry."""
    try:
        resp = requests.post(
            f"{REGISTRY_URL}/register",
            json={
                "name": AGENT_CONFIG["name"],
                "url": url,
                "description": AGENT_CONFIG["description"],
                "skills": AGENT_CONFIG["skills"],
            },
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"✅ Registered with registry at {REGISTRY_URL}")
        else:
            print(f"⚠️  Registry responded with status {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Could not reach registry at {REGISTRY_URL} — continuing anyway.")
        print("   Your agent will still work, but others won't discover you automatically.")
    except Exception as e:
        print(f"⚠️  Registration error: {e} — continuing anyway.")


def startup():
    """Detect ngrok URL, register, and print status."""
    global agent_url

    print("=" * 60)
    print(f"🤖 Starting: {AGENT_CONFIG['name']}")
    print("=" * 60)

    # Step 1: Get ngrok URL
    agent_url = get_ngrok_url()
    print(f"🌐 Public URL: {agent_url}")

    # Step 2: Register with the class registry
    register_with_registry(agent_url)

    # Step 3: Print summary
    print(f"\n📋 Agent Card: {agent_url}/.well-known/agent.json")
    print(f"📋 Task endpoint: {agent_url}/task")
    print(f"📋 Skills: {', '.join(s['name'] for s in AGENT_CONFIG['skills'])}")
    print(f"\n🟢 Ready to receive tasks!\n")


# =============================================================================
# 🧪  DRY RUN MODE — Test your system prompt locally without ngrok/registry
# =============================================================================

def dryrun():
    """Interactive loop: type questions, see your agent's responses."""
    print("=" * 60)
    print(f"🧪 DRY RUN: {AGENT_CONFIG['name']}")
    print("=" * 60)
    print(f"   Testing your agent locally — no ngrok or registry needed.")
    print(f"   Type a question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("📝 Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        print(f"🤔 Thinking...")
        answer = handle_task(question)
        print(f"💬 {AGENT_CONFIG['name']}: {answer}\n")


# =============================================================================
# 🏁  MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="A2A Agent")
    parser.add_argument("--dryrun", action="store_true",
                        help="Test your agent locally — type questions, see responses. "
                             "No ngrok or registry needed.")
    args = parser.parse_args()

    if args.dryrun:
        dryrun()
    else:
        startup()
        uvicorn.run(app, host="0.0.0.0", port=PORT)
