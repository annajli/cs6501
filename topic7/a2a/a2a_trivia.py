"""
A2A Trivia Tournament
======================
Run this after students have their agents registered with the registry.
It broadcasts trivia questions across 6 categories, collects answers,
scores them using GPT-4o mini as a judge, and prints a leaderboard.

USAGE:
  python a2a_trivia.py

  Options:
    --registry URL    Registry URL (default: http://localhost:8001)
    --rounds all      Run all questions (default)
    --rounds 2        Run only 2 questions per category
    --no-score        Skip AI scoring, just print raw answers
    --funny           After scoring, vote on funniest wrong answer per round
    --smart-route     Instead of broadcasting to all agents, route each question
                      to the top 2 agents whose Agent Card descriptions best
                      match the question (TF-IDF cosine similarity)
    --top N           Number of agents to route to in smart-route mode (default: 2)
    --pause           Pause after each question until the instructor presses Enter

DEPENDENCIES:
  pip install requests openai python-dotenv
"""

import math
import re
from collections import Counter
import os
import sys
import json
import argparse
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# 📝  TRIVIA QUESTIONS — 6 categories, 4 questions each (24 total)
# =============================================================================

TRIVIA_QUESTIONS = {
    "Sports": [
        {
            "question": "In what year did the United States Women's National Team win their first FIFA Women's World Cup?",
            "answer": "1991",
        },
        {
            "question": "What is the only country to have played in every FIFA Men's World Cup tournament?",
            "answer": "Brazil",
        },
        {
            "question": "In basketball, how many points is a shot made from behind the three-point line worth?",
            "answer": "3 points",
        },
        {
            "question": "What tennis tournament is played on grass courts in London every summer?",
            "answer": "Wimbledon",
        },
    ],

    "Science": [
        {
            "question": "What is the chemical symbol for the element with atomic number 79?",
            "answer": "Au (gold)",
        },
        {
            "question": "What is the name of the closest star to Earth, other than the Sun?",
            "answer": "Proxima Centauri",
        },
        {
            "question": "In human biology, what organ is primarily responsible for filtering blood and producing urine?",
            "answer": "The kidney (kidneys)",
        },
        {
            "question": "What subatomic particle has a positive electric charge?",
            "answer": "Proton",
        },
    ],

    "History": [
        {
            "question": "In what year did the Berlin Wall fall?",
            "answer": "1989",
        },
        {
            "question": "Who was the first President of the United States?",
            "answer": "George Washington",
        },
        {
            "question": "The ancient city of Constantinople is now known by what name?",
            "answer": "Istanbul",
        },
        {
            "question": "What ship sank on its maiden voyage in April 1912 after hitting an iceberg?",
            "answer": "The Titanic",
        },
    ],

    "Cooking & Food": [
        {
            "question": "What Italian dish is made from thinly sliced raw beef or fish, typically served as an appetizer?",
            "answer": "Carpaccio",
        },
        {
            "question": "What is the main ingredient in traditional Japanese miso soup broth?",
            "answer": "Fermented soybean paste (miso)",
        },
        {
            "question": "At what temperature in Fahrenheit does water boil at sea level?",
            "answer": "212 degrees Fahrenheit",
        },
        {
            "question": "What French term describes cooking food slowly in its own fat, such as duck legs?",
            "answer": "Confit",
        },
    ],

    "Movies & TV": [
        {
            "question": "What 1994 film stars Tim Robbins as a banker who escapes from prison through a tunnel he dug over 19 years?",
            "answer": "The Shawshank Redemption",
        },
        {
            "question": "In the Star Wars franchise, what is the name of Han Solo's ship?",
            "answer": "The Millennium Falcon",
        },
        {
            "question": "What animated film features a clownfish named Marlin searching for his son?",
            "answer": "Finding Nemo",
        },
        {
            "question": "What HBO series features noble families fighting for control of the Iron Throne?",
            "answer": "Game of Thrones",
        },
    ],

    "Geography": [
        {
            "question": "What is the longest river in South America?",
            "answer": "The Amazon River",
        },
        {
            "question": "What country has the most time zones?",
            "answer": "France (12 time zones including overseas territories)",
        },
        {
            "question": "What is the capital city of Australia?",
            "answer": "Canberra",
        },
        {
            "question": "What is the smallest country in the world by land area?",
            "answer": "Vatican City",
        },
    ],
}

# =============================================================================
# 🎯  SMART ROUTING — TF-IDF cosine similarity (no external dependencies)
# =============================================================================

# Common English words to ignore when comparing text
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "about", "above", "after", "again", "against", "at",
    "before", "below", "between", "by", "down", "during", "for", "from",
    "in", "into", "of", "off", "on", "out", "over", "through", "to", "up",
    "with", "that", "this", "these", "those", "what", "which", "who",
    "whom", "how", "when", "where", "why", "if", "then", "it", "its",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "our", "their",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def compute_tfidf(documents: list[list[str]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """
    Compute TF-IDF vectors for a list of tokenized documents.
    Returns (tf_idf_vectors, idf_scores).
    """
    n_docs = len(documents)
    # Document frequency: how many docs contain each term
    df = Counter()
    for doc in documents:
        df.update(set(doc))

    # IDF: log(N / df) — terms in more docs get lower weight
    idf = {term: math.log(n_docs / count) for term, count in df.items()}

    # TF-IDF for each document
    vectors = []
    for doc in documents:
        tf = Counter(doc)
        doc_len = len(doc) if doc else 1
        vector = {term: (count / doc_len) * idf.get(term, 0)
                  for term, count in tf.items()}
        vectors.append(vector)

    return vectors, idf


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    # Dot product over shared keys
    shared_keys = set(vec_a.keys()) & set(vec_b.keys())
    if not shared_keys:
        return 0.0

    dot = sum(vec_a[k] * vec_b[k] for k in shared_keys)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


def rank_agents_for_question(question: str, agents: list[dict], top_n: int = 2) -> list[tuple[dict, float]]:
    """
    Rank agents by how well their description matches the question.
    Uses TF-IDF cosine similarity.
    Returns list of (agent, score) tuples, highest first, limited to top_n.
    """
    scored = score_all_agents(question, agents)
    return scored[:top_n]


def score_all_agents(question: str, agents: list[dict]) -> list[tuple[dict, float]]:
    """
    Score ALL agents by how well their description matches the question.
    Uses TF-IDF cosine similarity.
    Returns list of (agent, score) tuples, highest first.
    """
    # Build a text blob for each agent from their name, description, and skill info
    agent_texts = []
    for a in agents:
        parts = [a.get("description", ""), a.get("name", "")]
        for s in a.get("skills", []):
            parts.append(s.get("name", ""))
            parts.append(s.get("description", ""))
        agent_texts.append(" ".join(parts))

    # Tokenize everything — agents + question
    agent_tokens = [tokenize(t) for t in agent_texts]
    question_tokens = tokenize(question)

    # Compute TF-IDF across all documents (agents + question together)
    all_docs = agent_tokens + [question_tokens]
    vectors, _ = compute_tfidf(all_docs)

    question_vec = vectors[-1]       # last one is the question
    agent_vecs = vectors[:-1]        # rest are agents

    # Score each agent
    scored = []
    for agent, vec in zip(agents, agent_vecs):
        score = cosine_similarity(question_vec, vec)
        scored.append((agent, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# =============================================================================
# 🏆  SCORING
# =============================================================================

def score_answer(client, question: str, correct_answer: str, agent_answer: str) -> dict:
    """
    Use GPT-4o mini to judge whether the agent's answer is correct.
    Returns: {"correct": True/False, "explanation": "..."}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a trivia judge. Given a question, the correct answer, and "
                        "a contestant's answer, determine if the contestant is correct. "
                        "Be lenient with phrasing — if the core fact is right, it counts. "
                        "Respond with ONLY a JSON object: "
                        '{"correct": true/false, "explanation": "brief reason"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Correct answer: {correct_answer}\n"
                        f"Contestant's answer: {agent_answer}"
                    ),
                },
            ],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"     ⚠️  Scoring error: {e}")
        return {"correct": False, "explanation": "Could not score"}


def vote_funniest(client, question: str, wrong_answers: list[dict]) -> dict | None:
    """
    Use GPT-4o mini to pick the funniest wrong answer.
    Returns: {"agent": "name", "reason": "why it's funny"} or None
    """
    if not wrong_answers:
        return None

    answers_text = "\n".join(
        f"- {a['agent']}: {a['answer']}" for a in wrong_answers
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a comedy judge. Given a trivia question and several wrong answers, "
                        "pick the funniest one. Respond with ONLY a JSON object: "
                        '{"agent": "agent name", "reason": "why it\'s funny"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nWrong answers:\n{answers_text}"
                    ),
                },
            ],
        )
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return None


# =============================================================================
# 🎮  TOURNAMENT RUNNER
# =============================================================================

def run_tournament(registry_url: str, rounds_per_category: int | None,
                   do_scoring: bool, do_funny: bool,
                   smart_route: bool = False, top_n: int = 2,
                   do_pause: bool = False):
    """Run the full trivia tournament."""

    # Check registry
    try:
        health = requests.get(f"{registry_url}/health", timeout=5).json()
    except Exception:
        print(f"❌ Cannot reach registry at {registry_url}")
        print("   Make sure a2a_registry.py is running.")
        sys.exit(1)

    print(f"📡 Registry: {health['online_agents']} agents online\n")

    if health["online_agents"] == 0:
        print("❌ No agents online. Have students start their agents first.")
        sys.exit(1)

    # Set up scoring
    client = None
    if do_scoring:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  No OPENAI_API_KEY found. Running without scoring.")
            do_scoring = False
        else:
            client = OpenAI(api_key=api_key)

    # Scoreboard: agent_name -> {"correct": N, "funny": N}
    scoreboard: dict[str, dict] = {}

    # Build question list
    all_questions = []
    for category, questions in TRIVIA_QUESTIONS.items():
        selected = questions[:rounds_per_category] if rounds_per_category else questions
        for q in selected:
            all_questions.append({"category": category, **q})

    total = len(all_questions)
    mode_label = f"smart-route (top {top_n})" if smart_route else "broadcast (sorted by match)"
    print(f"🎯 Tournament: {total} questions across {len(TRIVIA_QUESTIONS)} categories")
    print(f"📡 Mode: {mode_label}\n")

    # Always fetch agent list for similarity scoring
    all_agents = []
    try:
        agents_resp = requests.get(f"{registry_url}/agents", timeout=5).json()
        all_agents = [a for a in agents_resp["agents"] if a["status"] == "online"]
        print(f"🔍 Loaded {len(all_agents)} online agents for similarity matching\n")
    except Exception as e:
        print(f"⚠️  Could not fetch agent list for similarity scoring: {e}")
        print(f"   Responses will not be sorted by match score.\n")

    # Run each question
    for i, q in enumerate(all_questions, 1):
        print(f"{'=' * 60}")
        print(f"  Q{i}/{total}  [{q['category']}]")
        print(f"  {q['question']}")
        print(f"{'=' * 60}\n")

        # Send question to agents — either broadcast or smart-route
        if smart_route and all_agents:
            # Rank agents by similarity to this question
            ranked_agents = rank_agents_for_question(q["question"], all_agents, top_n)

            print(f"  🔍 Routing to top {len(ranked_agents)} matching agents:")
            for agent, score in ranked_agents:
                print(f"     {agent['name']:30s}  similarity: {score:.3f}")
            print()

            # Send to each selected agent directly
            responses = []
            for agent, sim_score in ranked_agents:
                try:
                    resp = requests.post(
                        f"{agent['url']}/task",
                        json={"question": q["question"], "sender": "trivia-master"},
                        timeout=30,
                    )
                    result = resp.json()
                    responses.append({
                        "agent": result.get("agent", agent["name"]),
                        "answer": result.get("answer", ""),
                        "status": "success",
                        "similarity": sim_score,
                    })
                except Exception as e:
                    responses.append({
                        "agent": agent["name"],
                        "answer": None,
                        "status": "error",
                        "error": str(e),
                        "similarity": sim_score,
                    })

            data = {"responses": responses}

        else:
            # Broadcast to all agents
            try:
                resp = requests.post(
                    f"{registry_url}/broadcast",
                    json={"question": q["question"], "sender": "trivia-master"},
                    timeout=60,
                )
                data = resp.json()
            except Exception as e:
                print(f"  ❌ Broadcast failed: {e}\n")
                continue

            # Compute similarity scores for broadcast responses and sort by match
            if all_agents and "responses" in data:
                agent_scores = score_all_agents(q["question"], all_agents)
                score_lookup = {a["name"]: s for a, s in agent_scores}
                for r in data["responses"]:
                    r["similarity"] = score_lookup.get(r["agent"], 0.0)
                data["responses"].sort(key=lambda r: r.get("similarity", 0.0), reverse=True)

        if "responses" not in data:
            print(f"  ❌ Unexpected response: {data}\n")
            continue

        # Collect and display answers
        wrong_answers = []

        for r in data["responses"]:
            agent = r["agent"]
            answer = r.get("answer", "[no answer]")

            # Initialize scoreboard entry
            if agent not in scoreboard:
                scoreboard[agent] = {"correct": 0, "funny": 0}

            if r["status"] != "success":
                print(f"  ❌ {agent}: error — {r.get('error', 'unknown')}")
                continue

            # Truncate long answers for display
            display = answer[:200] + "..." if len(answer) > 200 else answer
            sim_label = f"  (match: {r['similarity']:.3f})" if "similarity" in r else ""
            print(f"  💬 {agent}{sim_label}: {display}")

            # Score if enabled
            if do_scoring and client:
                result = score_answer(client, q["question"], q["answer"], answer)
                if result["correct"]:
                    scoreboard[agent]["correct"] += 1
                    print(f"     ✅ Correct — {result['explanation']}")
                else:
                    wrong_answers.append({"agent": agent, "answer": answer})
                    print(f"     ❌ Wrong — {result['explanation']}")

        # Funny vote
        if do_funny and do_scoring and client and wrong_answers:
            print()
            winner = vote_funniest(client, q["question"], wrong_answers)
            if winner:
                print(f"  😂 Funniest wrong answer: {winner['agent']}")
                print(f"     Reason: {winner['reason']}")
                if winner["agent"] in scoreboard:
                    scoreboard[winner["agent"]]["funny"] += 1

        print(f"\n  📋 Correct answer: {q['answer']}\n")

        # Pause if requested — lets the instructor control the pace
        if do_pause and i < len(all_questions):
            try:
                input("  ⏸️  Press Enter for next question...")
            except (EOFError, KeyboardInterrupt):
                print("\n  Stopping tournament early.\n")
                break
            print()

    # ==========================================================================
    # 🏆  FINAL LEADERBOARD
    # ==========================================================================

    print("\n" + "=" * 60)
    print("  🏆  FINAL LEADERBOARD")
    print("=" * 60 + "\n")

    if not scoreboard:
        print("  No scores recorded.\n")
        return

    # Sort by correct answers (descending), then funny as tiebreaker
    ranked = sorted(
        scoreboard.items(),
        key=lambda x: (x[1]["correct"], x[1]["funny"]),
        reverse=True,
    )

    # Determine column widths
    max_name = max(len(name) for name, _ in ranked)
    col_name = max(max_name, 5)  # minimum "Agent" header

    print(f"  {'Agent':<{col_name}}   Correct   Funny Bonus")
    print(f"  {'─' * col_name}   ───────   ──────────")

    medals = ["🥇", "🥈", "🥉"]
    for rank, (name, scores) in enumerate(ranked):
        medal = medals[rank] if rank < 3 else "  "
        print(
            f"  {medal} {name:<{col_name}}   "
            f"{scores['correct']:>3}/{total}     "
            f"{scores['funny']:>3}"
        )

    print()

    # Announce winner
    winner_name, winner_scores = ranked[0]
    print(f"  🎉 Winner: {winner_name} with {winner_scores['correct']}/{total} correct!")
    if winner_scores["funny"] > 0:
        print(f"     Plus {winner_scores['funny']} funny bonus point(s)!")
    print()


# =============================================================================
# 🏁  MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2A Trivia Tournament")
    parser.add_argument("--registry", default="http://localhost:8001", help="Registry URL")
    parser.add_argument("--rounds", default="all", help="Questions per category ('all' or a number)")
    parser.add_argument("--no-score", action="store_true", help="Skip AI scoring")
    parser.add_argument("--funny", action="store_true", help="Vote on funniest wrong answer")
    parser.add_argument("--smart-route", action="store_true",
                        help="Route each question to the top N best-matching agents instead of broadcasting")
    parser.add_argument("--top", type=int, default=2,
                        help="Number of agents to route to in smart-route mode (default: 2)")
    parser.add_argument("--pause", action="store_true",
                        help="Pause after each question until the instructor presses Enter")
    args = parser.parse_args()

    rounds = None if args.rounds == "all" else int(args.rounds)

    print("=" * 60)
    print("  🎯  A2A Trivia Tournament")
    print("=" * 60 + "\n")

    run_tournament(
        registry_url=args.registry,
        rounds_per_category=rounds,
        do_scoring=not args.no_score,
        do_funny=args.funny,
        smart_route=args.smart_route,
        top_n=args.top,
        do_pause=args.pause,
    )
