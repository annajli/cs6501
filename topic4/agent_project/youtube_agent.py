"""
YouTube Educational Video Agent

Given a YouTube URL, this agent:
1. Extracts the video transcript using youtube_transcript_api
2. Generates a summary, key concepts, and quiz questions using GPT-4o-mini
"""

from dotenv import load_dotenv
load_dotenv()

import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# ============================================================================
# TOOL
# ============================================================================

@tool
def get_youtube_transcript(url: str) -> str:
    """
    Fetch the transcript of a YouTube video given a full URL or video ID.

    Args:
        url: A YouTube URL (any format) or bare 11-character video ID.

    Returns:
        The full transcript as a single string, or an error message.
    """
    # Extract video ID from common YouTube URL formats
    patterns = [
        r'(?:v=)([0-9A-Za-z_-]{11})',        # youtube.com/watch?v=ID
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',  # youtu.be/ID
        r'(?:embed\/)([0-9A-Za-z_-]{11})',    # youtube.com/embed/ID
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',   # youtube.com/shorts/ID
    ]

    video_id = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break

    # Fall back to treating input as a raw video ID
    if not video_id:
        if re.match(r'^[0-9A-Za-z_-]{11}$', url.strip()):
            video_id = url.strip()
        else:
            return "Error: Could not extract a video ID from the provided input."

    api = YouTubeTranscriptApi()

    # Try English first, then fall back to any available language
    for kwargs in [{"languages": ["en", "en-US", "en-GB"]}, {}]:
        try:
            fetched = api.fetch(video_id, **kwargs)
            # v0.6.x returns FetchedTranscript (iterable of snippet objects)
            # older versions return list of dicts — handle both
            parts = []
            for entry in fetched:
                parts.append(entry.text if hasattr(entry, "text") else entry["text"])
            return f"[Transcript for video {video_id}]\n\n{' '.join(parts)}"
        except Exception as e:
            last_error = e

    print(f"[DEBUG] Transcript fetch failed for '{video_id}': {type(last_error).__name__}: {last_error}")
    return (
        f"Error fetching transcript for video '{video_id}': {type(last_error).__name__}: {last_error}\n"
        "Possible reasons: video is private, age-restricted, has no captions, "
        "or the video ID is incorrect."
    )


# ============================================================================
# AGENT
# ============================================================================

SYSTEM_PROMPT = """You are an educational assistant that helps students learn from YouTube videos.

When a student provides a YouTube URL:
1. Call get_youtube_transcript to fetch the transcript.
2. Analyze the transcript and produce ALL THREE sections below.

---

## Summary
A clear, concise summary (3–5 sentences) of the video's main message and purpose.

## Key Concepts
A bulleted list of 5–8 important concepts, terms, or ideas from the video.
For each concept provide a one-sentence explanation.

## Quiz Questions
5 questions that test understanding of the material.
Mix multiple-choice and short-answer formats.
Include the correct answer for each question.

---

Always produce all three sections, even if not explicitly asked.
If the transcript cannot be fetched, explain why and suggest alternatives."""

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [get_youtube_transcript], prompt=SYSTEM_PROMPT)


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("=" * 60)
    print("  YouTube Educational Video Analyzer")
    print("=" * 60)
    print("Paste a YouTube URL to get a summary, key concepts,")
    print("and quiz questions. Type 'exit' to quit.\n")

    while True:
        url = input("YouTube URL: ").strip()

        if not url:
            continue
        if url.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("\nFetching transcript and analyzing... please wait.\n")

        try:
            result = agent.invoke({
                "messages": [("user", f"Analyze this YouTube video: {url}")]
            })
            final_message = result["messages"][-1]
            print(final_message.content)
        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
