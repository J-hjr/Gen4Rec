import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv() # need to add .env at each person's side, then pip install python-dotenv openai


SYSTEM_INSTRUCTIONS = """
You are a music taste profiler and prompt engineer for text-to-music generation.

Your job is to convert a structured listener summary into:
1. a human-readable user profile paragraph
2. a Suno-friendly music generation prompt

Requirements:
- The profile paragraph should sound natural, specific, and musically informed.
- The Suno prompt should be concise but vivid, and optimized for text-to-music generation.
- Focus on genre, vocal style, instrumentation, mood, production texture, energy, and pacing.
- Do NOT mention retrieval scores, JSON, or metadata field names.
- Do NOT list too many song titles or artist names.
- Do NOT ask questions.
- Avoid saying "similar to [artist]" or asking the model to imitate a copyrighted artist.
- Avoid overly generic wording like "nice song" or "good vibes".
- The Suno prompt should feel like a production brief.
- Include a short "avoid" clause in the Suno prompt when helpful.
Return valid JSON only.
"""


def load_summary_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_output(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_user_prompt(summary: Dict[str, Any]) -> str:
    return f"""
Here is a structured listener summary:

{json.dumps(summary, indent=2, ensure_ascii=False)}

Please generate a JSON object with exactly these keys:
- "profile_paragraph": a single paragraph describing the listener's music taste
- "suno_generation_prompt": a concise but specific prompt for Suno
- "style_keywords": an array of 5 to 8 short keywords or phrases

Guidance for the profile paragraph:
- 90 to 140 words
- sound polished, natural, and specific
- describe likely taste in genre, emotional tone, instrumentation, and overall aesthetic

Guidance for the Suno prompt:
- must be no more than 500 characters total, including spaces and punctuation
- should be directly usable as a text-to-music prompt
- compact, vivid, and specific
- mention likely genre blend, vocal type, instrumentation, mood, arrangement density, pacing, and production feel
- include a short "avoid" clause if helpful
- do not mention user, profile, JSON, recommendation, or metadata

Return JSON only.
""".strip()


def generate_music_prompt(
    summary: Dict[str, Any],
    model: str = "gpt-5.3-mini",
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Please export your OpenAI API key first."
        )

    client = OpenAI(api_key=api_key)

    print("Calling OpenAI with summary for user:", summary.get("user_id"))
    print("Received response from OpenAI")

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": build_user_prompt(summary),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "music_profile_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "profile_paragraph": {"type": "string"},
                        "suno_generation_prompt": {"type": "string"},
                        "style_keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 5,
                            "maxItems": 8
                        }
                    },
                    "required": [
                        "profile_paragraph",
                        "suno_generation_prompt",
                        "style_keywords"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }
        },
    )

    def trim_to_char_limit(text: str, max_chars: int = 500) -> str:
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text

        trimmed = text[:max_chars].rstrip()

        if " " in trimmed:
            trimmed = trimmed.rsplit(" ", 1)[0]
        return trimmed

    output_text = response.output_text
    parsed = json.loads(output_text)

    suno_prompt = trim_to_char_limit(parsed["suno_generation_prompt"], max_chars=500)

    final_output = {
        "user_id": summary.get("user_id"),
        "input_summary": summary,
        "profile_paragraph": parsed["profile_paragraph"],
        "suno_generation_prompt": suno_prompt,
        "style_keywords": parsed["style_keywords"],
    }
    return final_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate profile paragraph and Suno prompt from condensed listener summary."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to condensed summary JSON."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save generated profile/prompt JSON."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.3-mini",
        help="OpenAI model name."
    )

    args = parser.parse_args()

    summary = load_summary_json(args.input)
    result = generate_music_prompt(summary, model=args.model)
    save_output(result, args.output)

    print("\nGenerated profile paragraph and Suno prompt.\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
