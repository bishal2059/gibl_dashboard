import re
import json

def extract_insights_json(text: str) -> dict:
    """
    Extracts and parses the JSON content from a structured markdown-like insights message.

    Args:
        text (str): The input message containing the JSON within triple backticks.

    Returns:
        dict: Parsed JSON object as a Python dictionary.

    Raises:
        ValueError: If no JSON block is found or JSON is malformed.
    """
    # Extract the JSON block between ```json ... ```
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in the message.")

    json_str = match.group(1).strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}")
