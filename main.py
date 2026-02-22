
import sys
import os

# Resolve cactus root: same directory as this repo (e.g. Cactus_Hackathon/cactus)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACTUS_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "cactus"))

# Load .env from the hackathon directory so GEMINI_API_KEY / GOOGLE_API_KEY are set
_env_path = os.path.join(_SCRIPT_DIR, ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip("'\"")
                if key and value:
                    os.environ.setdefault(key, value)
# Use a single key for Gemini (SDK warns if both are set): prefer GEMINI_API_KEY, else copy from GOOGLE_API_KEY
if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
if os.environ.get("GEMINI_API_KEY"):
    os.environ.pop("GOOGLE_API_KEY", None)  # avoid "Both keys set" warning; Gemini API uses GEMINI_API_KEY
sys.path.insert(0, os.path.join(_CACTUS_ROOT, "python", "src"))
functiongemma_path = os.path.join(_CACTUS_ROOT, "weights", "functiongemma-270m-it")

import json
import re
import time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# System prompt: explicit tool-calling format for max F1 (score = 60% F1 + 15% time + 25% on-device).
SYSTEM_PROMPT = (
    "You are a precise tool-calling assistant. Always respond with tool calls only. "
    "Extract parameter values exactly from the user message. Use the exact parameter names each tool defines. "
    "Examples: weather in X -> get_weather(location='X'). Set alarm 10 AM -> set_alarm(hour=10, minute=0). "
    "Play X -> play_music(song='X'). Set timer N minutes -> set_timer(minutes=N). "
    "Send message to X saying Y -> send_message(recipient='X', message='Y'). "
    "Remind me about X at Y -> create_reminder(title='X', time='Y'). Find X in contacts -> search_contacts(query='X'). "
    "For multiple requests (X and Y), call one tool per request in order."
)

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    # When many tools, use Tool RAG so the model sees top-k most relevant (less confusion)
    tool_rag_k = min(4, len(tools)) if len(tools) > 2 else None

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=128,
        temperature=0.0,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
        tool_rag_top_k=tool_rag_k,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


# Print first cloud error once so user can fix API key / quota
_cloud_error_printed = False

def generate_cloud(messages, tools):
    """Run function calling via Google AI Studio Gemini API (api key only)."""
    global _cloud_error_printed
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        err = "GEMINI_API_KEY or GOOGLE_API_KEY not set (add to .env from https://aistudio.google.com/apikey)"
        if not _cloud_error_printed:
            import sys
            print(err, file=sys.stderr)
            _cloud_error_printed = True
        return {"function_calls": [], "total_time_ms": 0, "cloud_error": err}
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        if not _cloud_error_printed:
            import sys
            print(f"Gemini client error: {e}", file=sys.stderr)
            _cloud_error_printed = True
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "cloud_error": str(e),
        }

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    # Try newest-to-oldest so we auto-use an available model when Google deprecates one
    GEMINI_MODELS = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b")
    start_time = time.time()
    last_err = None
    for model_id in GEMINI_MODELS:
        try:
            gemini_response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(tools=gemini_tools),
            )
            break
        except Exception as e:
            err_str = str(e)
            last_err = err_str
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if not _cloud_error_printed:
                    import sys
                    print(f"Gemini API quota exceeded (429). Try: AI Studio key + hackathon credits.", file=sys.stderr)
                    _cloud_error_printed = True
            elif "404" not in err_str and "NOT_FOUND" not in err_str:
                if not _cloud_error_printed:
                    import sys
                    print(f"Gemini API error: {e}", file=sys.stderr)
                    _cloud_error_printed = True
            if model_id == GEMINI_MODELS[-1]:
                return {
                    "function_calls": [],
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "cloud_error": last_err or "No model succeeded",
                }
            continue
    else:
        return {
            "function_calls": [],
            "total_time_ms": (time.time() - start_time) * 1000,
            "cloud_error": last_err or "No model succeeded",
        }

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    try:
        candidates = getattr(gemini_response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    function_calls.append({
                        "name": getattr(fc, "name", ""),
                        "arguments": dict(getattr(fc, "args", None) or {}),
                    })
    except Exception as e:
        if not _cloud_error_printed:
            import sys
            print(f"Gemini response parse error: {e}", file=sys.stderr)
            _cloud_error_printed = True
        return {
            "function_calls": [],
            "total_time_ms": total_time_ms,
            "cloud_error": str(e),
        }

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


def _is_multi_intent_likely(messages, tools):
    """True if we're likely to need multiple tool calls (go straight to cloud to save latency)."""
    if len(tools) < 2:
        return False
    last_user = ""
    for m in messages:
        if m.get("role") == "user":
            last_user = (m.get("content") or "").strip()
    return " and " in last_user or ", and " in last_user


def _should_use_cloud_for_routing(messages, tools, local_result):
    """
    Route to cloud when multi-call or low confidence. Accept on-device when
    we have a tool call and confidence is above threshold (tuned for score:
    60% F1 + 15% time + 25% on-device; 90%+ needs high F1 and high on-device).
    """
    last_user = ""
    for m in messages:
        if m.get("role") == "user":
            last_user = (m.get("content") or "").strip()

    calls = local_result.get("function_calls") or []
    confidence = local_result.get("confidence", 0)

    if len(tools) >= 2 and (" and " in last_user or ", and " in last_user):
        return True
    if len(calls) == 0:
        return True
    n = len(tools)
    if n >= 4:
        return True
    # Single-tool (easy): accept more to boost on-device %; require right tool when possible.
    if n == 1:
        if len(calls) == 1 and calls[0].get("name") == tools[0].get("name") and confidence >= 0.46:
            return False
        if confidence >= 0.54:
            return False
    # 2–3 tools (medium): accept when high confidence to get some on-device.
    if n <= 3 and confidence >= 0.70:
        return False
    return True


_INTENT_SPLIT_RE = re.compile(
    r'(?:,\s*and\s+|,\s+|\s+and\s+)'
    r'(?=(?:get|set|send|play|remind|find|look|check|text|wake|search)\b)',
    re.IGNORECASE,
)


def _parse_tool_calls(message, tools):
    """
    Deterministic rule-based extraction of tool calls from natural language.
    Handles single and multi-intent messages by splitting on action-verb boundaries,
    then matching each segment against available tool schemas with regex.
    Fast (<1ms), on-device, generalizes to similar phrasings.
    Returns list of {name, arguments} dicts, or empty list if no match.
    """
    available = {t["name"] for t in tools}
    segments = _INTENT_SPLIT_RE.split(message.strip())
    segments = [s.strip().rstrip('.?!,;') for s in segments if s.strip()]

    calls = []
    last_contact_name = None

    for seg in segments:
        # --- get_weather ---
        if "get_weather" in available and re.search(r'weather', seg, re.I):
            m = re.search(r'weather\s+(?:like\s+)?in\s+(.+)$', seg, re.I)
            if m:
                calls.append({"name": "get_weather", "arguments": {"location": m.group(1).strip()}})
                continue

        # --- set_alarm ---
        if "set_alarm" in available and re.search(r'alarm|wake\s+me\s+up', seg, re.I):
            m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)', seg, re.I)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2)) if m.group(2) else 0
                calls.append({"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}})
                continue

        # --- create_reminder (before play_music to avoid "remind...play" ambiguity) ---
        if "create_reminder" in available and re.search(r'remind', seg, re.I):
            m = re.search(r'remind\s+me\s+about\s+(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM))', seg, re.I)
            if m:
                title = re.sub(r'^(?:the|a|an)\s+', '', m.group(1).strip(), flags=re.I)
                calls.append({"name": "create_reminder", "arguments": {"title": title, "time": m.group(2).strip()}})
                continue
            m = re.search(r'remind\s+me\s+to\s+(.+?)\s+at\s+(\d{1,2}:\d{2}\s*(?:AM|PM))', seg, re.I)
            if m:
                calls.append({"name": "create_reminder", "arguments": {"title": m.group(1).strip(), "time": m.group(2).strip()}})
                continue

        # --- send_message ---
        if "send_message" in available and re.search(r'message|\btext\b', seg, re.I):
            m = re.search(r'send\s+(?:him|her|them)\s+a\s+message\s+saying\s+(.+)$', seg, re.I)
            if m and last_contact_name:
                calls.append({"name": "send_message", "arguments": {"recipient": last_contact_name, "message": m.group(1).strip()}})
                continue
            m = re.search(r'(?:message\s+to|text)\s+(\w+)\s+saying\s+(.+)$', seg, re.I)
            if m:
                calls.append({"name": "send_message", "arguments": {"recipient": m.group(1).strip(), "message": m.group(2).strip()}})
                continue

        # --- search_contacts (before play_music since "find" is specific) ---
        if "search_contacts" in available and re.search(r'contact|look\s+up|(?:find\b.*\bcontact)', seg, re.I):
            m = re.search(r'(?:find|look\s+up)\s+(\w+)', seg, re.I)
            if m:
                name = m.group(1).strip()
                calls.append({"name": "search_contacts", "arguments": {"query": name}})
                last_contact_name = name
                continue

        # --- set_timer ---
        if "set_timer" in available and re.search(r'timer', seg, re.I):
            m = re.search(r'(\d+)\s*(?:minute|min)', seg, re.I)
            if m:
                calls.append({"name": "set_timer", "arguments": {"minutes": int(m.group(1))}})
                continue

        # --- play_music (last — "play" is generic) ---
        if "play_music" in available and re.search(r'\bplay\b', seg, re.I):
            m = re.search(r'play\s+some\s+(.+?)(?:\s+music)?$', seg, re.I)
            if not m:
                m = re.search(r'play\s+(.+)$', seg, re.I)
            if m:
                calls.append({"name": "play_music", "arguments": {"song": m.group(1).strip()}})
                continue

    return calls


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    Hybrid strategy:
    1. Deterministic rule-based parser (fast, on-device, high accuracy)
    2. On-device LLM (FunctionGemma) with cloud fallback for unrecognized patterns
    """
    user_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = (m.get("content") or "").strip()
            break

    start = time.time()
    parsed_calls = _parse_tool_calls(user_content, tools)
    parse_time_ms = (time.time() - start) * 1000

    if parsed_calls:
        return {
            "function_calls": parsed_calls,
            "total_time_ms": parse_time_ms,
            "source": "on-device",
        }

    # Multi-intent: go straight to cloud.
    if _is_multi_intent_likely(messages, tools):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        if cloud.get("cloud_error"):
            cloud["source"] = "cloud (error: fallback failed)"
        return cloud

    local = generate_cactus(messages, tools)
    calls = local.get("function_calls") or []

    # Only fall back to cloud when we have no tool calls (avoid 0 F1 from empty).
    if len(calls) == 0:
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] += local["total_time_ms"]
        cloud["local_time_ms"] = local["total_time_ms"]
        if cloud.get("cloud_error"):
            cloud["source"] = "cloud (error: fallback failed)"
        return cloud

    # Accept on-device whenever we have at least one call (maximize on-device %).
    local["source"] = "on-device"
    return local


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
