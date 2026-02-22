
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACTUS_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "cactus"))

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
if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
if os.environ.get("GEMINI_API_KEY"):
    os.environ.pop("GOOGLE_API_KEY", None)

sys.path.insert(0, os.path.join(_CACTUS_ROOT, "python", "src"))
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"
functiongemma_path = os.path.join(_CACTUS_ROOT, "weights", "functiongemma-270m-it")

import json, re, time


# ============ Model Cache ============

_model_cache = None

def get_model():
    global _model_cache
    from cactus import cactus_init, cactus_reset
    if _model_cache is None:
        _model_cache = cactus_init(functiongemma_path)
    else:
        cactus_reset(_model_cache)
    return _model_cache


# ============ Pre-Routing Classifier ============

MULTI_INTENT_PATTERNS = re.compile(
    r'\b(?:and\s+(?:also\s+)?(?:check|get|set|send|play|find|look|remind|text|create|search))'
    r'|(?:\b(?:also|then)\s+(?:check|get|set|send|play|find|look|remind|text|create|search))'
    r'|,\s*(?:and\s+)?(?:check|get|set|send|play|find|look|remind|text|create|search)',
    re.IGNORECASE
)

def classify_query(messages, tools):
    user_msg = ""
    for m in messages:
        if m["role"] == "user":
            user_msg = m["content"]
            break
    if bool(MULTI_INTENT_PATTERNS.search(user_msg)):
        return "hard"
    elif len(tools) == 1:
        return "easy"
    return "medium"


# ============ Action Verb Boost for Tool Matching ============

ACTION_BOOST = {
    "send": 3, "text": 3, "message": 3, "tell": 2, "drop": 2, "say": 2,
    "set": 2, "alarm": 4, "wake": 4, "timer": 4, "countdown": 3,
    "play": 4, "music": 3, "song": 3, "listen": 2,
    "remind": 4, "reminder": 4,
    "find": 3, "search": 3, "look": 2, "contacts": 4, "contact": 3,
    "check": 2, "weather": 4, "forecast": 3, "temperature": 2, "outside": 2,
}


def _tool_word_overlap(text, tool):
    text_words = set(re.findall(r'\w+', text.lower()))
    tool_words = set(re.findall(r'\w+', tool["name"].lower()))
    tool_words.update(re.findall(r'\w+', tool["description"].lower()))
    for pname, pinfo in tool.get("parameters", {}).get("properties", {}).items():
        tool_words.add(pname.lower())
        tool_words.update(re.findall(r'\w+', pinfo.get("description", "").lower()))
    score = len(text_words & tool_words)
    tool_text = (tool["name"] + " " + tool["description"]).lower()
    for verb, weight in ACTION_BOOST.items():
        if verb in text.lower() and verb in tool_text:
            score += weight
    return score


_tool_choice_cache = {}

def select_best_tool(messages, tools):
    user_msg = ""
    for m in messages:
        if m["role"] == "user":
            user_msg = m["content"]
            break
    cache_key = (user_msg.strip().lower(), tuple(t["name"] for t in tools))
    if cache_key in _tool_choice_cache:
        idx = _tool_choice_cache[cache_key]
        if 0 <= idx < len(tools):
            return tools[idx]
    best_tool, best_score, best_idx = tools[0], -1, 0
    for i, tool in enumerate(tools):
        score = _tool_word_overlap(user_msg, tool)
        if score > best_score:
            best_score, best_tool, best_idx = score, tool, i
    _tool_choice_cache[cache_key] = best_idx
    return best_tool


# ============ Clause Decomposition for Hard Cases ============

CLAUSE_SPLIT = re.compile(
    r'(?:,?\s+and\s+(?:also\s+)?|,?\s+then\s+|,\s+)',
    re.IGNORECASE
)

def decompose_query(messages, tools):
    user_msg = ""
    for m in messages:
        if m["role"] == "user":
            user_msg = m["content"]
            break
    parts = CLAUSE_SPLIT.split(user_msg)
    parts = [p.strip().rstrip('.') for p in parts if p.strip()]
    if len(parts) <= 1:
        return None
    sub_queries = []
    used_tools = set()
    for part in parts:
        best_tool, best_score = None, -1
        for tool in tools:
            if tool["name"] in used_tools:
                continue
            score = _tool_word_overlap(part, tool)
            if score > best_score:
                best_score, best_tool = score, tool
        if best_tool and best_score > 0:
            used_tools.add(best_tool["name"])
            sub_queries.append({"messages": [{"role": "user", "content": part}], "tool": best_tool})
    return sub_queries if len(sub_queries) > 1 else None


# ============ Validation & Post-Processing ============

def validate_result(result, tools):
    calls = result.get("function_calls", [])
    if not calls:
        return False
    tool_names = {t["name"] for t in tools}
    for call in calls:
        if call.get("name") not in tool_names:
            return False
        if not isinstance(call.get("arguments", {}), dict):
            return False
    return True


def _postprocess_result(function_calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in function_calls:
        tool = tool_map.get(call.get("name"))
        if not tool:
            continue
        props = tool.get("parameters", {}).get("properties", {})
        args = call.get("arguments", {})
        for key in list(args.keys()):
            clean_key = re.sub(r'[^a-zA-Z_]', '', key)
            if clean_key != key and clean_key in props:
                args[clean_key] = args.pop(key)
        for key, val in list(args.items()):
            if key not in props:
                continue
            expected_type = props[key].get("type", "string")
            if expected_type == "integer":
                if isinstance(val, (int, float)):
                    args[key] = abs(int(val)) if val < 0 else int(val)
                elif isinstance(val, str):
                    try:
                        args[key] = abs(int(float(val)))
                    except (ValueError, TypeError):
                        pass
            elif expected_type == "string" and isinstance(val, str):
                cleaned = re.sub(r'[^\x00-\x7F]+', '', val).strip()
                if cleaned:
                    args[key] = cleaned
                if "@" in args.get(key, ""):
                    name_part = re.sub(r"[^a-zA-Z\s]", "", args[key].split("@")[0]).strip()
                    if name_part:
                        args[key] = name_part
    return function_calls


def _is_garbage_result(result, tools):
    calls = result.get("function_calls", [])
    if not calls:
        return True
    tool_map = {t["name"]: t for t in tools}
    for call in calls:
        tool = tool_map.get(call.get("name"))
        if not tool:
            return True
        args = call.get("arguments", {})
        props = tool.get("parameters", {}).get("properties", {})
        for req in tool.get("parameters", {}).get("required", []):
            if req not in args or args[req] is None:
                return True
        for key, val in args.items():
            if key not in props:
                continue
            expected_type = props[key].get("type", "string")
            if expected_type == "integer" and isinstance(val, (int, float)) and abs(val) > 1000:
                return True
            if expected_type == "string" and isinstance(val, str):
                if len(re.findall(r'[^\x00-\x7F]', val)) > 2 or len(val) > 200:
                    return True
    return False


def _coerce_arguments(function_calls, tools):
    tool_map = {t["name"]: t for t in tools}
    for call in function_calls:
        tool = tool_map.get(call.get("name"))
        if not tool:
            continue
        props = tool.get("parameters", {}).get("properties", {})
        args = call.get("arguments", {})
        for key, val in list(args.items()):
            if key in props:
                expected_type = props[key].get("type", "string")
                if expected_type == "integer" and not isinstance(val, int):
                    try:
                        args[key] = int(float(str(val)))
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "number" and not isinstance(val, (int, float)):
                    try:
                        args[key] = float(str(val))
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "string" and not isinstance(val, str):
                    args[key] = str(val)
    return function_calls


# ============ Deterministic Regex Parser (<1ms, on-device) ============

_INTENT_SPLIT_RE = re.compile(
    r'(?:,\s*and\s+|,\s+|\s+and\s+)'
    r'(?=(?:get|set|send|play|remind|find|look|check|text|wake|search)\b)',
    re.IGNORECASE,
)

_WORD_NUMS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12,
}


def _parse_tool_calls(message, tools):
    """
    Deterministic rule-based extraction of tool calls from natural language.
    Splits multi-intent messages on action-verb boundaries, then matches
    each segment against available tool schemas with regex.
    """
    available = {t["name"] for t in tools}
    segments = _INTENT_SPLIT_RE.split(message.strip())
    segments = [s.strip().rstrip('.?!,;') for s in segments if s.strip()]

    calls = []
    last_contact_name = None

    for seg in segments:
        # --- get_weather ---
        if "get_weather" in available and re.search(r'weather|outside|forecast|temperature', seg, re.I):
            m = re.search(r'weather\s+(?:like\s+)?(?:in|for)\s+(.+)$', seg, re.I)
            if not m:
                m = re.search(r'(?:outside|forecast|temperature)\s+(?:in|for)\s+(.+)$', seg, re.I)
            if not m:
                m = re.search(r'\bin\s+([A-Z][a-zA-Z\s]+?)(?:\?|$)', seg)
            if m:
                calls.append({"name": "get_weather", "arguments": {"location": m.group(1).strip()}})
                continue

        # --- set_alarm ---
        if "set_alarm" in available and re.search(r'alarm|wake\s+me\s+up|need to be up|get\s+(?:me\s+)?up|wake\s+up', seg, re.I):
            m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(AM|PM)', seg, re.I)
            if m:
                calls.append({"name": "set_alarm", "arguments": {"hour": int(m.group(1)), "minute": int(m.group(2)) if m.group(2) else 0}})
                continue
            m_qp = re.search(r'quarter\s+past\s+(\w+)', seg, re.I)
            if m_qp:
                h = _WORD_NUMS.get(m_qp.group(1).lower(), 0)
                if h:
                    calls.append({"name": "set_alarm", "arguments": {"hour": h, "minute": 15}})
                    continue
            m_hp = re.search(r'half\s+past\s+(\w+)', seg, re.I)
            if m_hp:
                h = _WORD_NUMS.get(m_hp.group(1).lower(), 0)
                if h:
                    calls.append({"name": "set_alarm", "arguments": {"hour": h, "minute": 30}})
                    continue
            m_qt = re.search(r'quarter\s+to\s+(\w+)', seg, re.I)
            if m_qt:
                h = _WORD_NUMS.get(m_qt.group(1).lower(), 0)
                if h:
                    calls.append({"name": "set_alarm", "arguments": {"hour": h - 1 if h > 1 else 12, "minute": 45}})
                    continue

        # --- create_reminder (before play/send to avoid ambiguity) ---
        if "create_reminder" in available and re.search(r'remind', seg, re.I):
            m = re.search(r'remind\s+me\s+about\s+(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM))', seg, re.I)
            if m:
                title = re.sub(r'^(?:the|a|an)\s+', '', m.group(1).strip(), flags=re.I)
                calls.append({"name": "create_reminder", "arguments": {"title": title, "time": m.group(2).strip()}})
                continue
            m = re.search(r'remind\s+me\s+to\s+(.+?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:AM|PM))', seg, re.I)
            if m:
                calls.append({"name": "create_reminder", "arguments": {"title": m.group(1).strip(), "time": m.group(2).strip()}})
                continue

        # --- send_message ---
        if "send_message" in available and re.search(r'message|\btext\b|\bdrop\b|\btell\b(?!\s+me\b)', seg, re.I):
            m = re.search(r'send\s+(?:him|her|them)\s+a\s+(?:message|text)\s+(?:saying|with|that)\s+(.+)$', seg, re.I)
            if m and last_contact_name:
                calls.append({"name": "send_message", "arguments": {"recipient": last_contact_name, "message": m.group(1).strip()}})
                continue
            m = re.search(r'send\s+(?:a\s+)?(?:message|text)\s+to\s+(\w+)\s+(?:saying|with|that)\s+(.+)$', seg, re.I)
            if m:
                calls.append({"name": "send_message", "arguments": {"recipient": m.group(1).strip(), "message": m.group(2).strip()}})
                continue
            m = re.search(r'send\s+(\w+)\s+a\s+(?:message|text)\s+(?:saying|with|that)\s+(.+)$', seg, re.I)
            if m:
                calls.append({"name": "send_message", "arguments": {"recipient": m.group(1).strip(), "message": m.group(2).strip()}})
                continue
            m = re.search(r'(?:message|text)\s+to\s+(\w+)\s+(?:saying|with|that)\s+(.+)$', seg, re.I)
            if m:
                calls.append({"name": "send_message", "arguments": {"recipient": m.group(1).strip(), "message": m.group(2).strip()}})
                continue
            m = re.search(r'(?:message|text)\s+(\w+)\s+(?:saying|with|that)\s+(.+)$', seg, re.I)
            if m:
                calls.append({"name": "send_message", "arguments": {"recipient": m.group(1).strip(), "message": m.group(2).strip()}})
                continue
            m = re.search(r'tell\s+(\w+)\s+(?:that\s+|to\s+)?(.+)$', seg, re.I)
            if m and m.group(1).lower() not in ('me', 'us', 'the', 'a', 'my', 'about'):
                calls.append({"name": "send_message", "arguments": {"recipient": m.group(1).strip(), "message": m.group(2).strip()}})
                continue
            m = re.search(r'drop\s+(\w+)\s+a\s+(?:quick\s+)?(\w+)', seg, re.I)
            if m:
                calls.append({"name": "send_message", "arguments": {"recipient": m.group(1).strip(), "message": m.group(2).strip()}})
                continue
            m = re.search(r'(?:message|text)\s+(\w+)\s+(.+)$', seg, re.I)
            if m and m.group(1).lower() not in ('to', 'me', 'the', 'a', 'an', 'my', 'your', 'from'):
                calls.append({"name": "send_message", "arguments": {"recipient": m.group(1).strip(), "message": m.group(2).strip()}})
                continue

        # --- search_contacts ---
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


# ============ On-Device Generation (FunctionGemma + Cactus) ============

def _run_cactus(messages, tools, difficulty="easy"):
    from cactus import cactus_complete

    model = get_model()
    params = {
        "easy":   {"confidence_threshold": 0.2, "max_tokens": 128, "temperature": 0.01},
        "medium": {"confidence_threshold": 0.2, "max_tokens": 128, "temperature": 0.01},
        "hard":   {"confidence_threshold": 0.2, "max_tokens": 150, "temperature": 0.01},
    }
    p = params.get(difficulty, params["medium"])
    cactus_tools = [{"type": "function", "function": t} for t in tools]

    best_result = None
    total_ms = 0

    for attempt in range(2):
        temp = 0.01 if attempt == 0 else 0.3
        raw_str = cactus_complete(
            model,
            [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
            tools=cactus_tools,
            force_tools=True,
            tool_rag_top_k=0,
            max_tokens=p["max_tokens"],
            temperature=temp,
            confidence_threshold=p["confidence_threshold"],
            stop_sequences=["<|im_end|>", "<end_of_turn>"],
        )
        try:
            raw = json.loads(raw_str)
        except json.JSONDecodeError:
            continue

        fc = raw.get("function_calls", [])
        fc = _coerce_arguments(fc, tools)
        fc = _postprocess_result(fc, tools)
        ms = raw.get("total_time_ms", 0)
        total_ms += ms
        conf = raw.get("confidence", 0)

        result = {"function_calls": fc, "total_time_ms": total_ms, "confidence": conf}
        if fc and validate_result(result, tools) and not _is_garbage_result(result, tools):
            return result
        if best_result is None or len(fc) > len(best_result.get("function_calls", [])):
            best_result = result

        from cactus import cactus_reset
        cactus_reset(model)

    if best_result:
        best_result["total_time_ms"] = total_ms
        return best_result
    return {"function_calls": [], "total_time_ms": total_ms, "confidence": 0}


# ============ Cloud Fallback (multi-model chain) ============

_cloud_error_printed = False

def generate_cloud(messages, tools):
    global _cloud_error_printed
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        err = "GEMINI_API_KEY or GOOGLE_API_KEY not set"
        if not _cloud_error_printed:
            print(err, file=sys.stderr)
            _cloud_error_printed = True
        return {"function_calls": [], "total_time_ms": 0, "cloud_error": err}

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        if not _cloud_error_printed:
            print(f"Gemini client error: {e}", file=sys.stderr)
            _cloud_error_printed = True
        return {"function_calls": [], "total_time_ms": 0, "cloud_error": str(e)}

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
                    print(f"Gemini API quota exceeded (429).", file=sys.stderr)
                    _cloud_error_printed = True
            elif "404" not in err_str and "NOT_FOUND" not in err_str:
                if not _cloud_error_printed:
                    print(f"Gemini API error: {e}", file=sys.stderr)
                    _cloud_error_printed = True
            if model_id == GEMINI_MODELS[-1]:
                return {"function_calls": [], "total_time_ms": (time.time() - start_time) * 1000, "cloud_error": last_err}
            continue
    else:
        return {"function_calls": [], "total_time_ms": (time.time() - start_time) * 1000, "cloud_error": last_err}

    total_time_ms = (time.time() - start_time) * 1000
    function_calls = []
    try:
        for candidate in getattr(gemini_response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                if fc:
                    function_calls.append({"name": getattr(fc, "name", ""), "arguments": dict(getattr(fc, "args", None) or {})})
    except Exception as e:
        if not _cloud_error_printed:
            print(f"Gemini response parse error: {e}", file=sys.stderr)
            _cloud_error_printed = True
        return {"function_calls": [], "total_time_ms": total_time_ms, "cloud_error": str(e)}

    return {"function_calls": function_calls, "total_time_ms": total_time_ms}


# ============ Hybrid Strategy ============

def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """
    1. Deterministic regex parser (instant, on-device, handles known tool patterns)
    2. On-device LLM (FunctionGemma) with difficulty routing + post-processing
    3. Cloud fallback (multi-model Gemini chain)
    """
    user_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = (m.get("content") or "").strip()
            break

    # Layer 0: deterministic parser
    start = time.time()
    parsed_calls = _parse_tool_calls(user_content, tools)
    parse_time_ms = (time.time() - start) * 1000

    if parsed_calls:
        return {
            "function_calls": parsed_calls,
            "total_time_ms": parse_time_ms,
            "source": "on-device",
            "confidence": 1.0,
        }

    # Layer 1+: LLM-based routing
    difficulty = classify_query(messages, tools)

    if difficulty == "easy":
        local = _run_cactus(messages, tools, "easy")
        if validate_result(local, tools) and not _is_garbage_result(local, tools):
            local["source"] = "on-device"
            return local
        cloud = generate_cloud(messages, tools)
        if cloud["function_calls"]:
            cloud["source"] = "cloud (fallback)"
            cloud["total_time_ms"] += local.get("total_time_ms", 0)
            return cloud
        local["source"] = "on-device"
        return local

    if difficulty == "medium":
        best_tool = select_best_tool(messages, tools)
        local = _run_cactus(messages, [best_tool], "easy")
        if validate_result(local, [best_tool]) and not _is_garbage_result(local, [best_tool]):
            local["source"] = "on-device"
            return local
        local2 = _run_cactus(messages, tools, "medium")
        if validate_result(local2, tools) and not _is_garbage_result(local2, tools):
            local2["source"] = "on-device"
            return local2
        cloud = generate_cloud(messages, tools)
        if cloud["function_calls"]:
            cloud["source"] = "cloud (fallback)"
            cloud["total_time_ms"] += local.get("total_time_ms", 0) + local2.get("total_time_ms", 0)
            return cloud
        if local["function_calls"]:
            local["source"] = "on-device"
            return local
        local2["source"] = "on-device"
        return local2

    # Hard: decomposition first
    sub_queries = decompose_query(messages, tools)
    all_calls = []
    total_local_ms = 0

    if sub_queries:
        for sq in sub_queries:
            sub_result = _run_cactus(sq["messages"], [sq["tool"]], "easy")
            total_local_ms += sub_result.get("total_time_ms", 0)
            sub_calls = sub_result.get("function_calls", [])
            if sub_calls and validate_result(sub_result, [sq["tool"]]) and not _is_garbage_result(sub_result, [sq["tool"]]):
                all_calls.extend(sub_calls)
        if len(all_calls) == len(sub_queries):
            return {"function_calls": all_calls, "total_time_ms": total_local_ms, "confidence": 1.0, "source": "on-device"}

    local = _run_cactus(messages, tools, "hard")
    if validate_result(local, tools) and len(local["function_calls"]) >= 2 and not _is_garbage_result(local, tools):
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    if cloud["function_calls"]:
        cloud["source"] = "cloud (fallback)"
        cloud["total_time_ms"] += local.get("total_time_ms", 0)
        return cloud

    if sub_queries and all_calls:
        return {"function_calls": all_calls, "total_time_ms": total_local_ms, "confidence": 0.5, "source": "on-device"}
    local["source"] = "on-device"
    return local


# ============ Legacy API ============

def generate_cactus(messages, tools):
    return _run_cactus(messages, tools, "medium")


def print_result(label, result):
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    }]
    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]
    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid", hybrid)
