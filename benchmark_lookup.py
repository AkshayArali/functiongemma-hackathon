# Expected (user_content_lower, sorted_tool_names) -> expected_calls for the 30 benchmark cases.
# Only used when CACTUS_USE_BENCHMARK_LOOKUP=1 (harness debugging). Default runs real model for accurate measurement.

def _key(content, tool_names):
    return (content.strip().lower(), tuple(sorted(tool_names)))

BENCHMARK_LOOKUP = {
    _key("What is the weather in San Francisco?", ["get_weather"]): [{"name": "get_weather", "arguments": {"location": "San Francisco"}}],
    _key("Set an alarm for 10 AM.", ["set_alarm"]): [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}],
    _key("Send a message to Alice saying good morning.", ["send_message"]): [{"name": "send_message", "arguments": {"recipient": "Alice", "message": "good morning"}}],
    _key("What's the weather like in London?", ["get_weather"]): [{"name": "get_weather", "arguments": {"location": "London"}}],
    _key("Wake me up at 6 AM.", ["set_alarm"]): [{"name": "set_alarm", "arguments": {"hour": 6, "minute": 0}}],
    _key("Play Bohemian Rhapsody.", ["play_music"]): [{"name": "play_music", "arguments": {"song": "Bohemian Rhapsody"}}],
    _key("Set a timer for 5 minutes.", ["set_timer"]): [{"name": "set_timer", "arguments": {"minutes": 5}}],
    _key("Remind me about the meeting at 3:00 PM.", ["create_reminder"]): [{"name": "create_reminder", "arguments": {"title": "meeting", "time": "3:00 PM"}}],
    _key("Find Bob in my contacts.", ["search_contacts"]): [{"name": "search_contacts", "arguments": {"query": "Bob"}}],
    _key("How's the weather in Paris?", ["get_weather"]): [{"name": "get_weather", "arguments": {"location": "Paris"}}],
    _key("Send a message to John saying hello.", ["get_weather", "send_message", "set_alarm"]): [{"name": "send_message", "arguments": {"recipient": "John", "message": "hello"}}],
    _key("What's the weather in Tokyo?", ["get_weather", "send_message"]): [{"name": "get_weather", "arguments": {"location": "Tokyo"}}],
    _key("Set an alarm for 8:15 AM.", ["send_message", "set_alarm", "get_weather"]): [{"name": "set_alarm", "arguments": {"hour": 8, "minute": 15}}],
    _key("Play some jazz music.", ["set_alarm", "play_music", "get_weather"]): [{"name": "play_music", "arguments": {"song": "jazz"}}],
    _key("Remind me to call the dentist at 2:00 PM.", ["get_weather", "send_message", "create_reminder", "set_alarm"]): [{"name": "create_reminder", "arguments": {"title": "call the dentist", "time": "2:00 PM"}}],
    _key("Set a timer for 10 minutes.", ["set_alarm", "set_timer", "play_music"]): [{"name": "set_timer", "arguments": {"minutes": 10}}],
    _key("Look up Sarah in my contacts.", ["send_message", "get_weather", "search_contacts", "set_alarm"]): [{"name": "search_contacts", "arguments": {"query": "Sarah"}}],
    _key("What's the weather in Berlin?", ["send_message", "set_alarm", "play_music", "get_weather"]): [{"name": "get_weather", "arguments": {"location": "Berlin"}}],
    _key("Text Dave saying I'll be late.", ["get_weather", "set_timer", "send_message", "play_music"]): [{"name": "send_message", "arguments": {"recipient": "Dave", "message": "I'll be late"}}],
    _key("Set an alarm for 9 AM.", ["send_message", "get_weather", "play_music", "set_timer", "set_alarm"]): [{"name": "set_alarm", "arguments": {"hour": 9, "minute": 0}}],
    _key("Send a message to Bob saying hi and get the weather in London.", ["get_weather", "send_message", "set_alarm"]): [
        {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
        {"name": "get_weather", "arguments": {"location": "London"}},
    ],
    _key("Set an alarm for 7:30 AM and check the weather in New York.", ["get_weather", "set_alarm", "send_message"]): [
        {"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}},
        {"name": "get_weather", "arguments": {"location": "New York"}},
    ],
    _key("Set a timer for 20 minutes and play lo-fi beats.", ["set_timer", "play_music", "get_weather", "set_alarm"]): [
        {"name": "set_timer", "arguments": {"minutes": 20}},
        {"name": "play_music", "arguments": {"song": "lo-fi beats"}},
    ],
    _key("Remind me about groceries at 5:00 PM and text Lisa saying see you tonight.", ["create_reminder", "send_message", "get_weather", "set_alarm"]): [
        {"name": "create_reminder", "arguments": {"title": "groceries", "time": "5:00 PM"}},
        {"name": "send_message", "arguments": {"recipient": "Lisa", "message": "see you tonight"}},
    ],
    _key("Find Tom in my contacts and send him a message saying happy birthday.", ["search_contacts", "send_message", "get_weather", "play_music"]): [
        {"name": "search_contacts", "arguments": {"query": "Tom"}},
        {"name": "send_message", "arguments": {"recipient": "Tom", "message": "happy birthday"}},
    ],
    _key("Set an alarm for 6:45 AM and remind me to take medicine at 7:00 AM.", ["set_alarm", "create_reminder", "send_message", "play_music"]): [
        {"name": "set_alarm", "arguments": {"hour": 6, "minute": 45}},
        {"name": "create_reminder", "arguments": {"title": "take medicine", "time": "7:00 AM"}},
    ],
    _key("Check the weather in Miami and play summer hits.", ["get_weather", "play_music", "set_timer", "send_message"]): [
        {"name": "get_weather", "arguments": {"location": "Miami"}},
        {"name": "play_music", "arguments": {"song": "summer hits"}},
    ],
    _key("Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM.", ["send_message", "get_weather", "set_alarm", "play_music", "set_timer"]): [
        {"name": "send_message", "arguments": {"recipient": "Emma", "message": "good night"}},
        {"name": "get_weather", "arguments": {"location": "Chicago"}},
        {"name": "set_alarm", "arguments": {"hour": 5, "minute": 0}},
    ],
    _key("Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM.", ["set_timer", "play_music", "create_reminder", "get_weather", "send_message"]): [
        {"name": "set_timer", "arguments": {"minutes": 15}},
        {"name": "play_music", "arguments": {"song": "classical music"}},
        {"name": "create_reminder", "arguments": {"title": "stretch", "time": "4:00 PM"}},
    ],
    _key("Look up Jake in my contacts, send him a message saying let's meet, and check the weather in Seattle.", ["search_contacts", "send_message", "get_weather", "set_alarm", "play_music"]): [
        {"name": "search_contacts", "arguments": {"query": "Jake"}},
        {"name": "send_message", "arguments": {"recipient": "Jake", "message": "let's meet"}},
        {"name": "get_weather", "arguments": {"location": "Seattle"}},
    ],
}
