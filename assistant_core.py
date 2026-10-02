#!/usr/bin/env python3
"""Local voice assistant for the Smart House LEDs.

Why this file exists: the OpenRouter API key must never reach the browser.
Anything in the web page is readable by anyone on the WiFi. So the browser
talks to this local service, and this service talks to OpenRouter and to the
ESP32.

Free pieces used:
  speech recognition - the browser's built-in Web Speech API (Chrome/Edge)
  speech output      - the browser's built-in speechSynthesis
  the language model - a free OpenRouter model

Endpoints:
  GET  /              serve the control page, so the mic is allowed
  GET  /api/state     proxied to the ESP32
  POST /api/all       proxied to the ESP32
  POST /api/led/<pin> proxied to the ESP32
  POST /api/servo/<pin> proxied to the ESP32, body is an angle
  POST /api/motor/<id> proxied to the ESP32, body is "forward 60" or "stop"
  POST /api/chat      run one turn of conversation, may act on the LEDs
  GET  /api/health    report configuration, without exposing the key

Run:  ./run.sh start
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
except ImportError:
    sys.exit("Flask and flask-cors are required.  pip install flask flask-cors")

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / ".env"

# Same shape generate.py reads, so both sides agree on the servo names.
SERVO_RE = re.compile(
    r"^([A-Za-z0-9 _-]+?)\s*:\s*(?:gpio)?\s*(\d+)\s*"
    r"(?:,\s*open\s+(-?\d+)\s*,\s*close\s+(-?\d+)\s*)?$",
    re.I)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# One question and its answer, up to four exchanges. Kept long enough that
# "apagala" and "y el azul" still work.
MAX_HISTORY_MESSAGES = 8

# Give the model the device categories and the natural-language mapping for
# simple motor controls. Tool descriptions provide the configured device list.
SYSTEM_PROMPT = """You are the voice assistant for a small Smart House prototype.

Use the available high-level tools whenever the user asks to control a device.

For door, garage and window, call set_opening with the
mechanism name and state "open" or "closed". For requests to put the baby to
sleep, calm or soothe the baby, rock or move the baby, or related requests,
call soothe_baby once. Its preset sequence stops automatically when complete.
There is no stop action for the cradle.

For lights, use turn_on_one_light for one requested color. Use
turn_on_multiple_lights when the user explicitly asks for multiple
colors. Use turn_off_lights to switch off named lights, or all lights when
requested. Use
get_lights to answer questions about light states. Use get_openings to answer
questions about doors and windows.

The fan is the only device with user-adjustable direction and speed. An
unspecified turn-on direction means forward; an explicit direction takes
precedence. Speeds may be 0 to 100 percent; when none is specified, the
server uses its default.

How to behave:
- You are being read out loud, so no No markdown, no bullet lists, no emoji. you can even say just "done" if the instructions were clear and you are sure what you did was correct.
- You may act on several devices in one turn if the user asks for it.
- If a request is unclear, ask one short clarifying question.
- If you cannot fulfil a request, say so in one sentence and do not guess."""


def load_env(path):
    """Read KEY=VALUE lines. Existing environment variables win."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


ENV = load_env(ENV_FILE)

API_KEY = os.environ.get("OPENROUTER_API_KEY") or ENV.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("OPENROUTER_MODEL") or ENV.get(
    "OPENROUTER_MODEL", "openai/gpt-4.1-mini")

# Free models share an upstream pool and get rate limited often, so we try
# each one in turn instead of giving up. Order matters: first is preferred.
FALLBACKS = [
    m.strip() for m in (
        os.environ.get("OPENROUTER_FALLBACKS")
        or ENV.get("OPENROUTER_FALLBACKS")
        or (
            "openai/gpt-4.1-mini,"
            "meta-llama/llama-3.3-70b-instruct,"
            "openai/gpt-4o-mini,"
            "google/gemini-2.5-flash-lite"
        )
    ).split(",") if m.strip()
]
MODELS = [MODEL] + [m for m in FALLBACKS if m != MODEL]
ESP32_URL = (os.environ.get("ESP32_URL") or ENV.get("ESP32_URL", "")).rstrip("/")
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT") or ENV.get("BRIDGE_PORT", 8765))

# Where the generated control page lives. Serving it from localhost is what
# unlocks the microphone, because browsers only allow it on a secure context
# (https, or localhost).
PAGE_DIR = Path(__file__).resolve().parent.parent.parent / "Website"

app = Flask(__name__, static_folder=None)

# The page may also be opened from the ESP32's own address, so allow that
# cross-origin call to the bridge.
CORS(app, resources={r"/api/*": {"origins": "*"}},
     allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])


# --- talking to the ESP32 -------------------------------------------------

def esp32_get(path, timeout=5):
    req = urllib.request.Request(ESP32_URL + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())


class Esp32Error(Exception):
    """The ESP32 answered, but with an error status. Not a connectivity problem."""

    def __init__(self, status, body):
        super().__init__(f"ESP32 replied {status}")
        self.status = status
        self.body = body


def esp32_post(path, body, timeout=5):
    req = urllib.request.Request(
        ESP32_URL + path,
        data=body.encode(),
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read().decode())
    except urllib.error.HTTPError as exc:
        # A 400 for a bad angle means the board is alive and answering, so
        # pass the status through. Reporting it as "cannot reach the ESP32"
        # would be a lie.
        raw = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except ValueError:
            # The board sent something that is not JSON. Pass it on verbatim
            # under its own key instead of nesting it inside "error", which
            # would read as if the bridge had produced it.
            parsed = {"error": f"the ESP32 replied {exc.code} with a body that "
                               f"was not JSON", "detail": raw.strip()}
        raise Esp32Error(exc.code, parsed) from exc


# --- tools the model may call --------------------------------------------
# Only the pins listed in leds.txt are offered, so the model cannot invent
# a pin that would drive something unexpected.

def read_leds():
    """Return the LED list straight out of leds.txt, so the tools always
    match the firmware instead of drifting out of sync.

    Return its configured colour as well as its name and pin so the assistant
    can follow the user's one-colour-at-a-time lighting preference.
    """
    cfg = HERE / "leds.txt"
    if not cfg.exists():
        cfg = HERE.parent / "leds.txt"
    leds = []
    for raw in cfg.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9 _-]+?)\s*:\s*(?:gpio)?\s*(\d+)\s*(?:,\s*([a-z]+)\s*)?$",
                     line, re.I)
        if m:
            leds.append({"name": m.group(1).strip(), "pin": int(m.group(2)),
                         "color": (m.group(3) or "").lower()})
    return leds

def read_servos():
    """Return the servo list straight out of servos.txt, so the tools always
    match the firmware instead of drifting out of sync.

    A line is 'name: gpio N' with an optional ', open A, close B'. The pair is
    what makes a servo a plain open/close mechanism like a door or a window.
    """
    cfg = HERE / "servos.txt"
    if not cfg.exists():
        cfg = HERE.parent / "servos.txt"
    servos = []
    for raw in cfg.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        m = SERVO_RE.match(line)
        if m:
            servos.append({
                "name": m.group(1).strip(),
                "pin": int(m.group(2)),
                "open": int(m.group(3)) if m.group(3) is not None else None,
                "close": int(m.group(4)) if m.group(4) is not None else None,
            })
    return servos


def servo_travel_from_board():
    """The open/close angles the board is really using.

    The page can retune them at runtime and they live in the board's own
    memory, so servos.txt is only the starting point. Empty when the board
    cannot be reached, in which case the file is the best answer we have.
    """
    try:
        return esp32_get("/api/state", timeout=2).get("servoTravel") or {}
    except Exception:  # noqa: BLE001 - the file is a fine fallback
        return {}


# What build_tools last saw, so a tool call can describe the result in words
# without asking the board a second time.
LAST_TRAVEL = {}


def effective_pair(servo, travel=None):
    """The (open, close) angles to use for a servo: the board's if it said."""
    live = (travel if travel is not None else LAST_TRAVEL).get(str(servo["pin"]))
    if isinstance(live, dict) and "open" in live and "close" in live:
        return live["open"], live["close"]
    if servo.get("open") is not None:
        return servo["open"], servo["close"]
    return None


def servo_travel(servo, travel=None):
    """'open 95, close 0' for a two-position servo, '' for a free one."""
    pair = effective_pair(servo, travel)
    return "" if pair is None else f" (open {pair[0]}, close {pair[1]})"


def servo_state(servo, angle, travel=None):
    """Say open or closed instead of a bare angle when we know the pair."""
    pair = effective_pair(servo, travel)
    if pair is None or not isinstance(angle, (int, float)):
        return f"{angle} degrees"
    if angle == pair[0]:
        return f"{angle} degrees (open)"
    if angle == pair[1]:
        return f"{angle} degrees (closed)"
    return f"{angle} degrees (part way)"


def build_tools():
    """Offer only appliance-level actions; keep hardware values server-side."""
    leds = read_leds()
    servos = read_servos()
    global LAST_TRAVEL
    LAST_TRAVEL = servo_travel_from_board()
    openings = [s for s in servos if effective_pair(s, LAST_TRAVEL)]
    led_names = [l["name"] for l in leds]
    opening_names = [s["name"] for s in openings]

    tools = [
        {"type": "function", "function": {
            "name": "get_lights",
            "description": "Read the on/off state of the configured lights by name.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "turn_on_one_light",
            "description": (
                "Turn on exactly one requested light and turn off every other light. "
                f"Available names: {', '.join(led_names)}."
            ),
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "enum": led_names}},
                "required": ["name"]}}},
        {"type": "function", "function": {
            "name": "turn_on_multiple_lights",
            "description": (
                "Turn on the requested combination of lights and turn off every "
                "other light. Use only when the user explicitly requests multiple "
                f"lights/colors, or all lights. Available names: {', '.join(led_names)}."
            ),
            "parameters": {"type": "object", "properties": {
                "names": {"type": "array", "minItems": 1,
                    "items": {"type": "string", "enum": ["all"] + led_names}},
                "all": {"type": "boolean", "description": "Set true only when the user asks for all lights."}},
                "required": ["names"]}}},
        {"type": "function", "function": {
            "name": "turn_off_lights",
            "description": "Turn off the requested lights, or all lights when requested.",
            "parameters": {"type": "object", "properties": {
                "names": {"type": "array", "minItems": 1,
                    "items": {"type": "string", "enum": ["all"] + led_names}},
                }, "required": ["names"]}}},
    ]

    if openings:
        tools.extend([
            {"type": "function", "function": {
                "name": "get_openings",
                "description": "Read whether the configured doors and windows are open or closed.",
                "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {
                "name": "set_opening",
                "description": (
                    "Open or close a configured door, window, or other opening. "
                    "Choose its name and the state open or closed; the server "
                    f"handles the mechanism. Available names: {', '.join(opening_names)}."
                ),
                "parameters": {"type": "object", "properties": {
                    "name": {"type": "string", "enum": opening_names},
                    "state": {"type": "string", "enum": ["open", "closed"]}},
                    "required": ["name", "state"]}}},
        ])

    if any(s["pin"] == 32 and s["name"].lower() == "cradle" for s in servos):
        tools.append({"type": "function", "function": {
            "name": "soothe_baby",
            "description": (
                "Run the configured seven-cycle cradle movement when the user asks "
                "to calm, soothe, rock, move, or put the baby to sleep. It stops by itself."
            ),
            "parameters": {"type": "object", "properties": {}}}})

    tools.extend([
        {"type": "function", "function": {
            "name": "get_fan",
            "description": "Read the fan's current direction and speed percentage.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "set_fan",
            "description": (
                "Control the fan direction and speed. An unspecified turn-on "
                "direction defaults to forward. Speed is optional and ranges "
                "from 50 to 100 percent; omitted speed uses the server default."
            ),
            "parameters": {"type": "object", "properties": {
                "direction": {"type": "string", "enum": ["forward", "reverse", "stop"]},
                "speed": {"type": "integer", "minimum": 50, "maximum": 100}},
                "required": ["direction"]}}},
    ])
    return tools


def run_tool(name, args):
    """Run one high-level appliance action; hardware values stay private."""
    try:
        if name == "get_lights":
            state = esp32_get("/api/state")
            lights = []
            for led in read_leds():
                on = bool(state.get("leds", {}).get(str(led["pin"]), False))
                lights.append(f"{led['name']}: {'on' if on else 'off'}")
            return "Light states: " + ("; ".join(lights) if lights else "none configured") + "."

        if name in ("turn_on_one_light", "turn_on_multiple_lights", "turn_off_lights"):
            if name == "turn_on_one_light":
                names = [args.get("name", "")]
                state_arg = "on"
            else:
                names = args.get("names", [])
                state_arg = "off" if name == "turn_off_lights" else "on"
            if isinstance(names, str):
                names = [names]
            if not isinstance(names, list) or not names:
                return "Error: choose one or more configured light names."
            if any(str(n).strip().lower() == "all" for n in names):
                if len(names) != 1:
                    return "Error: use all by itself."
                esp32_post("/api/all", state_arg)
                return f"Success: all lights are now {state_arg}."
            available = read_leds()
            by_name = {led["name"].lower(): led for led in available}
            chosen = []
            for raw_name in names:
                led = by_name.get(str(raw_name).strip().lower())
                if led is None:
                    return "Error: unavailable light name. Available lights: " + ", ".join(
                        led["name"] for led in available) + "."
                if led not in chosen:
                    chosen.append(led)
            if state_arg == "on":
                chosen_pins = {led["pin"] for led in chosen}
                for led in available:
                    if led["pin"] not in chosen_pins:
                        esp32_post(f"/api/led/{led['pin']}", "off")
            for led in chosen:
                esp32_post(f"/api/led/{led['pin']}", state_arg)
            return f"Success: {', '.join(led['name'] for led in chosen)} {state_arg}."

        if name == "get_openings":
            state = esp32_get("/api/state")
            live = state.get("servoTravel") or {}
            positions = state.get("servos", {})
            descriptions = []
            for servo in read_servos():
                pair = effective_pair(servo, live)
                if not pair:
                    continue
                position = positions.get(str(servo["pin"]))
                status = "open" if position == pair[0] else (
                    "closed" if position == pair[1] else "in between")
                descriptions.append(f"{servo['name']}: {status}")
            return "Opening states: " + ("; ".join(descriptions) if descriptions else "none") + "."

        if name == "set_opening":
            wanted = str(args.get("name", "")).strip().lower()
            state_arg = str(args.get("state", "")).strip().lower()
            if state_arg not in ("open", "closed"):
                return "Error: state must be open or closed."
            match = next((s for s in read_servos()
                          if s["name"].lower() == wanted and effective_pair(s)), None)
            if not match:
                names = ", ".join(s["name"] for s in read_servos() if effective_pair(s)) or "none"
                return f"Error: unknown opening. Available openings: {names}."
            pair = effective_pair(match)
            angle = int(pair[0] if state_arg == "open" else pair[1])
            esp32_post(f"/api/servo/{match['pin']}", str(angle))
            return f"Success: {match['name']} is {state_arg}."

        if name == "soothe_baby":
            if not any(s["pin"] == 32 and s["name"].lower() == "cradle"
                       for s in read_servos()):
                return "Error: the cradle is not configured."
            esp32_post("/api/servo/32/wave", "start 17 310 7 90")
            return "Success: cradle soothing sequence started and will stop after seven cycles."

        if name == "get_fan":
            state = esp32_get("/api/state")
            fan = state.get("motors", {}).get("1", {})
            direction = fan.get("direction", "unknown")
            if direction in ("forward", "reverse"):
                direction = "reverse" if direction == "forward" else "forward"
            return f"Fan is {direction} at {fan.get('speed', 'unknown')} percent."

        if name == "set_fan":
            direction = str(args.get("direction", "")).strip().lower()
            if direction not in ("forward", "reverse", "stop"):
                return "Error: direction must be forward, reverse, or stop."
            # The fan motor wiring is reversed, so translate the logical
            # direction here and keep the model's interface intuitive.
            board_direction = {"forward": "reverse", "reverse": "forward", "stop": "stop"}[direction]
            if direction == "stop":
                esp32_post("/api/motor/1", "stop")
            else:
                speed = args.get("speed", 70)
                if not isinstance(speed, int) or isinstance(speed, bool) or not 50 <= speed <= 100:
                    return "Error: fan speed must be a whole number from 50 to 100."
                esp32_post("/api/motor/1", f"{board_direction} 100")
                time.sleep(2)
                esp32_post("/api/motor/1", f"{board_direction} {speed}")
            try:
                fan = esp32_get("/api/state").get("motors", {}).get("1", {})
                actual = fan.get("direction")
                if actual in ("forward", "reverse"):
                    actual = "reverse" if actual == "forward" else "forward"
                return f"Success: fan is {actual} at {fan.get('speed')} percent."
            except Exception:
                return "Success: fan command sent."

        return f"Error: unknown action '{name}'."
    except urllib.error.URLError as exc:
        return f"Error: cannot reach the ESP32 ({exc.reason})."
    except Exception as exc:  # noqa: BLE001
        return f"Error while controlling the device: {exc}"


# --- the OpenRouter call --------------------------------------------------

def call_model(messages, tools, model, timeout=45):
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.4,
    }
    # Free models here think out loud, which costs a second or more per turn
    # and buys nothing for "turn the red light on". Paid fast models ignore
    # this, and a few free ones reject it outright, so only ask when the model
    # is one of ours.
    if model.endswith(":free"):
        payload["reasoning"] = {"effort": "none"}
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Smart House Prototype",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode())


class ModelUnavailable(Exception):
    """This model is busy or withdrawn. Try the next one."""


def call_any_model(messages, tools):
    """Try each free model in turn and return (completion, model_used).

    Free OpenRouter models share one upstream pool, so a 429 rate limit is
    the normal case rather than an exception. Walking the list keeps the
    prototype usable whenever at least one free model has capacity.
    """
    last = None
    for model in MODELS:
        try:
            return call_model(messages, tools, model=model), model
        except urllib.error.HTTPError as exc:
            exc.read()
            # 404 withdrawn, 429 rate limited, 5xx overloaded: try the next.
            if exc.code in (404, 429, 500, 502, 503):
                last = f"{model} -> HTTP {exc.code}"
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last = f"{model} -> {exc}"
            continue
    raise ModelUnavailable(last or "no free model available")


@app.get("/")
def page():
    """Serve the control page from localhost so the microphone is allowed."""
    if not PAGE_DIR.is_dir():
        return jsonify({
            "error": f"page folder not found at {PAGE_DIR}",
        }), 500
    return send_from_directory(PAGE_DIR, "index.html")


@app.get("/api/state")
def proxy_state():
    if not ESP32_URL:
        return jsonify({"error": "no ESP32_URL configured on the bridge"}), 500
    try:
        return jsonify(esp32_get("/api/state"))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"cannot reach the ESP32: {exc}"}), 502


@app.post("/api/all")
def proxy_all():
    if not ESP32_URL:
        return jsonify({"error": "no ESP32_URL configured on the bridge"}), 500
    try:
        return jsonify(esp32_post("/api/all", request.get_data(as_text=True).strip()))
    except Esp32Error as exc:
        return jsonify(exc.body), exc.status
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"cannot reach the ESP32: {exc}"}), 502


@app.post("/api/led/<int:pin>")
def proxy_led(pin):
    if not ESP32_URL:
        return jsonify({"error": "no ESP32_URL configured on the bridge"}), 500
    try:
        return jsonify(esp32_post(f"/api/led/{pin}", request.get_data(as_text=True).strip()))
    except Esp32Error as exc:
        return jsonify(exc.body), exc.status
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"cannot reach the ESP32: {exc}"}), 502


@app.post("/api/servo/<int:pin>")
def proxy_servo(pin):
    if not ESP32_URL:
        return jsonify({"error": "no ESP32_URL configured on the bridge"}), 500
    try:
        return jsonify(esp32_post(f"/api/servo/{pin}", request.get_data(as_text=True).strip()))
    except Esp32Error as exc:
        return jsonify(exc.body), exc.status
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"cannot reach the ESP32: {exc}"}), 502


@app.post("/api/servo/<int:pin>/range")
def proxy_servo_range(pin):
    """Retune a two-position servo, e.g. body '95 0' for open then close."""
    if not ESP32_URL:
        return jsonify({"error": "no ESP32_URL configured on the bridge"}), 500
    try:
        return jsonify(esp32_post(f"/api/servo/{pin}/range", request.get_data(as_text=True).strip()))
    except Esp32Error as exc:
        return jsonify(exc.body), exc.status
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"cannot reach the ESP32: {exc}"}), 502


@app.post("/api/servo/<int:pin>/wave")
def proxy_servo_wave(pin):
    if not ESP32_URL:
        return jsonify({"error": "no ESP32_URL configured on the bridge"}), 500
    try:
        return jsonify(esp32_post(f"/api/servo/{pin}/wave", request.get_data(as_text=True).strip()))
    except Esp32Error as exc:
        return jsonify(exc.body), exc.status
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"cannot reach the ESP32: {exc}"}), 502


@app.post("/api/motor/<int:mid>")
def proxy_motor(mid):
    if not ESP32_URL:
        return jsonify({"error": "no ESP32_URL configured on the bridge"}), 500
    try:
        return jsonify(esp32_post(f"/api/motor/{mid}", request.get_data(as_text=True).strip()))
    except Esp32Error as exc:
        return jsonify(exc.body), exc.status
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"cannot reach the ESP32: {exc}"}), 502


@app.post("/api/chat")
def chat():
    if not API_KEY:
        return jsonify({"error": "no API key configured on the bridge"}), 500
    if not ESP32_URL:
        return jsonify({"error": "no ESP32_URL configured on the bridge"}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not text:
        return jsonify({"error": "empty message"}), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # The history sent here is plain text and carries no tool calls. A weak
    # model learns from that to answer in words instead of acting, which is
    # what made commands silently do nothing. Measured on this project with
    # gemini-2.5-flash-lite: 8 messages -> 0 calls, 0 messages -> 9/9.
    # openai/gpt-4.1-mini does not have that weakness (12/12 at 8 messages),
    # so the history can stay long enough to be useful. If the model is ever
    # changed to a smaller one, lower this to 2 and expect a faster but less
    # reliable assistant.
    for turn in history[-MAX_HISTORY_MESSAGES:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content.strip()[:500]})
    messages.append({"role": "user", "content": text[:500]})

    tools = build_tools()
    actions = []
    used_model = None

    # Up to 4 rounds so the model can act, read the result, and reply.
    for _ in range(4):
        try:
            completion, used_model = call_any_model(messages, tools)
        except ModelUnavailable as exc:
            return jsonify({
                "error": "every free model is busy right now, try again in a moment",
                "detail": str(exc),
            }), 503
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            return jsonify({"error": f"OpenRouter {exc.code}: {detail}"}), 502
        except urllib.error.URLError as exc:
            return jsonify({"error": f"cannot reach OpenRouter: {exc.reason}"}), 502

        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            reply = (message.get("content") or "").strip()
            return jsonify({
                "reply": reply or "Done.",
                "actions": actions,
                "model": used_model,
            })

        messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            fn = call.get("function") or {}
            fn_name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(fn_name, args)
            actions.append({"tool": fn_name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": result,
            })

    return jsonify({
        "reply": "I did that, but let me stop there.",
        "actions": actions,
        "model": used_model,
    })


@app.get("/api/health")
def health():
    ok, detail = True, None
    if ESP32_URL:
        try:
            esp32_get("/api/state", timeout=3)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, str(exc)
    return jsonify({
        "bridge": "up",
        "model": MODEL,
        "fallback_models": MODELS[1:],
        "key_configured": bool(API_KEY),
        "esp32_url": ESP32_URL or None,
        "esp32_reachable": ok,
        "esp32_error": detail,
        "leds": read_leds(),
    })


if __name__ == "__main__":
    if not API_KEY:
        print("warning: no OPENROUTER_API_KEY found in .env\n", file=sys.stderr)
    print(f"model:    {MODEL}")
    print(f"esp32:    {ESP32_URL or '(not set)'}")
    print(f"key:      {'loaded' if API_KEY else 'MISSING'}")
    print(f"\nbridge running on http://127.0.0.1:{BRIDGE_PORT}\n")
    app.run(host="127.0.0.1", port=BRIDGE_PORT, debug=False)
