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

You control the configured LEDs, servos, and motors on an ESP32 microcontroller.
Use the available tools whenever the user asks to operate one of those devices.
The configured tools list the devices and supported actions. For the fan, an
unspecified "turn on" direction means run it in reverse; an
explicit direction from the user takes precedence. "Turn off" means stop it.
When a speed is requested, allow any value from 0 to 100 percent. The firmware
briefly starts at reverse 90 percent, then applies the requested direction and
speed. When the user
asks to put the baby to sleep, calm or soothe the baby, rock or move the baby,
or makes a related request, call wave_servo with command=start, amplitude=17,
wait_ms=310, repetitions=7, and center=90. Use these exact values each time.
When the user asks to stop rocking or moving the baby, call wave_servo with
command=stop.

Lighting preference: the user usually wants only one light color on at a time.
Before turning on a named light or color, call get_leds. Turn off active LEDs
of other colors, then turn on the requested LED(s); LEDs of the same configured
color may remain on together. Keep multiple colors on only when the user
explicitly asks for multiple colors, a combination, or all lights. When only
asked to turn a light off, do not change the other lights.

How to behave:
- Be brief and natural. You are being read out loud, so one or two short
  sentences is plenty. No markdown, no bullet lists, no emoji.
- After a tool succeeds, briefly say what happened in plain words.
- You may act on several devices in one turn if the user asks for it.
- If the user asks which devices exist, use the available tool descriptions.
- If a request is unclear, ask one short clarifying question.
- Never invent pin numbers. Only use the pins listed in your tools.
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
    leds = read_leds()
    servos = read_servos()
    global LAST_TRAVEL
    LAST_TRAVEL = servo_travel_from_board()
    led_enum = ", ".join(
        f"{l['name']} ({l['color']}, GPIO{l['pin']})" if l.get("color")
        else f"{l['name']} (GPIO{l['pin']})" for l in leds)
    servo_enum = ", ".join(f"{s['name']} (GPIO{s['pin']}){servo_travel(s)}"
                           for s in servos)

    # A worked example, built from the angles in force right now, because the
    # page can change them and a stale example would teach the model a wrong
    # number.
    servo_example = ""
    for s in servos:
        pair = effective_pair(s, LAST_TRAVEL)
        if pair:
            servo_example = (f" For example: {{'{s['name']}': {pair[1]}}} to close, "
                             f"{{'{s['name']}': {pair[0]}}} to open")
            break

    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_led",
                "description": (
                    "Turn one LED on or off. "
                    f"Available LEDs: {led_enum}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "LED name, for example 'red' or 'blue 2'.",
                        },
                        "state": {
                            "type": "string",
                            "enum": ["on", "off"],
                            "description": "Whether to switch it on or off.",
                        },
                    },
                    "required": ["name", "state"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_all_leds",
                "description": "Turn every LED on, or every LED off.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "state": {
                            "type": "string",
                            "enum": ["on", "off"],
                            "description": "Whether to switch them all on or all off.",
                        },
                    },
                    "required": ["state"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_leds",
                "description": "Read which LEDs are currently on or off.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    # Add servo tools if servos exist
    if servos:
        tools.append({
            "type": "function",
            "function": {
                "name": "set_servo",
                "description": (
                    "Set one servo to an angle in degrees. "
                    f"Available servos: {servo_enum}. "
                    "A servo listed with an open and a close angle only has those "
                    "two positions, so use those exact angles."
                    f"{servo_example}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Servo name, for example 'puerta de garage' or 'window'.",
                        },
                        "angle": {
                            "type": "number",
                            "description": "Angle in degrees, from 0 to 180.",
                        },
                    },
                    "required": ["name", "angle"],
                },
            },
        })
        tools.append({
            "type": "function",
            "function": {
                "name": "get_servos",
                "description": "Read which servos are currently positioned where.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        })

        if any(servo["pin"] == 32 for servo in servos):
            tools.append({
                "type": "function",
                "function": {
                    "name": "wave_servo",
                    "description": (
                        "Start or stop the cradle servo on GPIO32. For baby soothing/rocking requests, "
                        "starting always uses the fixed pattern: amplitude 17 degrees, wait 310 ms, "
                        "7 repetitions, center 90 degrees."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "enum": ["start", "stop"]},
                        },
                        "required": ["command"],
                    },
                },
            })


        tools.append({
            "type": "function",
            "function": {
                "name": "get_motors",
                "description": "Read the current state of motors/fan (direction and speed).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        })
        tools.append({
            "type": "function",
            "function": {
                "name": "set_motor",
                "description": (
                    "Control the fan. An unspecified turn-on direction means reverse; use stop to turn it off. "
                    "When starting, the firmware gives a brief reverse 90% kick then applies this target speed. "
                    "Available motors: fan (GPIO21 ENB, GPIO22 IN3, GPIO23 IN4). "
                    "Speed range is 0-100%%. Default speed when starting without a number is 70%%. "
                    "Valid commands include 'forward', 'reverse', 'stop', optionally followed by a speed "
                    "(e.g. 'forward 50', 'reverse 90', 'stop')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Motor name, e.g. 'fan'.",
                        },
                        "command": {
                            "type": "string",
                            "description": (
                                "Command string like 'forward', 'forward 50', 'reverse 90', 'stop'."
                            ),
                        },
                    },
                    "required": ["name", "command"],
                },
            },
        })

    return tools


def run_tool(name, args):
    """Execute one tool call against the ESP32. Returns a string for the model."""
    try:
        if name == "get_leds":
            state = esp32_get("/api/state")
            on = [str(pin) for pin, v in state.get("leds", {}).items() if v]
            off = [str(pin) for pin, v in state.get("leds", {}).items() if not v]
            return f"LEDs on (GPIO): {', '.join(on) if on else 'none'}. " \
                   f"LEDs off (GPIO): {', '.join(off) if off else 'none'}."

        if name == "set_all_leds":
            state_arg = str(args.get("state", "")).lower()
            if state_arg not in ("on", "off"):
                return "Error: state must be 'on' or 'off'."
            esp32_post("/api/all", state_arg)
            return f"Success: all LEDs are now {state_arg}."

        if name == "set_led":
            wanted = str(args.get("name", "")).strip().lower()
            state_arg = str(args.get("state", "")).lower()
            if state_arg not in ("on", "off"):
                return "Error: state must be 'on' or 'off'."

            match = None
            for led in read_leds():
                if led["name"].lower() == wanted:
                    match = led
                    break
            if match is None:
                names = ", ".join(l["name"] for l in read_leds())
                return f"Error: no LED named '{wanted}'. Available: {names}."

            esp32_post(f"/api/led/{match['pin']}", state_arg)
            return f"Success: {match['name']} (GPIO{match['pin']}) is now {state_arg}."

        if name == "get_servos":
            state = esp32_get("/api/state")
            servos_state = state.get("servos", {})
            # The reply already carries the live angles, so trust those over
            # whatever servos.txt says.
            live = state.get("servoTravel") or {}
            servo_descs = []
            for servo in read_servos():
                pin = servo['pin']
                angle = servos_state.get(str(pin), "unknown")
                servo_descs.append(f"{servo['name']} (GPIO{pin}): "
                                   f"{servo_state(servo, angle, live)}")
            return f"Servos: {', '.join(servo_descs) if servo_descs else 'none'}"

        if name == "wave_servo":
            command = str(args.get("command", "start")).strip().lower()
            if command not in ("start", "stop"):
                return "Error: command must be 'start' or 'stop'."
            if not any(servo["pin"] == 32 for servo in read_servos()):
                return "Error: no cradle servo is configured on GPIO32."
            if command == "stop":
                esp32_post("/api/servo/32/wave", "stop")
                return "Success: cradle stopped."
            amplitude, wait_ms, repetitions, center = 17, 310, 7, 90
            body = f"start {amplitude} {wait_ms} {repetitions} {center}"
            esp32_post("/api/servo/32/wave", body)
            return (f"Success: cradle started with {amplitude} degrees each side of {center}, "
                    f"changing sides every {wait_ms} ms for "
                    f"{'continuously' if repetitions == 0 else str(repetitions) + ' cycles'}.")

        if name == "set_servo":
            wanted = str(args.get("name", "")).strip().lower()
            angle_arg = args.get("angle")
            if not isinstance(angle_arg, (int, float)) or not (0 <= angle_arg <= 180):
                return "Error: angle must be a number between 0 and 180."
            
            match = None
            for servo in read_servos():
                if servo["name"].lower() == wanted:
                    match = servo
                    break
            if match is None:
                names = ", ".join(s["name"] for s in read_servos())
                return f"Error: no servo named '{wanted}'. Available: {names}."
            
            esp32_post(f"/api/servo/{match['pin']}", str(int(angle_arg)))
            return (f"Success: {match['name']} (GPIO{match['pin']}) is now at "
                    f"{servo_state(match, int(angle_arg))}.")


        if name == "get_motors":
            state = esp32_get("/api/state")
            motors = state.get("motors", {})
            descs = []
            for mid, m in motors.items():
                descs.append(f"motor {mid}: {m.get('direction')} {m.get('speed')}%")
            return f"Motors: {', '.join(descs) if descs else 'none'}."

        if name == "set_motor":
            wanted = str(args.get("name", "")).strip().lower()
            cmd = str(args.get("command", "")).strip()
            if not cmd:
                return "Error: command must be provided (e.g. 'forward 90' or 'stop')."
            simple_command = re.sub(r"\s+", " ", cmd.lower())
            if simple_command in ("on", "turn on", "start", "switch on"):
                cmd = "reverse"
            elif simple_command in ("off", "turn off", "stop", "switch off"):
                cmd = "stop"
            # find motor match by name
            match_id = None
            # only one motor 'fan' maps to id 1
            if wanted in ("fan", "motor 1", "motor1"):
                match_id = 1
            if match_id is None:
                return f"Error: no motor named '{wanted}'. Available: fan."
            esp32_post(f"/api/motor/{match_id}", cmd)
            # return updated state snippet
            try:
                st = esp32_get("/api/state")
                m = st.get("motors", {}).get(str(match_id), {})
                return f"Success: fan set to {m.get('direction')} at {m.get('speed')}%."
            except Exception:
                return "Success."

        return f"Error: unknown tool '{name}'."
    except urllib.error.URLError as exc:
        return f"Error: cannot reach the ESP32 at {ESP32_URL} ({exc.reason})."
    except Exception as exc:  # noqa: BLE001 - report anything back to the model
        return f"Error while running '{name}': {exc}"


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
