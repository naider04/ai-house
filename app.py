#!/usr/bin/env python3
"""Render app: public AI chat, approved accounts, owner console, home connector."""
import hmac
import json
import os
import queue
import secrets
import sqlite3
import threading
import time
from functools import wraps
from pathlib import Path

from flask import Flask, g, jsonify, request, send_file, session
from werkzeug.security import check_password_hash, generate_password_hash

import assistant_core as core

BASE = Path(__file__).resolve().parent
DB_PATH = os.environ.get("DATABASE_PATH", str(BASE / "data.sqlite3"))
APP_SECRET = os.environ.get("SECRET_KEY", "")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "")
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "").strip()
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
app = Flask(__name__, static_folder=None)
app.secret_key = APP_SECRET or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=os.environ.get("RENDER", "") == "true",
                  MAX_CONTENT_LENGTH=16_384)
pending_jobs = queue.Queue()
job_results = {}
job_lock = threading.Lock()


def connect_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=20)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    with connect_db() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user',
          status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        if ADMIN_USERNAME and ADMIN_PASSWORD:
            row = db.execute("SELECT id FROM users WHERE username=?", (ADMIN_USERNAME,)).fetchone()
            if not row:
                db.execute("INSERT INTO users(username,password_hash,role,status) VALUES(?,?,?,?)",
                           (ADMIN_USERNAME, generate_password_hash(ADMIN_PASSWORD), "admin", "active"))
            else:
                # Render environment settings are the recovery path for the
                # owner account. Updating ADMIN_PASSWORD and restarting the
                # service replaces the stored password hash.
                db.execute("UPDATE users SET password_hash=?,role='admin',status='active' WHERE username=?",
                           (generate_password_hash(ADMIN_PASSWORD), ADMIN_USERNAME))


init_db()


def current_user():
    if "user" not in g:
        uid = session.get("uid")
        g.user = None
        if uid:
            with connect_db() as db:
                row = db.execute("SELECT id,username,role,status FROM users WHERE id=?", (uid,)).fetchone()
            if row and row["status"] == "active":
                g.user = dict(row)
                g.user.pop("status", None)
    return g.user


def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            return jsonify({"error": "Sign in with an approved account to control the house."}), 401
        return fn(*args, **kwargs)
    return wrapped


def admin_required(fn):
    @wraps(fn)
    @login_required
    def wrapped(*args, **kwargs):
        if current_user()["role"] != "admin":
            return jsonify({"error": "Owner access required."}), 403
        return fn(*args, **kwargs)
    return wrapped


def agent_authorized():
    header = request.headers.get("Authorization", "")
    token = header[7:] if header.startswith("Bearer ") else ""
    return bool(AGENT_TOKEN) and hmac.compare_digest(token, AGENT_TOKEN)


def agent_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not agent_authorized():
            return jsonify({"error": "Invalid connector credentials."}), 401
        return fn(*args, **kwargs)
    return wrapped


@app.get("/")
def public_page():
    return send_file(BASE / "public.html")


@app.get("/owner")
def owner_page():
    return send_file(BASE / "owner.html")


@app.get("/controls")
@login_required
def controls_page():
    if current_user()["role"] != "admin":
        return "Owner access required", 403
    return send_file(BASE / "controls.html")


@app.post("/auth/register")
def register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    if len(username) < 3 or len(username) > 40 or not all(c.isalnum() or c in "._-" for c in username):
        return jsonify({"error": "Choose a 3–40 character username using letters, numbers, dot, dash, or underscore."}), 400
    if len(password) < 12:
        return jsonify({"error": "Use a password with at least 12 characters."}), 400
    try:
        with connect_db() as db:
            db.execute("INSERT INTO users(username,password_hash,role,status) VALUES(?,?,?,?)",
                       (username, generate_password_hash(password), "user", "pending"))
    except sqlite3.IntegrityError:
        return jsonify({"error": "That username is already registered."}), 409
    return jsonify({"message": "Account request received. The owner must approve it before you can use house controls."}), 201


@app.post("/auth/login")
def login():
    data = request.get_json(silent=True) or {}
    with connect_db() as db:
        user = db.execute("SELECT * FROM users WHERE username=?", (str(data.get("username", "")).strip(),)).fetchone()
    if not user or not check_password_hash(user["password_hash"], str(data.get("password", ""))):
        return jsonify({"error": "Username or password is incorrect."}), 401
    if user["status"] != "active":
        return jsonify({"error": "This account is pending approval or has been blocked."}), 403
    session.clear()
    session["uid"] = user["id"]
    return jsonify({"username": user["username"], "role": user["role"]})


@app.post("/auth/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/auth/me")
def me():
    return jsonify(current_user() or {})


@app.get("/admin/users")
@admin_required
def list_users():
    with connect_db() as db:
        rows = db.execute("SELECT id,username,role,status,created_at FROM users ORDER BY created_at DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/admin/users/<int:user_id>/status")
@admin_required
def set_user_status(user_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status")
    if status not in ("active", "blocked", "pending"):
        return jsonify({"error": "status must be active, blocked, or pending"}), 400
    if user_id == current_user()["id"] and status != "active":
        return jsonify({"error": "You cannot block your own owner account."}), 400
    with connect_db() as db:
        cur = db.execute("UPDATE users SET status=? WHERE id=? AND role!='admin'", (status, user_id))
    if cur.rowcount != 1:
        return jsonify({"error": "User not found or is an owner."}), 404
    return jsonify({"ok": True})


@app.post("/admin/wifi")
@admin_required
def configure_wifi():
    data = request.get_json(silent=True) or {}
    ssid, password = str(data.get("ssid", "")), str(data.get("password", ""))
    if not ssid or len(ssid) > 32 or len(password) > 63:
        return jsonify({"error": "SSID is required (up to 32 characters); password must be 63 characters or fewer."}), 400
    try:
        # Line-based protocol keeps credentials out of persistent web storage.
        core.esp32_post("/api/wifi/config", ssid + "\n" + password)
        return jsonify({"ok": True})
    except core.Esp32Error as exc:
        return jsonify(exc.body), exc.status
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.get("/admin/wifi/networks")
@admin_required
def wifi_networks():
    try:
        return jsonify(core.esp32_get("/api/wifi/networks"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


def relay(method, path, body=""):
    """Queue an ESP32 request for the authenticated home connector."""
    job_id = secrets.token_urlsafe(18)
    waiter = threading.Event()
    with job_lock:
        job_results[job_id] = {"event": waiter, "result": None}
    pending_jobs.put({"id": job_id, "method": method, "path": path, "body": body})
    if not waiter.wait(timeout=28):
        with job_lock:
            job_results.pop(job_id, None)
        raise TimeoutError("Home connector is offline. Start the connector on your home computer and try again.")
    with job_lock:
        result = job_results.pop(job_id, {}).get("result") or {}
    if result.get("status", 200) >= 400:
        raise core.Esp32Error(result["status"], result.get("body", {}))
    return result.get("body", {})


core.esp32_get = lambda path, timeout=5: relay("GET", path)
core.esp32_post = lambda path, body, timeout=5: relay("POST", path, body)


@app.get("/connector/next")
@agent_required
def connector_next():
    try:
        job = pending_jobs.get(timeout=0.7)
    except queue.Empty:
        return jsonify({"job": None})
    with job_lock:
        if job["id"] not in job_results:
            return jsonify({"job": None})
    return jsonify({"job": job})


@app.post("/connector/result/<job_id>")
@agent_required
def connector_result(job_id):
    data = request.get_json(silent=True) or {}
    with job_lock:
        job = job_results.get(job_id)
        if not job:
            return jsonify({"error": "Unknown command."}), 404
        job["result"] = data
        job["event"].set()
    return jsonify({"ok": True})


@app.get("/api/state")
@login_required
def state():
    try:
        return jsonify(core.esp32_get("/api/state"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/<path:subpath>")
@login_required
def device_api(subpath):
    allowed = ("all", "led/", "servo/", "motor/")
    if not (subpath == "all" or subpath.startswith(allowed[1:])):
        return jsonify({"error": "Unknown device endpoint."}), 404
    try:
        result = core.esp32_post("/api/" + subpath, request.get_data(as_text=True).strip())
        return jsonify(result)
    except core.Esp32Error as exc:
        return jsonify(exc.body), exc.status
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


def run_chat(include_details):
    if not core.API_KEY:
        return jsonify({"error": "The owner has not configured the AI key."}), 503
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()[:500]
    if not message:
        return jsonify({"error": "Write or say a message first."}), 400
    messages = [{"role": "system", "content": core.SYSTEM_PROMPT}]
    for turn in (data.get("history") or [])[-core.MAX_HISTORY_MESSAGES:]:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant"):
            content = turn.get("content")
            if isinstance(content, str) and content.strip():
                messages.append({"role": turn["role"], "content": content.strip()[:500]})
    messages.append({"role": "user", "content": message})
    actions, used_model = [], None
    for _ in range(4):
        try:
            completion, used_model = core.call_any_model(messages, core.build_tools())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 502
        model_msg = (completion.get("choices") or [{}])[0].get("message") or {}
        calls = model_msg.get("tool_calls") or []
        if not calls:
            result = {"reply": (model_msg.get("content") or "Done.").strip(),
                      "tool_call_count": len(actions)}
            if include_details:
                result.update({"actions": actions, "model": used_model})
            return jsonify(result)
        messages.append({"role": "assistant", "content": model_msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = core.run_tool(fn.get("name", ""), args)
            actions.append({"tool": fn.get("name", ""), "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})
    result = {"reply": "I completed the requested device actions.",
              "tool_call_count": len(actions)}
    if include_details:
        result.update({"actions": actions, "model": used_model})
    return jsonify(result)


@app.post("/api/ask")
@login_required
def ask():
    return run_chat(False)


@app.post("/api/chat")
@admin_required
def chat():
    return run_chat(True)


@app.get("/health")
def health():
    return jsonify({"ok": True, "ai_configured": bool(core.API_KEY), "connector_configured": bool(AGENT_TOKEN)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
