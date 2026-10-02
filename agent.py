#!/usr/bin/env python3
"""Run at home; securely polls Render and forwards approved jobs to the ESP32."""
import json
import os
import sys
import time
import urllib.error
import urllib.request


PUBLIC_APP_URL = os.environ.get("PUBLIC_APP_URL", "").rstrip("/")
AGENT_TOKEN = os.environ.get("AGENT_TOKEN", "")
ESP32_URL = os.environ.get("ESP32_URL", "").rstrip("/")


def call(url, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Authorization": "Bearer " + AGENT_TOKEN}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=35) as response:
        raw = response.read().decode()
        return json.loads(raw) if raw else {}


def esp32(job):
    url = ESP32_URL + job["path"]
    req = urllib.request.Request(url, method=job["method"])
    if job["method"] == "POST":
        req.data = job.get("body", "").encode()
        req.add_header("Content-Type", "text/plain")
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            raw = response.read().decode()
            return {"status": response.status, "body": json.loads(raw) if raw else {}}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"error": raw[:500]}
        return {"status": exc.code, "body": parsed}
    except Exception as exc:
        return {"status": 502, "body": {"error": "Cannot reach ESP32: " + str(exc)}}


def main():
    if not (PUBLIC_APP_URL and AGENT_TOKEN and ESP32_URL):
        sys.exit("Set PUBLIC_APP_URL, AGENT_TOKEN, and ESP32_URL before starting the home connector.")
    print("Smart House connector running; polling " + PUBLIC_APP_URL)
    while True:
        try:
            data = call(PUBLIC_APP_URL + "/connector/next")
            job = data.get("job")
            if not job:
                time.sleep(0.5)
                continue
            result = esp32(job)
            call(PUBLIC_APP_URL + "/connector/result/" + job["id"], "POST", result)
        except KeyboardInterrupt:
            print("\nConnector stopped.")
            return
        except Exception as exc:
            print("Connector error: " + str(exc), file=sys.stderr)
            time.sleep(3)


if __name__ == "__main__":
    main()
