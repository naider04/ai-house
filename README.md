# AI House (Render)

This is the public AI chat and its private owner console. The public page only
shows the chat; it does not show device sliders, tool-call logs, or owner tools.
New accounts wait for owner approval. The owner can approve or block accounts
at `/owner`.

## Deploy on Render

Create a **Blueprint** from this repository. Render reads `render.yaml`. During
setup, enter these environment values when prompted:

- `OPENROUTER_API_KEY`: the server-side OpenRouter key
- `ADMIN_USERNAME`: your owner login
- `ADMIN_PASSWORD`: a unique password with at least 12 characters

Render creates `SECRET_KEY` and `AGENT_TOKEN`. The SQLite database is kept on
the service's persistent disk, so the Blueprint uses a paid Starter web
service. Do not remove the disk unless you move the database to another
persistent store.

Never add a Wi-Fi password, OpenRouter key, or Render secret to this repository.

## Connect the house from home

The Render service cannot connect directly to an ESP32 behind a home router.
Run the included connector on a computer that stays on at home. It makes an
outbound HTTPS connection to Render; there is no router port-forwarding.

1. In Render, copy the `AGENT_TOKEN` value from the service's Environment page.
2. On the home computer, install Python 3 and set the three variables below.
3. Run `python3 agent.py` from this repository directory.

```bash
export PUBLIC_APP_URL='https://YOUR-SERVICE.onrender.com'
export AGENT_TOKEN='paste-the-render-agent-token-here'
export ESP32_URL='http://192.168.1.61'
python3 agent.py
```

Change `ESP32_URL` to the board's current address. A DHCP reservation in the
router prevents that address from changing. Keep this connector running. The
Render app stores queued commands only in memory and is configured as one web
worker; the home connector token is required for its private command channel.

## ESP32 Wi-Fi

The owner console has a Wi-Fi form. It sends credentials over the paired home
connector and the ESP32 stores them in its Preferences memory, then reconnects.
It does not save the Wi-Fi password in the website database. If the ESP32
cannot join a network, current firmware starts the `SmartHouse-Setup` access
point. Connect locally and open `http://192.168.4.1/wifi` to enter valid
credentials. Wi-Fi should be 2.4 GHz.

Flash the current firmware from the project before using Wi-Fi setup:

```bash
cd ESP32
python3 generate.py --upload
```

## Owner links

- Public AI chat: `/`
- Owner access and Wi-Fi console: `/owner`
- Private LED, servo, and fan controls: `/controls`

New users choose **Request access** on the public page. Their accounts remain
pending until the owner approves them. Blocking an account prevents future
AI and device API requests for that account.
