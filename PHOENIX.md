# Phoenix tracing (notarealemail fork)

This fork adds optional [Arize Phoenix](https://github.com/Arize-ai/phoenix) tracing
for `OllamaGenerateV2` and `OllamaChat`. Tracing is **fail-open**: if Phoenix is
down or packages are missing, ComfyUI generation continues.

Upstream: [stavsap/comfyui-ollama](https://github.com/stavsap/comfyui-ollama)

## Install (git — not ComfyUI Manager)

Install as a git clone so Manager updates do not overwrite customizations:

```bash
cd /mnt/f/ComfyUI/custom_nodes
mv comfyui-ollama comfyui-ollama.manager-backup   # if migrating from Manager
git clone https://github.com/notarealemail/comfyui-ollama.git
cd comfyui-ollama
git remote add upstream https://github.com/stavsap/comfyui-ollama.git

/mnt/f/ComfyUI/scripts/venv/bin/pip install -r requirements.txt
```

Or use the homelab helper: `scripts/workstation/install-comfyui-ollama.sh` in
[notarealemail/homelab](https://github.com/notarealemail/homelab).

**Do not** re-enable ComfyUI Manager auto-update for this node.

## Environment variables

Set before starting ComfyUI (system env, `.env` beside ComfyUI, or launcher script):

| Variable | Example | Purpose |
|----------|---------|---------|
| `PHOENIX_ENABLED` | `1` | Master switch (default on if packages installed) |
| `PHOENIX_PROJECT_NAME` | `comfyui-ollama` | Phoenix project |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://192.168.6.124:4317/v1/traces` | Homelab Phoenix OTLP HTTP |
| `PHOENIX_API_KEY` | *(System API key)* | Required when Phoenix auth is enabled |
| `PHOENIX_CAPTURE_CONTENT` | `1` | Full prompt/response on private LAN collectors |
| `PHOENIX_BATCH` | `0` | `0` = export each span immediately (easier debugging) |
| `PHOENIX_CONTENT_LIMIT` | `20000` | Max chars per text attribute |

Homelab Phoenix UI: `https://phoenix.infra.realemail.app`

## Verify

After one `OllamaGenerateV2` run, open Phoenix → project `comfyui-ollama`:

- Span kind **LLM** (not Unknown)
- Non-zero token counts
- Readable input/output messages (not a Python object repr)

## Merge upstream releases

```bash
cd /mnt/f/ComfyUI/custom_nodes/comfyui-ollama
git fetch upstream
git merge upstream/main
# Resolve conflicts in CompfyuiOllama.py if upstream touched the same regions
git push origin main
```

Phoenix helpers are marked in `CompfyuiOllama.py`:

```text
# --- Phoenix tracing helpers (Codex) ---
...
# --- End Phoenix tracing helpers (Codex) ---
```
