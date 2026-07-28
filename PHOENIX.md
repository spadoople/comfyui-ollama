# Phoenix tracing (spadoople fork)

This fork adds optional [Arize Phoenix](https://github.com/Arize-ai/phoenix) tracing
for `OllamaGenerateV2` and `OllamaChat`. Tracing is **fail-open**: if Phoenix is
down or packages are missing, ComfyUI generation continues.

Upstream: [stavsap/comfyui-ollama](https://github.com/stavsap/comfyui-ollama)

## Install (git — not ComfyUI Manager)

Install as a git clone so Manager updates do not overwrite customizations:

```bash
cd /mnt/f/ComfyUI/custom_nodes
mv comfyui-ollama comfyui-ollama.manager-backup   # if migrating from Manager
# Move the backup OUT of custom_nodes (ComfyUI loads every folder here):
mv comfyui-ollama.manager-backup /mnt/f/ComfyUI/custom_nodes/.disabled/
git clone https://github.com/spadoople/comfyui-ollama.git
cd comfyui-ollama
git remote add upstream https://github.com/stavsap/comfyui-ollama.git

/mnt/f/ComfyUI/scripts/venv/bin/pip install -r requirements.txt
```

Or use the homelab helper: `scripts/workstation/install-comfyui-ollama.sh` in
[spadoople/homelab](https://github.com/spadoople/homelab).

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
| `PHOENIX_SESSION_PREFIX` | `flux-lora` | Optional namespace prepended to session IDs |
| `PHOENIX_TAGS` | `lora-select,dev` | Extra comma-separated Phoenix filter tags |

Homelab Phoenix UI: `https://phoenix.infra.realemail.app`

### UmeAiRT launcher file pattern

If you start ComfyUI with `UmeAiRT-Start-ComfyUI.bat`, these one-line files under
`F:\ComfyUI\scripts\` are loaded automatically (same pattern as the API key):

| File | Sets |
|------|------|
| `phoenix_collector_endpoint` | `PHOENIX_COLLECTOR_ENDPOINT` |
| `phoenix_api_key` | `PHOENIX_API_KEY` |
| `phoenix_session_prefix` | `PHOENIX_SESSION_PREFIX` |
| `phoenix_tags` | `PHOENIX_TAGS` |

## Sessions

Phoenix **Sessions** group related traces into a conversation-style thread.

| Node | Session ID | Groups |
|------|------------|--------|
| `OllamaGenerateV2` | `comfyui-prompt:<prompt_id>` | All Ollama generate calls in one ComfyUI queue run |
| `OllamaChat` | `ollama-chat:<session_key>` | Multi-turn chat history across workflow runs |

With `PHOENIX_SESSION_PREFIX=flux-lora`, session IDs become
`flux-lora:comfyui-prompt:<prompt_id>`.

## Tags and metadata

Every traced span gets Phoenix filter tags:

- `node-type:OllamaGenerateV2` or `node-type:OllamaChat`
- `model:<ollama-model-name>`
- `workflow:<comfyui-prompt-id>` when ComfyUI execution context is available
- Any extras from `PHOENIX_TAGS` (comma-separated)

OpenInference `metadata` JSON on each span includes `comfy.prompt_id`,
`comfy.node_id`, and `llm.model_name` for structured filtering.

View in Phoenix → **Sessions** tab. Spans also include `comfy.prompt_id` and
`comfy.node_id` for filtering in the Traces view.

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
