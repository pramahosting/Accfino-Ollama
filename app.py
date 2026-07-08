import csv
import io
import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://accfino-ollama:11434")
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
EXAMPLES_FILE = DATA_DIR / "examples.json"

try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not EXAMPLES_FILE.exists():
        EXAMPLES_FILE.write_text("[]")
    STORAGE_OK = True
    STORAGE_ERROR = None
except Exception as e:
    # Don't let a volume/permission problem take the whole app down at
    # startup -- fall back to in-memory storage and surface the error
    # in /api/health instead of crashing on import.
    STORAGE_OK = False
    STORAGE_ERROR = str(e)
    _memory_examples = []

# A curated list of common Ollama library models. Ollama has no public API to
# list its full catalog, so this is maintained by hand. Any model name can
# still be typed in manually and pulled even if it's not in this list --
# check https://ollama.com/library for the current full catalog and tags.
CURATED_MODELS = [
    {"name": "llama3.2", "note": "Meta, general purpose, 3B/1B variants"},
    {"name": "llama3.1", "note": "Meta, general purpose, 8B/70B variants"},
    {"name": "mistral", "note": "Mistral AI, general purpose, 7B"},
    {"name": "qwen2.5", "note": "Alibaba, strong at structured/JSON output"},
    {"name": "qwen2.5-coder", "note": "Alibaba, code-focused"},
    {"name": "phi3", "note": "Microsoft, small and fast"},
    {"name": "gemma2", "note": "Google, general purpose"},
    {"name": "deepseek-r1", "note": "Reasoning-focused"},
    {"name": "codellama", "note": "Meta, code-focused"},
    {"name": "tinyllama", "note": "Very small, fast, lower quality"},
    {"name": "0xroyce/Plutus-3B", "note": "Used by rdr.py in this project"},
]

app = FastAPI()


def load_examples():
    if not STORAGE_OK:
        return _memory_examples
    try:
        return json.loads(EXAMPLES_FILE.read_text())
    except Exception:
        return []


def save_examples(examples):
    if not STORAGE_OK:
        global _memory_examples
        _memory_examples = examples
        return
    EXAMPLES_FILE.write_text(json.dumps(examples, indent=2))


def score(prompt_words, example_input):
    ex_words = set(example_input.lower().split())
    return len(prompt_words & ex_words)


@app.get("/api/health")
async def health():
    result = {"storage_ok": STORAGE_OK, "storage_error": STORAGE_ERROR}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(OLLAMA_BASE_URL + "/")
            result.update({"ollama_reachable": r.status_code == 200, "ollama_base_url": OLLAMA_BASE_URL})
            return result
    except Exception as e:
        result.update({"ollama_reachable": False, "error": str(e), "ollama_base_url": OLLAMA_BASE_URL})
        return JSONResponse(result)


@app.get("/api/tags")
async def tags():
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(OLLAMA_BASE_URL + "/api/tags")
        return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/api/library")
async def library():
    return CURATED_MODELS


@app.post("/api/pull")
async def pull(payload: dict):
    model = payload.get("model", "").strip()

    async def stream_pull():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", OLLAMA_BASE_URL + "/api/pull", json={"model": model, "stream": True}
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield line + "\n"

    return StreamingResponse(stream_pull(), media_type="application/x-ndjson")


@app.delete("/api/tags/{model_name:path}")
async def delete_model(model_name: str):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.request("DELETE", OLLAMA_BASE_URL + "/api/delete", json={"model": model_name})
        return JSONResponse({"ok": r.status_code == 200}, status_code=r.status_code)


@app.get("/api/examples")
async def get_examples():
    ex = load_examples()
    return {"count": len(ex), "examples": ex[-50:]}


@app.delete("/api/examples")
async def clear_examples():
    save_examples([])
    return {"ok": True}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    new_examples = []

    if file.filename.endswith(".json"):
        data = json.loads(content)
        for row in data:
            inp = row.get("input") or row.get("description") or row.get("transaction") or ""
            out = row.get("output") or row.get("category") or row.get("label") or ""
            if inp and out:
                new_examples.append({"input": inp.strip(), "output": out.strip()})
    else:
        text = content.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            keys = {k.lower(): k for k in row.keys()}
            inp_key = next((keys[k] for k in ("input", "description", "transaction") if k in keys), None)
            out_key = next((keys[k] for k in ("output", "category", "label") if k in keys), None)
            if inp_key and out_key and row.get(inp_key) and row.get(out_key):
                new_examples.append({"input": row[inp_key].strip(), "output": row[out_key].strip()})

    existing = load_examples()
    existing.extend(new_examples)
    save_examples(existing)
    return {"added": len(new_examples), "total": len(existing)}


@app.post("/api/ask")
async def ask(payload: dict):
    model = payload.get("model", "").strip()
    prompt = payload.get("prompt", "").strip()
    use_examples = payload.get("use_examples", True)

    used = []
    final_prompt = prompt

    if use_examples:
        examples = load_examples()
        prompt_words = set(prompt.lower().split())
        scored = [(score(prompt_words, e["input"]), e) for e in examples]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda x: -x[0])
        used = [e for _, e in scored[:3]]

        if used:
            context_lines = "\n".join(f'- "{e["input"]}" -> {e["output"]}' for e in used)
            final_prompt = (
                "Reference examples from prior data (for guidance only, do not repeat verbatim "
                "unless it matches):\n" + context_lines + "\n\nNow answer this:\n" + prompt
            )

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(
            OLLAMA_BASE_URL + "/api/generate",
            json={"model": model, "prompt": final_prompt, "stream": False},
        )
        data = r.json()

    return {
        "response": data.get("response", ""),
        "model": data.get("model", model),
        "used_examples": used,
        "eval_count": data.get("eval_count"),
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
