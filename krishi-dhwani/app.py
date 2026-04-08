"""
Krishi Dhwani — Voice-enabled agricultural advisory agent
Databricks Apps entry point
"""

import os
import base64
import json
import pickle
import tempfile
import shutil
import faiss
import numpy as np
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer
from databricks.connect import DatabricksSession
from databricks_langchain import ChatDatabricks
from databricks.sdk import WorkspaceClient
from pathlib import Path

LOCAL_CACHE = "/tmp/faiss_cache"
FAISS_VOLUME_PATH = "/Volumes/krishi_dhwani/silver/faiss_index"

# ─── Config from environment (injected by app.yaml) ─────────────────────────
SARVAM_API_KEY = "sk_d1nvpbo5_0LAo4g5aR8ZwiPg8Lgq6Y2QP"
SARVAM_BASE    = "https://api.sarvam.ai"

FAISS_INDEX_PATH = "/Volumes/krishi_dhwani/silver/faiss_index/icar.index" 
METADATA_PATH    = "/Volumes/krishi_dhwani/silver/faiss_index/metadata.pkl"

LANGUAGE_MAP = {
    "English": "en-IN",   # ← add this at the top
    "Hindi":   "hi-IN",
    "Punjabi": "pa-IN",
    "Tamil":   "ta-IN",
    "Telugu":  "te-IN",
    "Marathi": "mr-IN",
    "Kannada": "kn-IN",
    "Gujarati":"gu-IN",
    "Bengali": "bn-IN",
    "Odia":    "od-IN",
}

# ─── Spark session (Databricks Connect) ─────────────────────────────────────
spark = (
    DatabricksSession.builder
    .serverless(True)
    .getOrCreate()
)

# ─── LLM ────────────────────────────────────────────────────────────────────
llm = ChatDatabricks(
    endpoint="databricks-llama-4-maverick",
    temperature=0.3,
    max_tokens=300,
)


def get_local_index():
    local_path = Path(LOCAL_CACHE)
    index_file = local_path / "icar.index"
    meta_file = local_path / "metadata.pkl"

    # 1. Check if files exist AND are not empty
    if index_file.exists() and os.path.getsize(index_file) > 0:
        if meta_file.exists() and os.path.getsize(meta_file) > 0:
            print(f"✅ Valid index found in cache ({os.path.getsize(index_file)} bytes)")
            return str(index_file), str(meta_file)

    # 2. If empty or missing, clear and download
    print(f"📦 Downloading/Refreshing index from {FAISS_VOLUME_PATH}...")
    local_path.mkdir(parents=True, exist_ok=True)
    
    w = WorkspaceClient()
    for item in w.files.list_directory_contents(FAISS_VOLUME_PATH):
        if item.is_directory:
            continue
            
        dest = local_path / item.name
        print(f"  Downloading {item.name}...")
        
        # Download and force write to disk
        with w.files.download(item.path).contents as src:
            with open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno()) # Force write to physical disk

        # 3. Immediate Verification
        if os.path.getsize(dest) == 0:
            raise RuntimeError(f"❌ Download failed: {item.name} is 0 bytes. Check permissions for the App Service Principal.")
    
    return str(index_file), str(meta_file)

# Use the function to get the paths
local_idx_file, local_meta_file = get_local_index()

print("Loading FAISS index …")
_faiss_index = faiss.read_index(local_idx_file)
with open(local_meta_file, "rb") as f:
    _faiss_data = pickle.load(f)

# ─── FAISS + Embeddings (loaded once at startup) ─────────────────────────────
# print("Loading FAISS index …")
# _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
# with open(METADATA_PATH, "rb") as f:
#     _faiss_data = pickle.load(f)

_embed_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
print("Models ready ✅")


# ─── Tool functions ──────────────────────────────────────────────────────────

def search_icar_knowledge(query: str, top_k: int = 3) -> str:
    query_vec = _embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)
    distances, indices = _faiss_index.search(query_vec, top_k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx >= 0:
            chunk  = _faiss_data["chunks"][idx]
            source = _faiss_data["metadata"][idx]["source"]
            results.append(f"[Source: {source}]\n{chunk}")
    return "\n\n---\n\n".join(results)


def get_farmer_soil_health(farmer_id: str) -> dict:
    rows = spark.sql(f"""
        SELECT * FROM krishi_dhwani.gold.farmer_soil_health
        WHERE farmer_id = '{farmer_id}'
    """).collect()
    return rows[0].asDict() if rows else {"error": f"Farmer {farmer_id} not found"}


def get_weather_forecast(pincode: str) -> list:
    rows = spark.sql(f"""
        SELECT forecast_date, rainfall_mm, temp_max_c, condition
        FROM krishi_dhwani.silver.weather
        WHERE pincode = '{pincode}'
        ORDER BY forecast_date
        LIMIT 7
    """).collect()
    return [r.asDict() for r in rows]


def get_market_price(crop: str) -> dict:
    rows = spark.sql(f"""
        SELECT * FROM krishi_dhwani.gold.market_price_trends
        WHERE lower(crop) = lower('{crop}')
    """).collect()
    return rows[0].asDict() if rows else {"error": f"No price data for {crop}"}


# ─── Agent ───────────────────────────────────────────────────────────────────

def run_krishi_agent(english_query: str, farmer_id: str) -> str:
    farmer       = get_farmer_soil_health(farmer_id)
    pincode      = farmer.get("pincode", "110001")
    primary_crop = farmer.get("primary_crop", "Wheat")

    icar_context = search_icar_knowledge(english_query)
    weather      = get_weather_forecast(pincode)
    market       = get_market_price(primary_crop)

    system_prompt = (
        "You are Krishi Dhwani, an expert agricultural advisor for Indian farmers. "
        "You have access to ICAR research, weather forecasts, and market prices. "
        "Give specific, actionable advice. Always cite sources. "
        "Keep answer concise (3–5 sentences). End with a clear recommendation."
    )

    context_block = f"""
FARMER PROFILE:
{json.dumps(farmer, indent=2)}

7-DAY WEATHER (Pincode {pincode}):
{json.dumps(weather, indent=2)}

MARKET PRICE TREND ({primary_crop}):
{json.dumps(market, indent=2)}

ICAR KNOWLEDGE:
{icar_context}
"""
    full_prompt = f"SYSTEM:\n{system_prompt}\n\nCONTEXT:\n{context_block}\n\nUSER:\n{english_query}"
    response = llm.invoke(full_prompt)
    return response.content


# ─── Sarvam helpers ──────────────────────────────────────────────────────────

def sarvam_stt(audio_bytes: bytes, language_code: str = "hi-IN") -> str:
    resp = requests.post(
        f"{SARVAM_BASE}/speech-to-text",
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": "saaras:v3", "language_code": language_code},
        headers={"api-subscription-key": SARVAM_API_KEY},
    )
    resp.raise_for_status()
    return resp.json().get("transcript", "")


def sarvam_translate(text: str, src: str, tgt: str) -> str:
    resp = requests.post(
        f"{SARVAM_BASE}/translate",
        json={
            "input": text,
            "source_language_code": src,
            "target_language_code": tgt,
            "model": "mayura:v1",
            "enable_preprocessing": True,
        },
        headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json().get("translated_text", text)


def sarvam_tts(text: str, language_code: str = "hi-IN") -> bytes:
    resp = requests.post(
        f"{SARVAM_BASE}/text-to-speech",
        json={
            "inputs": [text.replace("\n", " ")[:500]],
            "target_language_code": language_code,
            "speaker": "priya",
            "model": "bulbul:v3",
        },
        headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return base64.b64decode(resp.json()["audios"][0])


# ─── Full pipeline ────────────────────────────────────────────────────────────

def full_voice_pipeline(audio_path: str, farmer_id: str, language: str) -> tuple:
    lang_code = LANGUAGE_MAP.get(language, "hi-IN")
    is_english = language == "English"

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    regional_text  = sarvam_stt(audio_bytes, lang_code)
    english_query  = regional_text if is_english else sarvam_translate(regional_text, lang_code, "en-IN")
    english_answer = run_krishi_agent(english_query, farmer_id)
    regional_answer= english_answer if is_english else sarvam_translate(english_answer, "en-IN", lang_code)
    audio_out      = sarvam_tts(regional_answer, lang_code)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_out)
    tmp.close()

    return tmp.name, regional_answer, english_query, english_answer


# ─── Text chat pipeline ───────────────────────────────────────────────────────

def text_chat_pipeline(message: str, history: list, farmer_id: str, language: str):
    if not message.strip():
        return history, ""
    
    lang_code = LANGUAGE_MAP.get(language, "hi-IN")
    is_english = language == "English"

    english_query  = message if is_english else sarvam_translate(message, lang_code, "en-IN")
    english_answer = run_krishi_agent(english_query, farmer_id)
    regional_answer= english_answer if is_english else sarvam_translate(english_answer, "en-IN", lang_code)

    history.append((message, regional_answer))
    return history, ""
    
# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tiro+Devanagari+Hindi&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-deep:   #1B3A0F;
    --green-mid:    #2D5A1B;
    --green-light:  #4A8C2A;
    --gold:         #C8963E;
    --gold-light:   #E8B96A;
    --cream:        #F5EDD8;
    --cream-dark:   #EAD9B8;
    --soil:         #6B4226;
    --white:        #FDFAF4;
    --text-dark:    #1B2A0E;
    --text-mid:     #3D5228;
    --shadow:       0 4px 24px rgba(27,58,15,0.13);
    --radius:       16px;
}

/* ── Page shell ── */
body, .gradio-container {
    background: linear-gradient(160deg, #EEF5E6 0%, #F5EDD8 60%, #EDE4C8 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 0 16px 48px !important;
}

/* ── Hero header ── */
.kd-hero {
    background: linear-gradient(135deg, var(--green-deep) 0%, var(--green-mid) 60%, var(--green-light) 100%);
    border-radius: 0 0 32px 32px;
    padding: 36px 40px 32px;
    margin: 0 -16px 32px;
    position: relative;
    overflow: hidden;
}

.kd-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}

.kd-hero-title {
    font-family: 'Tiro Devanagari Hindi', serif;
    font-size: 2.4rem;
    color: var(--cream);
    margin: 0 0 6px;
    letter-spacing: -0.5px;
    position: relative;
}

.kd-hero-title span {
    color: var(--gold-light);
}

.kd-hero-sub {
    font-size: 0.95rem;
    color: rgba(245,237,216,0.75);
    margin: 0 0 20px;
    font-weight: 300;
    position: relative;
}

.kd-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    position: relative;
}

.kd-badge {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: var(--cream);
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    backdrop-filter: blur(4px);
}

/* ── Controls row ── */
.kd-controls {
    background: var(--white);
    border: 1px solid var(--cream-dark);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
}

/* ── Tabs ── */
.tabs > .tab-nav {
    background: var(--cream) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--cream-dark) !important;
    margin-bottom: 16px !important;
    gap: 4px !important;
}

.tabs > .tab-nav > button {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--text-mid) !important;
    border: none !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
    background: transparent !important;
}

.tabs > .tab-nav > button.selected {
    background: var(--green-deep) !important;
    color: var(--cream) !important;
    box-shadow: 0 2px 8px rgba(27,58,15,0.25) !important;
}

/* ── Tab panels ── */
.tabitem {
    background: var(--white) !important;
    border: 1px solid var(--cream-dark) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    box-shadow: var(--shadow) !important;
}

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg, var(--green-mid), var(--green-deep)) !important;
    border: none !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(27,58,15,0.3) !important;
    letter-spacing: 0.3px !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(27,58,15,0.4) !important;
}

button.secondary {
    background: var(--cream) !important;
    border: 1.5px solid var(--cream-dark) !important;
    color: var(--text-mid) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 10px !important;
}

/* ── Inputs & labels ── */
label > span, .form > span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-mid) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    margin-bottom: 6px !important;
}

input, textarea, select {
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 10px !important;
    border-color: var(--cream-dark) !important;
    background: var(--white) !important;
    color: var(--text-dark) !important;
}

input:focus, textarea:focus {
    border-color: var(--green-light) !important;
    box-shadow: 0 0 0 3px rgba(74,140,42,0.12) !important;
}

/* ── Dropdowns ── */
.wrap-inner {
    border-radius: 10px !important;
    border-color: var(--cream-dark) !important;
}

/* ── Chatbot ── */
.message-wrap {
    font-family: 'DM Sans', sans-serif !important;
}

.message.user {
    background: linear-gradient(135deg, var(--green-mid), var(--green-deep)) !important;
    color: var(--cream) !important;
    border-radius: 18px 18px 4px 18px !important;
}

.message.bot {
    background: var(--cream) !important;
    color: var(--text-dark) !important;
    border: 1px solid var(--cream-dark) !important;
    border-radius: 18px 18px 18px 4px !important;
}

/* ── Audio component ── */
.audio-component-wrapper {
    border-radius: 12px !important;
    border-color: var(--cream-dark) !important;
}

/* ── Accordion ── */
.accordion > .label-wrap {
    background: var(--cream) !important;
    border-radius: 10px !important;
    border: 1px solid var(--cream-dark) !important;
    color: var(--text-mid) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* ── Footer ── */
.kd-footer {
    text-align: center;
    font-size: 0.8rem;
    color: var(--text-mid);
    margin-top: 32px;
    padding: 20px;
    border-top: 1px solid var(--cream-dark);
    opacity: 0.8;
}

.kd-footer strong { color: var(--green-mid); }

/* ── Responsive ── */
@media (max-width: 600px) {
    .kd-hero { padding: 24px 20px 20px; border-radius: 0 0 20px 20px; }
    .kd-hero-title { font-size: 1.8rem; }
    .tabs > .tab-nav > button { padding: 8px 12px !important; font-size: 0.82rem !important; }
    .tabitem { padding: 16px !important; }
    .kd-controls { padding: 16px !important; }
}
"""

with gr.Blocks(
    title="🌾 Krishi Dhwani",
    theme=gr.themes.Base(),
    css=CSS,
) as demo:

    # ── Hero ──────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="kd-hero">
        <div class="kd-hero-title">🌾 <span>Krishi</span> Dhwani</div>
        <div class="kd-hero-sub">Voice-Enabled Agricultural Advisory · कृषि ध्वनि</div>
        <div class="kd-badges">
            <span class="kd-badge">🌦 Live Weather</span>
            <span class="kd-badge">📈 Mandi Prices</span>
            <span class="kd-badge">📚 ICAR Knowledge</span>
            <span class="kd-badge">🗣 9 Indian Languages</span>
            <span class="kd-badge">⚡ Llama 4 Maverick</span>
        </div>
    </div>
    """)

    # ── Shared controls ───────────────────────────────────────────────────────
    with gr.Group(elem_classes="kd-controls"):
        with gr.Row():
            farmer_id = gr.Dropdown(
                choices=["F001 — Rajinder Singh (Punjab, Wheat)", "F002 — Sunita Devi (Rajasthan, Mustard)"],
                value="F001 — Rajinder Singh (Punjab, Wheat)",
                label="👤 Farmer Profile",
            )
            language = gr.Dropdown(
                choices=list(LANGUAGE_MAP.keys()),
                value="Hindi",
                label="🗣️ Language",
            )

    # Mapping display → actual farmer_id
    FARMER_MAP = {
        "F001 — Rajinder Singh (Punjab, Wheat)": "F001",
        "F002 — Sunita Devi (Rajasthan, Mustard)": "F002",
    }

    # ── Tabs ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Voice tab ─────────────────────────────────────────────────────────
        with gr.Tab("🎤 Voice Advisory"):

            gr.HTML("<p style='color:#3D5228;font-size:0.88rem;margin:0 0 16px;'>Record your question or upload a WAV file. You will hear the answer in your language.</p>")

            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Your Question",
            )

            submit_btn = gr.Button("🚀 Get Advisory", variant="primary", size="lg")

            with gr.Row():
                audio_out    = gr.Audio(label="🔊 Listen to Advisory", type="filepath")
                regional_out = gr.Textbox(label="📝 Advisory Text", lines=5, show_copy_button=True)

            with gr.Accordion("🔍 Debug — Internal Details", open=False):
                with gr.Row():
                    english_q   = gr.Textbox(label="English Query", lines=2)
                    english_ans = gr.Textbox(label="English Answer", lines=4)

            def voice_fn(audio, farmer_display, lang):
                return full_voice_pipeline(audio, FARMER_MAP[farmer_display], lang)

            submit_btn.click(
                fn=voice_fn,
                inputs=[audio_in, farmer_id, language],
                outputs=[audio_out, regional_out, english_q, english_ans],
                api_name=False,
            )

        # ── Chat tab ──────────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):

            gr.HTML("<p style='color:#3D5228;font-size:0.88rem;margin:0 0 16px;'>Type in your language. Press Enter or Send — the bot replies in the same language.</p>")

            chatbot = gr.Chatbot(
                label="",
                height=420,
                bubble_full_width=False,
                show_label=False,
                placeholder="<div style='text-align:center;color:#6B8F4A;padding:40px 0'>🌾 Ask anything about your crops, soil, weather, or market prices</div>",
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="अपना सवाल यहाँ लिखें... / Type your question here...",
                    label="",
                    scale=9,
                    container=False,
                    lines=1,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)

            def chat_fn(message, history, farmer_display, lang):
                return text_chat_pipeline(message, history, FARMER_MAP[farmer_display], lang)

            send_btn.click(
                fn=chat_fn,
                inputs=[chat_input, chatbot, farmer_id, language],
                outputs=[chatbot, chat_input],
                api_name=False,
            )
            chat_input.submit(
                fn=chat_fn,
                inputs=[chat_input, chatbot, farmer_id, language],
                outputs=[chatbot, chat_input],
                api_name=False,
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="kd-footer">
        <strong>Krishi Dhwani</strong> · Built on Databricks Free Edition ·
        Sarvam AI (STT · Translate · TTS) · Llama 4 Maverick · ICAR Knowledge Base
        <br>Made with ❤️ for the next billion farmers · Bharat Bricks Hackathon 2026
    </div>
    """)

# =========================================================
# SAFE DATABRICKS APP LAUNCH
# =========================================================

if __name__ == "__main__":

    demo.queue()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.getenv("DATABRICKS_APP_PORT", "8000")
        ),
        share=False,
        show_api=False   # 🔥 prevents gradio schema bug
    )