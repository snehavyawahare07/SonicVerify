"""
SonicVerify — AI-Powered Real-Time Voice Cloning & Impersonation Risk Detector
--------------------------------------------------------------------------------
A Streamlit front-end for a voice-integrity verification tool. Users can
record or upload an audio clip; the app analyzes it and produces a
Risk % score (probability the voice is AI-generated / cloned), a
breakdown across Acoustic / Prosody / Spectral dimensions, a plain-
language explanation, alerts, and recommended actions.
"""

import io
import time
import wave
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
import requests
BACKEND_URL = os.getenv(
    "SONICVERIFY_BACKEND_URL",
    "https://sonicverify.onrender.com"
)

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SonicVerify — Voice Cloning Risk Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    "theme": "dark",
    "page": "Dashboard",
    "history": [],           # list of result dicts (no raw audio)
    "current_audio": None,   # np.ndarray samples
    "current_sr": None,
    "current_source": None,  # filename / "Live Recording"
    "last_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================================
# THEME & CSS INJECTION
# ============================================================================
def inject_theme(theme: str):
    if theme == "dark":
        bg, bg2, text, sub, card, border = (
            "#070c1f", "#0f1a3a", "#f4f6fc", "#a8b6db", "#121f45", "#22325f",
        )
        grad1, grad2 = "#ff7a1a", "#ffa94d"
    else:
        bg, bg2, text, sub, card, border = (
            "#fdf6ee", "#f7ead9", "#1a2140", "#5c6690", "#fffaf2", "#ecd9bd",
        )
        grad1, grad2 = "#e8590c", "#ff8a3d"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&display=swap');

        /* Default Font Scope */
        :root {{
            --font-display: 'Sora', sans-serif;
            --primary-color: {grad1};
            --background-color: {bg};
            --secondary-background-color: {card};
            --text-color: {text};
            color: {text};
        }}

        /* Apply Sora font globally without breaking Material Symbols/Icons */
        html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-icons):not([class*="material-symbols"]):not(i) {{
            font-family: "Sora", sans-serif;
            color: {text};
        }}

        /* Global Markdown & Paragraph Visibility Fix */
        .stApp p, .stApp span, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
            color: {text} !important;
        }}

        /* STRICT FIX FOR DOUBLE ARROW TEXT / MATERIAL ICONS */
        [data-testid="stIconMaterial"],
        span[data-testid="stIconMaterial"],
        .material-icons,
        .material-symbols-outlined,
        .material-symbols-rounded,
        [class*="material-symbols"],
        [data-testid="stExpander"] summary svg,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            font-weight: normal !important;
            font-style: normal !important;
            line-height: 1 !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            word-wrap: normal !important;
            white-space: nowrap !important;
            direction: ltr !important;
            -webkit-font-smoothing: antialiased !important;
        }}

        /* App Backgrounds */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        .main, .block-container, [data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
        [data-testid="stForm"], [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {{
            background-color: {bg} !important;
        }}
        header[data-testid="stHeader"] {{ background-color: {bg} !important; }}
        header[data-testid="stHeader"] * {{ color: {text} !important; fill: {text} !important; }}
        [data-testid="stToolbar"] {{ background-color: {bg} !important; }}
        section[data-testid="stSidebar"] {{ background-color: {bg2} !important; border-right: 1px solid {border} !important; }}
        section[data-testid="stSidebar"] * {{ color: {text} !important; }}

        /* SELECTBOX & DROPDOWN THEME MATCHING FIX */
        [data-testid="stSelectbox"] > div > div {{
            background-color: {card} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stSelectbox"] * {{
            color: {text} !important;
            fill: {text} !important;
        }}
        [data-baseweb="select"] * {{
            background-color: {card} !important;
            color: {text} !important;
        }}
        [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"] {{
            background-color: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
        }}
        li[role="option"] {{
            background-color: {card} !important;
            color: {text} !important;
        }}
        li[role="option"]:hover, li[aria-selected="true"] {{
            background-color: {bg2} !important;
            color: {grad1} !important;
        }}

        /* Expander Styling & Fixes */
        [data-testid="stExpander"] {{
            background-color: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 12px !important;
            overflow: hidden;
            margin-bottom: 12px;
        }}
        [data-testid="stExpander"] details {{
            background-color: {card} !important;
        }}
        [data-testid="stExpander"] summary {{
            background-color: {card} !important;
            color: {text} !important;
            padding: 12px 16px !important;
            border-radius: 12px !important;
        }}
        [data-testid="stExpander"] summary:hover {{
            background-color: {bg2} !important;
        }}
        [data-testid="stExpander"] div[role="group"] {{
            background-color: {card} !important;
            padding: 16px !important;
        }}

        /* UPLOAD AUDIO & FILE UPLOADER THEME MATCHING */
        [data-testid="stFileUploader"] {{
            background-color: {card} !important;
            border: 1px solid {border} !important;
            border-radius: 14px !important;
            padding: 16px !important;
        }}
        [data-testid="stFileUploader"] label, 
        [data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small {{
            color: {text} !important;
            font-weight: 600 !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {bg2} !important;
            border: 2px dashed {border} !important;
            border-radius: 12px !important;
            color: {text} !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {text} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {card} !important;
            color: {text} !important;
            border: 1px solid {border} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            border-color: {grad1} !important;
            color: {grad1} !important;
        }}
        [data-testid="stFileUploaderFile"] {{
            background-color: {bg2} !important;
            border: 1px solid {border} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploaderFile"] * {{
            color: {text} !important;
        }}
        [data-testid="stFileUploaderFileName"] {{
            color: {text} !important;
            font-weight: 500 !important;
        }}

        /* Audio Recorder Theme Match */
        [data-testid="stAudioInput"] {{
            background-color: {card} !important;
            border: 1px dashed {border} !important;
            border-radius: 14px !important;
            padding: 14px !important;
        }}
        [data-testid="stAudioInput"] * {{
            background-color: transparent !important;
            color: {text} !important;
        }}
        [data-testid="stAudioInput"] svg {{ fill: {grad1} !important; color: {grad1} !important; }}

        /* Typography & Custom Elements */
        .sv-hero {{
            background: linear-gradient(120deg, {grad1}22, {grad2}22);
            border: 1px solid {border};
            border-radius: 20px; padding: 28px 32px; margin-bottom: 22px;
        }}
        .sv-hero h1 {{
            font-weight: 700 !important;
            font-size: clamp(32px, 4vw, 50px) !important;
            letter-spacing: -0.04em !important;
            line-height: 1.1 !important;
            margin: 0;
            background: linear-gradient(90deg, {grad1}, {grad2});
            -webkit-background-clip: text; background-clip: text; color: transparent !important;
        }}
        .sv-hero p {{
            color: {sub} !important; margin-top: 8px; font-size: 1.02rem;
            line-height: 1.6 !important;
        }}

        /* Card Container & Explicit Text Color Fixes */
        .sv-card {{
            background-color: {card} !important; border: 1px solid {border} !important;
            border-radius: 16px; padding: 22px 24px; margin-bottom: 18px;
            color: {text} !important;
        }}
        .sv-card *, .sv-card p, .sv-card li, .sv-card span, .sv-card div {{
            color: {text} !important;
        }}

        .sv-badge {{
            display: inline-block; padding: 5px 14px; border-radius: 999px;
            font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em;
        }}
        .sv-alert-high {{
            background: #e74c3c22; border: 1px solid #e74c3c; color: #ff6b5b !important;
            border-radius: 12px; padding: 14px 18px; font-weight: 600;
        }}
        .sv-alert-med {{
            background: #f1c40f22; border: 1px solid #f1c40f; color: #f1c40f !important;
            border-radius: 12px; padding: 14px 18px; font-weight: 600;
        }}
        .sv-alert-low {{
            background: #2ecc7122; border: 1px solid #2ecc71; color: #2ecc71 !important;
            border-radius: 12px; padding: 14px 18px; font-weight: 600;
        }}
        .sv-subtle {{ color: {sub} !important; }}
        .sv-navlabel {{ color: {sub} !important; font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: 0.08em; margin: 14px 0 4px 2px; }}
        .sv-bar-track {{
            background-color: {border}; border-radius: 8px; height: 12px; width: 100%;
            overflow: hidden; margin-top: 4px;
        }}
        .sv-bar-fill {{ height: 100%; border-radius: 8px; }}

        /* Buttons & Metrics */
        div[data-testid="stMetric"] {{
            background-color: {card} !important; border: 1px solid {border} !important;
            border-radius: 14px; padding: 14px 18px;
        }}
        [data-testid="stMetricLabel"] p {{ color: {sub} !important; }}
        [data-testid="stMetricValue"] {{ color: {text} !important; }}

        .stButton>button, .stDownloadButton>button {{
            border-radius: 10px; font-weight: 600; border: 1px solid {border};
            background-color: {card}; color: {text} !important;
        }}
        .stButton>button p {{ color: {text} !important; }}
        .stButton>button:hover {{ border: 1px solid {grad1}; color: {grad1} !important; }}
        .stButton>button:hover p {{ color: {grad1} !important; }}
        button[kind="primary"] {{
            background: linear-gradient(90deg, {grad1}, {grad2}) !important;
            border: none !important;
        }}
        button[kind="primary"] p {{ color: #0a1128 !important; font-weight: 700; }}

        /* Custom HTML Table */
        .sv-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
        .sv-table th {{
            text-align: left; color: {sub} !important; font-weight: 600; font-size: 0.78rem;
            text-transform: uppercase; letter-spacing: 0.05em;
            padding: 8px 12px; border-bottom: 1px solid {border};
        }}
        .sv-table td {{ padding: 10px 12px; border-bottom: 1px solid {border}; color: {text} !important; }}
        .sv-table tr:hover td {{ background-color: {bg2} !important; }}
        audio {{ border-radius: 10px; width: 100%; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return dict(bg=bg, bg2=bg2, text=text, sub=sub, card=card, border=border, grad1=grad1, grad2=grad2)


C = inject_theme(st.session_state.theme)


def risk_color(risk):
    if risk >= 65:
        return "#e74c3c"
    if risk >= 35:
        return "#f1c40f"
    return "#2ecc71"


def render_table(rows, columns):
    head = "".join(f"<th>{label}</th>" for _, label in columns)
    body_rows = []
    for r in rows:
        cells = "".join(f"<td>{r.get(k, '')}</td>" for k, _ in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    html = f'<table class="sv-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)


def score_bar(label, value, color):
    st.markdown(
        f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; font-size:0.92rem;">
            <span style="color:{C['text']};">{label}</span><span style="font-weight:700; color:{C['text']};">{value:.0f}%</span>
          </div>
          <div class="sv-bar-track">
            <div class="sv-bar-fill" style="width:{value}%; background-color:{color};"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# AUDIO LOADING
# ============================================================================
def load_audio(uploaded_file):
    raw = uploaded_file.read()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    if HAS_SOUNDFILE:
        try:
            data, sr = sf.read(io.BytesIO(raw), always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32), sr
        except Exception:
            pass

    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            sampwidth = wf.getsampwidth()
            n_channels = wf.getnchannels()
            frames = wf.readframes(n_frames)
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
            data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
            data = data / float(np.iinfo(dtype).max)
            return data, sr
    except Exception:
        return None, None


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================
def compute_scores(samples: np.ndarray, sr: int):
    duration = len(samples) / sr if sr else 0
    frame_len = max(1, int(sr * 0.02)) if sr else 512
    n_frames = max(1, len(samples) // frame_len)
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)

    energy_var = float(np.var(energy))
    silence_ratio = float(np.mean(energy < (0.02 * (energy.max() + 1e-9))))
    zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2)

    spec = np.abs(np.fft.rfft(samples * np.hanning(len(samples)))) + 1e-9
    spectral_flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))

    acoustic = 100 * np.clip(1 - spectral_flatness * 3, 0, 1)
    prosody = 100 * np.clip(1 - abs(zcr - 0.08) * 4, 0, 1) * (1 - 0.5 * silence_ratio)
    prosody = float(np.clip(prosody, 0, 100))
    spectral_consistency = 100 * np.clip(1 - abs(energy_var - 0.01) * 8, 0, 1)

    naturalness = 0.4 * acoustic + 0.3 * prosody + 0.3 * spectral_consistency
    risk = float(np.clip(100 - naturalness, 0, 100))

    if risk >= 65:
        verdict, tier = "Likely AI-Generated / Cloned Voice", "high"
    elif risk >= 35:
        verdict, tier = "Uncertain — Possible AI Voice", "medium"
    else:
        verdict, tier = "Likely Human Voice", "low"

    return {
        "duration": duration, "sample_rate": sr,
        "energy_var": energy_var, "silence_ratio": silence_ratio,
        "zcr": zcr, "spectral_flatness": spectral_flatness,
        "acoustic": float(acoustic), "prosody": float(prosody),
        "spectral": float(spectral_consistency),
        "risk": risk, "verdict": verdict, "tier": tier,
        "color": risk_color(risk),
    }


def explain(res):
    lines = []
    if res["spectral_flatness"] > 0.25:
        lines.append("The frequency spectrum is unusually flat/uniform — a pattern often seen in neural speech synthesis rather than natural vocal tracts.")
    else:
        lines.append("Spectral texture shows the natural peaks and troughs typical of a human vocal tract.")

    if res["silence_ratio"] > 0.35:
        lines.append("An abnormally high proportion of near-silent frames was detected, which can indicate splicing or generation artifacts.")
    elif res["silence_ratio"] < 0.05:
        lines.append("Very little natural pausing was found — continuous, unbroken speech can be a synthetic-voice indicator.")
    else:
        lines.append("Pause and silence patterns fall within a normal conversational range.")

    if abs(res["zcr"] - 0.08) > 0.05:
        lines.append("Zero-crossing rate (a proxy for voicing/noisiness) deviates from typical human speech norms.")
    else:
        lines.append("Zero-crossing rate is consistent with natural voiced/unvoiced speech transitions.")

    if res["energy_var"] < 0.003:
        lines.append("Loudness/energy stays oddly constant over time — human speech usually has more natural dynamic variation.")
    else:
        lines.append("Energy dynamics (loudness variation) look consistent with natural speech delivery.")
    return lines


def recommendations(tier):
    if tier == "high":
        return [
            "🚨 Do NOT approve any transaction, share credentials, or disclose confidential information on this call.",
            "📞 Terminate the call and re-establish contact using a known, previously verified phone number.",
            "🧑‍💼 Escalate immediately to your security / fraud team and log the incident.",
            "🔐 Trigger secondary verification: multi-factor authentication or a pre-agreed passphrase.",
        ]
    if tier == "medium":
        return [
            "⚠️ Pause before acting — ask a personal or pre-agreed verification question the caller should know.",
            "📞 Offer to call back on a known/registered number before proceeding.",
            "🧑‍💼 Loop in a supervisor for high-value or sensitive requests.",
            "🗒️ Log this interaction for review even if it ultimately proceeds.",
        ]
    return [
        "✅ No immediate action required — signals are consistent with a genuine human voice.",
        "🔁 For very high-value transactions, periodic re-verification is still good practice.",
        "🗂️ Archive the interaction summary for audit trail purposes.",
    ]


def precautions(tier):
    if tier == "high":
        return [
            "Stop the call now. Do not send money, share OTPs, or reveal confidential information.",
            "Call the person back on a number you already trust, not one the caller gives you.",
            "Report this to your security or fraud team right away and keep the recording.",
        ]
    if tier == "medium":
        return [
            "Do not share OTPs, passwords, or card details until the caller is verified.",
            "Ask a question only the real person would know, or call back on a saved number.",
            "Ignore urgency or pressure. Take your time and involve a supervisor.",
        ]
    return [
        "Signals look consistent with a genuine human voice, but no automated check is perfect.",
        "For high-value requests, still confirm through a second channel before acting.",
        "Keep a record of the call in case you need to review it later.",
    ]


# ============================================================================
# CHARTS
# ============================================================================
def waveform_fig(samples, sr, color, upto=None):
    n = upto if upto else len(samples)
    fig, ax = plt.subplots(figsize=(9, 2.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    t = np.linspace(0, len(samples) / sr if sr else len(samples), num=len(samples))
    ax.plot(t[:n], samples[:n], linewidth=0.6, color=color)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Time (s)", color=C["text"])
    ax.set_yticks([])
    ax.tick_params(colors=C["text"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def spectrogram_fig(samples, sr):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.specgram(samples, Fs=sr if sr else 22050, cmap="magma")
    ax.set_xlabel("Time (s)", color=C["text"])
    ax.set_ylabel("Freq (Hz)", color=C["text"])
    ax.tick_params(colors=C["text"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def gauge_fig(risk, color):
    fig, ax = plt.subplots(figsize=(3.2, 3.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    track = "#22325f" if st.session_state.theme == "dark" else "#ecd9bd"
    ax.pie([risk, 100 - risk], colors=[color, track], startangle=90,
           counterclock=False, wedgeprops={"width": 0.32, "edgecolor": "none"})
    ax.text(0, 0.08, f"{risk:.0f}%", ha="center", va="center", fontsize=28,
            fontweight="bold", color=C["text"])
    ax.text(0, -0.28, "AI Voice Risk", ha="center", va="center", fontsize=10,
            color=C["sub"])
    return fig


def animate_waveform(samples, sr, color):
    placeholder = st.empty()
    n = len(samples)
    steps = min(24, max(4, n // 4000))
    for i in range(1, steps + 1):
        upto = int(n * i / steps)
        placeholder.pyplot(waveform_fig(samples, sr, color, upto=upto), use_container_width=True)
        time.sleep(0.02)
    placeholder.pyplot(waveform_fig(samples, sr, color), use_container_width=True)


# ============================================================================
# ANALYSIS RUNNER
# ============================================================================
def send_to_backend(
    audio_bytes,
    filename,
    content_type,
    phone_number,
    financial_request=False,
    identity_claim="",
):
    response = requests.post(
        f"{BACKEND_URL}/api/analyze",
        files={
            "audio": (
                filename,
                audio_bytes,
                content_type or "audio/wav",
            )
        },
        data={
            "phone_number": phone_number,
            "financial_request": str(financial_request).lower(),
            "identity_claim": identity_claim,
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()


def run_backend_analysis(
    audio_bytes,
    source_name,
    content_type,
    phone_number,
    samples=None,
    sr=None,
):
    try:
        result = send_to_backend(
            audio_bytes=audio_bytes,
            filename=source_name,
            content_type=content_type,
            phone_number=phone_number,
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Could not connect to the SonicVerify backend. "
            "Please check that the backend is running."
        )
        return False

    except requests.exceptions.Timeout:
        st.error(
            "❌ The backend took too long to analyze the audio. "
            "Please try a shorter recording."
        )
        return False

    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Backend returned an error: {e}")
        return False

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
        return False

    if not result.get("success"):
        st.error(
            result.get(
                "error",
                "The backend could not analyze this audio."
            )
        )
        return False

    risk = result["risk_assessment"]
    voice = result["voice_analysis"]

    risk_score = float(risk["risk_score"]) * 100

    if risk["risk_level"] == "HIGH":
        tier = "high"
        verdict = "Likely AI-Generated / Cloned Voice"

    elif risk["risk_level"] == "UNCERTAIN":
        tier = "medium"
        verdict = "Uncertain — Possible AI Voice"

    else:
        tier = "low"
        verdict = "Likely Human Voice"

    synthetic_probability = voice.get("synthetic_probability")
    authentic_probability = voice.get("authentic_probability")
    confidence = voice.get("confidence")

    res = {
        "filename": source_name,
        "timestamp": dt.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "risk": risk_score,
        "verdict": verdict,
        "tier": tier,
        "color": risk_color(risk_score),

        "synthetic_probability": (
            float(synthetic_probability) * 100
            if synthetic_probability is not None
            else None
        ),

        "authentic_probability": (
            float(authentic_probability) * 100
            if authentic_probability is not None
            else None
        ),

        "confidence": (
            float(confidence) * 100
            if confidence is not None
            else None
        ),

        "risk_level": risk["risk_level"],
        "summary": voice.get("summary", ""),
        "reasons": risk.get("reasons", []),
        "recommendation": result.get("recommendation", ""),
        "disclaimer": result.get("disclaimer", ""),

        "analysis_id": result.get("analysis_id"),

        "duration": (
            len(samples) / sr
            if samples is not None and sr
            else 0
        ),

        "sample_rate": sr,

        "phone_number": phone_number,
    }

    st.session_state.current_audio = samples
    st.session_state.current_sr = sr
    st.session_state.current_source = source_name
    st.session_state.last_result = res

    st.session_state.history.append(res)

    st.session_state.page = "Analysis Result"

    return True

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
NAV = [
    ("Dashboard", "🏠"),
    ("Record Audio", "🎙️"),
    ("Upload Audio", "📁"),
    ("Analysis Result", "📊"),
    ("Alerts & History", "🔔"),
    ("Recommendations", "🧭"),
]

with st.sidebar:
    st.markdown("## 🛡️ SonicVerify")
    st.caption("Real-time voice cloning & impersonation risk detection")
    st.markdown('<div class="sv-navlabel">Navigate</div>', unsafe_allow_html=True)
    for name, icon in NAV:
        is_active = st.session_state.page == name
        if st.button(f"{icon}  {name}", key=f"nav_{name}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = name
            st.rerun()

    st.markdown("---")
    theme_choice = st.toggle("🌙 Dark mode", value=(st.session_state.theme == "dark"))
    new_theme = "dark" if theme_choice else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("---")
    st.caption(
        "⚠️ Prototype heuristic engine — analyzes acoustic, prosody and "
        "spectral signal statistics. Not a certified forensic tool; use "
        "alongside human judgment and secondary verification."
    )

page = st.session_state.page


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "Dashboard":
    st.markdown(
        """
        <div class="sv-hero">
          <h1>🛡️ SonicVerify</h1>
          <p>AI-powered real-time detection of voice cloning &amp; synthetic-speech
          impersonation — screen calls before you trust them.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hist = st.session_state.history
    total = len(hist)
    high = sum(1 for h in hist if h["tier"] == "high")
    avg_risk = np.mean([h["risk"] for h in hist]) if hist else 0
    last_verdict = hist[-1]["verdict"] if hist else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", total)
    c2.metric("High-Risk Flags", high)
    c3.metric("Average Risk", f"{avg_risk:.0f}%")
    c4.metric("Last Verdict", last_verdict)

    st.markdown("#### Quick Actions")
    a, b, c = st.columns(3)
    with a:
        if st.button("🎙️ Record a Call / Voice", use_container_width=True):
            st.session_state.page = "Record Audio"; st.rerun()
    with b:
        if st.button("📁 Upload an Audio File", use_container_width=True):
            st.session_state.page = "Upload Audio"; st.rerun()
    with c:
        if st.button("🔔 View Alerts & History", use_container_width=True):
            st.session_state.page = "Alerts & History"; st.rerun()

    st.markdown("#### Recent Activity")
    if not hist:
        st.info("No scans yet. Record or upload a clip to get started.")
    else:
        recent = []
        for h in hist[-5:][::-1]:
            recent.append({
                "timestamp": h["timestamp"], "filename": h["filename"],
                "verdict_html": f'<span style="color:{h["color"]}; font-weight:700;">{h["verdict"]}</span>',
                "risk_html": f'<b>{h["risk"]:.0f}%</b>',
            })
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        render_table(recent, [("timestamp", "Time"), ("filename", "Source"),
                               ("verdict_html", "Verdict"), ("risk_html", "Risk")])
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# PAGE: RECORD AUDIO
# ============================================================================
elif page == "Record Audio":
    st.markdown('<div class="sv-hero"><h1>🎙️ Record Audio</h1><p>Record a live sample directly from your microphone to screen it for cloning risk.</p></div>', unsafe_allow_html=True)

    if "recorder_key" not in st.session_state:
        st.session_state.recorder_key = 0

    with st.expander("🔧 If recording doesn't start, read this first", expanded=False):
        st.markdown(
            """
            - Open the app's **Local URL** in a real browser tab (Chrome, Edge, or
              Firefox) — recording will **not** work inside VS Code's built-in
              "Simple Browser" preview or other embedded webviews.
            - When you click the mic icon, your browser will ask for **microphone
              permission** — click **Allow**. If you accidentally blocked it,
              click the 🔒/ⓘ icon in the address bar and re-enable the mic for
              this site, then refresh the page.
            - Speak for at least 1–2 seconds — very short clips can't be analyzed.
            - If none of this helps, use **Upload Audio** instead with a
              pre-recorded WAV file — the analysis works identically either way.
            """
        )

    rec = None
    audio_input_supported = hasattr(st, "audio_input")
    if not audio_input_supported:
        st.error(
            "Your installed Streamlit version doesn't support in-browser "
            "recording. Run `pip install -U streamlit`, restart the "
            "app, then reload this page. In the meantime, use **Upload Audio**."
        )
    else:
        try:
            rec = st.audio_input("Tap the microphone to record",
                                  key=f"recorder_{st.session_state.recorder_key}")
        except Exception as e:
            st.error(f"Recording widget failed to load: {e}")

    if rec is not None:
        raw_bytes = rec.getvalue() if hasattr(rec, "getvalue") else rec.read()
        st.success(f"✅ Captured {len(raw_bytes)/1024:.0f} KB of audio.")
        st.audio(rec)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            analyze_clicked = st.button("▶️ Analyze Recording", type="primary", use_container_width=True)
        with col_b:
            if st.button("🔄 Re-record", use_container_width=True):
                st.session_state.recorder_key += 1
                st.rerun()

        if analyze_clicked:
            try:
                rec.seek(0)
            except Exception:
                pass
            samples, sr = load_audio(rec)
            if samples is None:
                st.error(
                    "Couldn't decode this recording. Make sure `soundfile` "
                    "installed correctly (`pip install soundfile`), then try "
                    "again — or use Upload Audio with a WAV file instead."
                )
            elif sr and len(samples) / sr < 0.3:
                st.warning("That recording was too short to analyze. Please record at least 1–2 seconds of speech.")
            else:
                with st.spinner("Analyzing acoustic, prosody and spectral patterns…"):
                    time.sleep(0.6)
                run_analysis(samples, sr, "Live Recording")
                st.rerun()
    else:
        st.info("Click the microphone icon above to start recording.")


# ============================================================================
# PAGE: UPLOAD AUDIO
# ============================================================================
elif page == "Upload Audio":
    st.markdown('<div class="sv-hero"><h1>📁 Upload Audio</h1><p>Upload a call recording or voice sample (WAV / FLAC / OGG) for analysis.</p></div>', unsafe_allow_html=True)

    up = st.file_uploader("Upload Audio File", type=["wav", "flac", "ogg"])
    if up is not None:
        st.audio(up)
        if st.button("▶️ Analyze File", type="primary"):
            samples, sr = load_audio(up)
            if samples is None:
                st.error("Could not read this file. Please try a standard PCM WAV file.")
            else:
                with st.spinner("Analyzing acoustic, prosody and spectral patterns…"):
                    time.sleep(0.6)
                run_analysis(samples, sr, up.name)
                st.rerun()

    with st.expander("📦 Batch-analyze multiple files"):
        batch = st.file_uploader("Choose multiple audio files", type=["wav", "flac", "ogg"],
                                  accept_multiple_files=True, key="batch_uploader")
        if batch and st.button("▶️ Analyze All"):
            prog = st.progress(0, text="Starting…")
            for i, f in enumerate(batch):
                prog.progress(i / len(batch), text=f"Analyzing {f.name}…")
                samples, sr = load_audio(f)
                if samples is not None:
                    res = compute_scores(samples, sr)
                    res["filename"] = f.name
                    res["timestamp"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append(res)
                    st.session_state.last_result = res
            prog.progress(1.0, text="Done")
            time.sleep(0.3)
            st.success(f"Analyzed {len(batch)} files — see Alerts & History.")


# ============================================================================
# PAGE: ANALYSIS RESULT
# ============================================================================
elif page == "Analysis Result":
    res = st.session_state.last_result
    if res is None:
        st.markdown('<div class="sv-hero"><h1>📊 Analysis Result</h1><p>No analysis yet.</p></div>', unsafe_allow_html=True)
        st.info("Go to **Record Audio** or **Upload Audio** to run your first check.")
    else:
        st.markdown(
            f'<div class="sv-hero"><h1>📊 Analysis Result</h1>'
            f'<p>Source: <b>{res["filename"]}</b> · Analyzed {res["timestamp"]}</p></div>',
            unsafe_allow_html=True,
        )

        alert_class = {"high": "sv-alert-high", "medium": "sv-alert-med", "low": "sv-alert-low"}[res["tier"]]
        alert_msg = {
            "high": "🚨 HIGH RISK — Strong signs of AI-generated or cloned voice detected.",
            "medium": "⚠️ MEDIUM RISK — Some signals are inconsistent with natural human speech. Proceed with caution.",
            "low": "✅ LOW RISK — Signals are consistent with a genuine human voice.",
        }[res["tier"]]
        st.markdown(f'<div class="{alert_class}">{alert_msg}</div>', unsafe_allow_html=True)

        tier_title = {"high": "High risk precautions", "medium": "Medium risk precautions", "low": "Low risk precautions"}[res["tier"]]
        st.markdown(
            f'<div class="sv-card" style="border:1px solid {res["color"]}; margin-top:12px;">'
            f'<div style="font-weight:700; color:{res["color"]} !important; margin-bottom:8px;">{tier_title}</div>'
            + "".join(f'<div style="margin:6px 0; color:{C["text"]};">{i}. {line}</div>' for i, line in enumerate(precautions(res["tier"]), 1))
            + "</div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1.4])
        with left:
            st.pyplot(gauge_fig(res["risk"], res["color"]), use_container_width=True)
            st.markdown(
                f'<div style="text-align:center;">'
                f'<span class="sv-badge" style="background-color:{res["color"]}22; color:{res["color"]}; border:1px solid {res["color"]}">{res["verdict"]}</span>'
                f'</div>', unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Duration", f"{res['duration']:.1f}s")
            m2.metric("Sample rate", f"{res['sample_rate']} Hz" if res["sample_rate"] else "—")

        with right:
            st.markdown("**Risk Breakdown**")
            score_bar("Acoustic Authenticity", res["acoustic"], "#ff7a1a")
            score_bar("Prosody Naturalness", res["prosody"], "#ffa94d")
            score_bar("Spectral Consistency", res["spectral"], "#5b8def")
            st.caption("Higher bars = more consistent with genuine human speech.")

        st.markdown("#### 🧠 Why this score?")
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        for line in explain(res):
            st.markdown(f"- {line}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### 🌊 Waveform")
        if st.session_state.current_audio is not None and st.session_state.current_source == res["filename"]:
            animate_waveform(st.session_state.current_audio, st.session_state.current_sr, res["color"])
            st.markdown("#### 🎼 Spectrogram")
            st.pyplot(spectrogram_fig(st.session_state.current_audio, st.session_state.current_sr), use_container_width=True)
        else:
            st.caption("Waveform unavailable for this record (re-run analysis to view it).")

        st.markdown("#### ✅ Recommended Actions")
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        for r in recommendations(res["tier"]):
            st.markdown(f"- {r}")
        st.markdown("</div>", unsafe_allow_html=True)

        report_text = (
            f"SonicVerify Voice Integrity Report\n"
            f"{'='*40}\n"
            f"Source: {res['filename']}\n"
            f"Analyzed: {res['timestamp']}\n\n"
            f"VERDICT: {res['verdict']}\n"
            f"AI Voice Risk: {res['risk']:.1f}%\n\n"
            f"Breakdown:\n"
            f"  Acoustic Authenticity:  {res['acoustic']:.0f}%\n"
            f"  Prosody Naturalness:    {res['prosody']:.0f}%\n"
            f"  Spectral Consistency:   {res['spectral']:.0f}%\n\n"
            f"Explanation:\n" + "\n".join(f"  - {l}" for l in explain(res)) + "\n\n"
            f"Recommended Actions:\n" + "\n".join(f"  - {r}" for r in recommendations(res["tier"])) + "\n\n"
            f"Note: heuristic prototype score — not a certified forensic result.\n"
        )
        st.download_button("⬇️ Share / Download Report (.txt)", data=report_text,
                            file_name=f"SonicVerify_Report_{res['filename']}.txt",
                            mime="text/plain", use_container_width=True)


# ============================================================================
# PAGE: ALERTS & HISTORY
# ============================================================================
elif page == "Alerts & History":
    st.markdown('<div class="sv-hero"><h1>🔔 Alerts &amp; History</h1><p>Every scan run this session, with risk level and verdict.</p></div>', unsafe_allow_html=True)

    hist = st.session_state.history
    if not hist:
        st.info("No scans yet — nothing to show here.")
    else:
        filt = st.selectbox("Filter by risk level", ["All", "High", "Medium", "Low"])
        rows = hist
        if filt != "All":
            rows = [h for h in hist if h["tier"] == filt.lower()]

        for h in reversed(rows):
            badge_color = h["color"]
            st.markdown(
                f'<div class="sv-card" style="display:flex; justify-content:space-between; align-items:center;">'
                f'<div><b>{h["filename"]}</b><br>'
                f'<span class="sv-subtle">{h["timestamp"]}</span></div>'
                f'<div style="text-align:right;">'
                f'<span class="sv-badge" style="background-color:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}">{h["verdict"]}</span><br>'
                f'<span style="font-weight:800; font-size:1.2rem;">{h["risk"]:.0f}%</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        df = pd.DataFrame(hist)[["timestamp", "filename", "verdict", "risk", "acoustic", "prosody", "spectral"]]
        df.columns = ["Time", "Source", "Verdict", "Risk %", "Acoustic", "Prosody", "Spectral"]
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Export history (CSV)", data=df.to_csv(index=False),
                                file_name="sonicverify_history.csv", mime="text/csv",
                                use_container_width=True)
        with c2:
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.history = []
                st.session_state.last_result = None
                st.rerun()


# ============================================================================
# PAGE: RECOMMENDATIONS
# ============================================================================
elif page == "Recommendations":
    st.markdown('<div class="sv-hero"><h1>🧭 Recommendations Playbook</h1><p>General best practices for handling suspected voice-cloning or impersonation calls.</p></div>', unsafe_allow_html=True)

    cols = st.columns(3)
    playbooks = [
        ("🟢 Low Risk", "#2ecc71", recommendations("low")),
        ("🟡 Medium Risk", "#f1c40f", recommendations("medium")),
        ("🔴 High Risk", "#e74c3c", recommendations("high")),
    ]
    text_color = C["text"]
    for col, (title, color, items) in zip(cols, playbooks):
        with col:
            items_html = "".join(f"<p style='color:{text_color} !important;'>{i}</p>" for i in items)
            st.markdown(
                f'<div class="sv-card"><h4 style="color:{color} !important;">{title}</h4>'
                + items_html
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### General Guidance")
    st.markdown(
        f"""
        <div class="sv-card">
        <ul style="color:{C['text']} !important;">
          <li style="color:{C['text']} !important;">Never approve high-value transfers or share credentials based on a phone call alone — always verify through a second, independent channel.</li>
          <li style="color:{C['text']} !important;">Establish pre-agreed verification phrases with executives and finance teams for sensitive requests.</li>
          <li style="color:{C['text']} !important;">Treat urgency and pressure tactics ("do this now, don't tell anyone") as a red flag regardless of the risk score.</li>
          <li style="color:{C['text']} !important;">Log every flagged interaction, even low-risk ones, to help spot patterns over time.</li>
          <li style="color:{C['text']} !important;">Keep this tool's output as one input among several — combine it with organizational verification policy.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
