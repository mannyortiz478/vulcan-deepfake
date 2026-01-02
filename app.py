import streamlit as st
import plotly.graph_objects as go


def create_custom_gauge(probability):
    """Create a three-band gauge for deepfake triage."""
    score = float(probability) * 100.0

    if score <= 29:
        status_text = "REAL"
        status_color = "#2ecc71"
    elif 30 <= score <= 69:
        status_text = "MOST LIKELY FAKE (BUT COULD BE REAL)"
        status_color = "#f1c40f"
    else:
        status_text = "DEEPFAKE (NOT REAL)"
        status_color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>{status_text}</b>",
            'font': {'size': 20, 'color': status_color}
        },
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            # Needle color set to requested hex
            'bar': {'color': '#669bbc'},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 29], 'color': 'rgba(46, 204, 113, 0.8)'},
                {'range': [30, 69], 'color': 'rgba(241, 196, 15, 0.8)'},
                {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.8)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


import sys
import os
# Add the current directory to the search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
from inference import MesoNetDetector
from pathlib import Path

# Try to import the Hugging Face helper. If it's unavailable we'll fall back.
try:
    from huggingface_hub import hf_hub_download  # type: ignore
    HAS_HF_HUB = True
except Exception:
    hf_hub_download = None  # pragma: no cover - optional dependency
    HAS_HF_HUB = False
# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Voice Guard", page_icon="🛡️")

# --- CACHE MODEL LOADING ---
# This ensures the model only loads ONCE when the server starts
@st.cache_resource
def get_detector():
    # Allow deployment to override model path via env var
    model_path = os.environ.get("MODEL_PATH", "mesonet_whisper_mfcc_finetuned.pth")
    config_path = os.environ.get("MODEL_CONFIG", "configs/finetuning/whisper_frontend_mesonet.yaml")

    hf_repo_id = os.environ.get("HF_REPO_ID")
    hf_token = os.environ.get("HF_TOKEN")
    auto_download = os.environ.get("MODEL_AUTO_DOWNLOAD", "1").lower() not in ("0", "false")

    # Try a few common candidate locations before attempting downloads
    candidates = [model_path]
    if not os.path.isabs(model_path):
        candidates.extend([
            os.path.join(os.getcwd(), model_path),
            os.path.join("models", model_path),
            os.path.join("/opt/render/project/src/models", model_path),
        ])

    found = None
    for p in candidates:
        if os.path.exists(p):
            found = p
            break

    # If not found and HF info is provided, attempt to download from Hugging Face
    if not found and hf_repo_id and auto_download:
        if not HAS_HF_HUB:
            st.warning(
                "Hugging Face Hub not available in this environment. To enable automatic downloads, "
                "install the `huggingface-hub` package (e.g. `pip install huggingface-hub`)."
            )
        else:
            # Determine the filename we expect
            filename = os.path.basename(model_path) if model_path else "mesonet_whisper_mfcc_finetuned.pth"
            target_dir = Path("models")
            target_dir.mkdir(parents=True, exist_ok=True)

            with st.spinner(f"Downloading model {filename} from Hugging Face ({hf_repo_id})..."):
                try:
                    downloaded = hf_hub_download(
                        repo_id=hf_repo_id,
                        filename=filename,
                        token=hf_token,
                        local_dir=str(target_dir),
                    )
                    if downloaded and os.path.exists(downloaded):
                        found = downloaded
                        st.success(f"Model downloaded to {downloaded}")
                except Exception as e:
                    st.error(f"Failed to download model from Hugging Face: {e}")

    if not found:
        # Return None so the UI can display a friendly error message
        return None

    return MesoNetDetector(found, config_path)

detector = get_detector()
if detector is None:
    st.error(
        "Model file not found. The app tried to locate the model but couldn't find it. "
        "Please set the MODEL_PATH environment variable to the model file path, or place `mesonet_whisper_mfcc_finetuned.pth` in the project root or `models/` folder. "
        "See README.md for download and placement instructions."
    )
    st.stop()

# --- UI DESIGN ---
st.title("🛡️ AI Voice Deepfake Detector")
st.markdown("Upload an audio clip to verify if it's a **Human Voice** or **AI-Generated Synthetic Speech**.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # 1. Save temporary file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 2. Display Audio Player
    st.audio(uploaded_file)
    
    if st.button("🔍 Analyze Audio"):
        with st.spinner("Analyzing vocal artifacts..."):
            # 3. Run Inference
            result = detector.predict("temp_audio.wav")
            
            # 4. Create a custom triage gauge
            prob = float(result.get('raw_score', 0.0))
            fig = create_custom_gauge(prob)
            st.plotly_chart(fig)

            # 5. Final Verdict
            if result['is_fake']:
                st.error(f"🚨 **Verdict: HIGH RISK.** This audio shows signs of AI generation (Confidence: {result['confidence']*100:.1f}%)")
            else:
                st.success(f"✅ **Verdict: LIKELY REAL.** This audio appears to be human (Confidence: {result['confidence']*100:.1f}%)")

# --- FOOTER ---
st.divider()
st.caption("Note: This tool is for educational purposes and provides a probability score based on MesoNet + Whisper features.")