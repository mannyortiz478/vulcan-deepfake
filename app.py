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
# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Voice Guard", page_icon="🛡️")

# --- CACHE MODEL LOADING ---
# This ensures the model only loads ONCE when the server starts
@st.cache_resource
def get_detector():
    model_path = "mesonet_whisper_mfcc_finetuned.pth"
    config_path = "configs/finetuning/whisper_frontend_mesonet.yaml"
    return MesoNetDetector(model_path, config_path)

detector = get_detector()

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