import streamlit as st
from ma import MedicalInferenceEngine

# Page Configuration
st.set_page_config(
    page_title="MediSage AI | Clinical Expert",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .medical-card {
        background-color: white; 
        padding: 30px; 
        border-radius: 20px;
        border-left: 10px solid #007bff; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    .stButton>button {
        background: linear-gradient(to right, #007bff, #00d2ff);
        color: white; 
        border-radius: 12px; 
        font-weight: 700; 
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Engine
@st.cache_resource
def get_engine():
    return MedicalInferenceEngine()

engine = get_engine()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3843/3843180.png", width=90)
    st.header("Control Panel")
    mode = st.radio("Inference Strategy:", ["Forward Chaining", "Backward Chaining"])
    st.divider()
    st.info("Hybrid Neuro-Symbolic System\nPharos University")

# Main Header
st.title("🏥 MediSage AI - Clinical Support")
st.caption("Forward & Backward Chaining | Powered by Llama 3.2 + Chroma DB")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# User Input
if prompt := st.chat_input("Enter symptoms (e.g., stomach pain and vomiting and fever)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing Clinical Correlation..."):
            try:
                if prompt.lower() in ["hi", "hello", "hey", "صباح الخير", "سلام", "مرحبا"]:
                    response = "## 👋 Hello! Please describe the patient's symptoms."
                else:
                    # Input Cleaning - Supports "and" and ","
                    clean_input = prompt.replace('،', ',') \
                                       .replace(' and ', ',') \
                                       .replace(',,', ',') \
                                       .strip()
                    
                    # Remove trailing or leading comma
                    if clean_input.endswith(','):
                        clean_input = clean_input[:-1]
                    if clean_input.startswith(','):
                        clean_input = clean_input[1:]
                    
                    s_list = [s.strip() for s in clean_input.split(',') if s.strip()]
                    mode_key = mode.replace(" ", "")
                    
                    if mode_key == "ForwardChaining":
                        results = engine.forward_inference(s_list)
                    else:
                        results = engine.backward_inference(
                            s_list[0], 
                            s_list[1:] if len(s_list) > 1 else []
                        )
                    
                    report_content = engine.generate_explanation(results, prompt, mode_key)
                    response = f"<div class='medical-card'>{report_content}</div>"
                
                st.markdown(response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")