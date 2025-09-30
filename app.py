import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from model.fracture_detector import FractureDetector
from utils.image_processor import process_xray_image, enhance_bone_visibility

# Page configuration
st.set_page_config(
    page_title="Bone Fracture Detector",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fracture-detected {
        background-color: #ffebee;
        border-left: 5px solid #ff4b4b;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .no-fracture {
        background-color: #e8f5e8;
        border-left: 5px solid #00cc96;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-high { color: #ff4b4b; font-weight: bold; font-size: 1.3rem; }
    .confidence-medium { color: #ffa500; font-weight: bold; font-size: 1.3rem; }
    .confidence-low { color: #00cc96; font-weight: bold; font-size: 1.3rem; }
    .bone-area {
        background-color: #fff3cd;
        border: 2px dashed #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_fracture_model():
    """Load and cache the fracture detection model"""
    try:
        model = FractureDetector()
        model.load_model()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def create_confidence_gauge(confidence):
    """Create a visual confidence gauge"""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Create horizontal bar
    colors = ['#00cc96', '#ffa500', '#ff4b4b']
    if confidence < 0.3:
        color = colors[0]
    elif confidence < 0.7:
        color = colors[1]
    else:
        color = colors[2]
    
    ax.barh(0, confidence, color=color, height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(f'Fracture Confidence: {confidence*100:.1f}%', fontsize=14, fontweight='bold')
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🦴 AI Bone Fracture Detector</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Bone Fracture Detection AI! Upload an X-ray image to detect potential bone fractures.
    This tool uses ResNet-50 deep learning model trained on bone fracture datasets.
    """)
    
    # Sidebar
    st.sidebar.title("About Fracture Detection")
    st.sidebar.info("""
    **Fracture Detection AI** features:
    - Detects fractures in various bones
    - Provides confidence scores
    - Highlights potential fracture areas
    - Supports multiple bone types
    
    **Common Fracture Types:**
    - Simple fractures
    - Compound fractures
    - Hairline fractures
    - Greenstick fractures
    - Comminuted fractures
    """)
    
    st.sidebar.warning("""
    ⚠️ **Medical Disclaimer**
    This tool is for educational and research purposes only. 
    Always consult radiologists and healthcare professionals 
    for accurate medical diagnosis.
    """)
    
    # Load model
    with st.spinner("🔄 Loading Fracture Detection Model... This may take a few seconds."):
        model = load_fracture_model()
    
    if model is None:
        st.error("Failed to load the fracture detection model. Please check the setup.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a bone X-ray image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload X-ray images of arms, legs, ribs, or other bones"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original X-Ray Image", use_column_width=True)
            
            # Enhancement options
            st.subheader("🛠️ Image Enhancement")
            enhance = st.checkbox("Enhance bone visibility", value=True)
            
            if enhance:
                enhanced_image = enhance_bone_visibility(image)
                st.image(enhanced_image, caption="Enhanced X-Ray Image", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Detect Fractures", type="primary", use_container_width=True):
                with st.spinner("🦴 Analyzing bone structure for fractures..."):
                    try:
                        # Process image
                        processed_image = process_xray_image(image)
                        
                        # Make prediction
                        fracture_prob, heatmap = model.predict_with_heatmap(processed_image)
                        
                        # Store results in session state
                        st.session_state.fracture_prob = fracture_prob
                        st.session_state.heatmap = heatmap
                        st.session_state.original_image = image
                        
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
    
    with col2:
        st.subheader("📊 Detection Results")
        
        if uploaded_file is None:
            st.info("👆 Upload a bone X-ray image to analyze for fractures")
        elif 'fracture_prob' in st.session_state:
            display_fracture_results(
                st.session_state.fracture_prob, 
                st.session_state.heatmap,
                st.session_state.original_image
            )

def display_fracture_results(fracture_prob, heatmap, original_image):
    """Display fracture detection results"""
    
    st.subheader("🩺 Fracture Analysis")
    
    # Confidence gauge
    gauge_fig = create_confidence_gauge(fracture_prob)
    st.pyplot(gauge_fig)
    
    # Diagnosis result
    if fracture_prob > 0.7:
        st.markdown('<div class="fracture-detected">', unsafe_allow_html=True)
        st.error("🚨 **FRACTURE DETECTED**")
        st.write(f"**Confidence Level:** {fracture_prob*100:.2f}%")
        st.warning("High probability of bone fracture detected in the image")
        st.markdown('</div>', unsafe_allow_html=True)
        
    elif fracture_prob > 0.3:
        st.markdown('<div class="fracture-detected">', unsafe_allow_html=True)
        st.warning("⚠️ **POSSIBLE FRACTURE**")
        st.write(f"**Confidence Level:** {fracture_prob*100:.2f}%")
        st.info("Moderate probability of fracture. Further examination recommended.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="no-fracture">', unsafe_allow_html=True)
        st.success("✅ **NO FRACTURE DETECTED**")
        st.write(f"**Confidence Level:** {(1-fracture_prob)*100:.2f}%")
        st.info("No significant signs of bone fracture detected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed analysis
    st.subheader("🔬 Detailed Analysis")
    
    col_anal1, col_anal2 = st.columns(2)
    
    with col_anal1:
        st.metric("Fracture Probability", f"{fracture_prob*100:.2f}%")
        st.metric("Normal Bone Confidence", f"{(1-fracture_prob)*100:.2f}%")
    
    with col_anal2:
        if fracture_prob > 0.7:
            risk_level = "High"
        elif fracture_prob > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        st.metric("Risk Level", risk_level)
    
    # Medical recommendations
    st.subheader("💡 Recommended Actions")
    
    if fracture_prob > 0.7:
        st.error("""
        🏥 **Immediate Actions Required:**
        - Seek emergency medical attention
        - Immobilize the affected area
        - Apply ice to reduce swelling
        - Avoid putting weight on injured bone
        - Get follow-up X-rays and CT scan
        """)
    elif fracture_prob > 0.3:
        st.warning("""
        🩺 **Recommended Steps:**
        - Schedule orthopedic consultation
        - Get additional imaging if needed
        - Limit activity in affected area
        - Monitor for pain or swelling changes
        - Consider follow-up X-ray in 1-2 weeks
        """)
    else:
        st.success("""
        ✅ **General Advice:**
        - No immediate intervention needed
        - Continue normal activities
        - Maintain bone health with calcium/vitamin D
        - Regular check-ups if symptoms persist
        """)
    
    # Bone health tips
    with st.expander("💪 Bone Health Tips"):
        st.write("""
        **To maintain strong bones:**
        - Consume calcium-rich foods (dairy, leafy greens)
        - Get adequate vitamin D (sunlight, supplements)
        - Regular weight-bearing exercises
        - Avoid smoking and excessive alcohol
        - Regular bone density checks after 50
        """)

if __name__ == "__main__":
    main()
