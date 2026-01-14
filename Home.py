import streamlit as st
from streamlit_option_menu import option_menu

# Page config
st.set_page_config(page_title="SmartKhet", layout="wide")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Disease Detection", "Crop Recommendation", "Fertilizer Suggestion", "Action Advisory", "About"],
        icons=["house", "activity", "bar-chart", "flask", "lightbulb", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Custom centered header with tighter spacing
st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; padding-top: 10px; padding-bottom: 5px;">
        <h1 style='color: #2E8B57; font-size: 42px; margin: 0;'>🌾 SmartKhet 
        <span style='font-size: 20px; color: gray;'>(Agriculture + Innovation)</span></h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Home Page
if selected == "Home":
    with st.container():
        st.markdown("## 👋 Welcome to SmartKhet!")
        st.markdown('> *"Empowering Farmers with Innovation and Intelligence."*')

        st.markdown("""
        **SmartKhet** is your all-in-one smart farming companion 🌿.
        This platform supports farmers and agri-enthusiasts by integrating machine learning–driven solutions to optimize crop health, productivity, and sustainability.
        """)

    st.markdown("### 🔧 Core Features")

    col1, col2 = st.columns(2)
    with col1:
        st.success("🌱 **Plant Disease Detection**\n\nDetect leaf and crop diseases using deep learning.")
        st.warning("💊 **Fertilizer Suggestion**\n\nChoose the right fertilizers based on soil and crop type.")
    with col2:
        st.info("🌾 **Crop Recommendation System**\n\nRecommend best-suited crops based on region and season.")
        st.error("📈 **Action Advisory**\n\nGet timely advice based on agri-trends and weather data.")

    st.markdown("### 🚀 Quick Navigation")
    col3, col4 = st.columns(2)
    with col3:
        st.page_link("pages/disease_detection.py", label="🔍 Disease Detection", icon="🦠")
        st.page_link("pages/fertilizer_suggestion.py", label="💊 Fertilizer Suggestion", icon="🧪")
    with col4:
        st.page_link("pages/crop_recommendation.py", label="🌿 Crop Recommendation", icon="🌾")
        st.page_link("pages/action_advisory.py", label="📈 Action Advisory", icon="📊")

    st.markdown("----")

# About Page
elif selected == "About":
    st.markdown("## 🤝 About Us")
    st.write("""
        **SmartKhet** is a smart farming web platform developed by a passionate team from **NIT Delhi** 🚀.

        ### 👨‍💻 Team Members:
        - **Aman Kumar** – 231220008 
        - **Srijan Kumar** – 231220062

        ### 🌟 Our Mission:
        To empower farmers through a fusion of:
        - 🌐 Modern technology  
        - 📊 Data-driven decisions  
        - 🌿 Sustainable agricultural practices  

        We believe that the future of agriculture lies in **innovation** — and SmartKhet is your partner in that journey.
    """)

    st.info("📬 For collaborations or queries, reach out at: 231220008@nitdelhi.ac.in")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>© 2025 SmartKhet. Made for Smart Farming.</p>", unsafe_allow_html=True)
