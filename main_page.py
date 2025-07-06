import streamlit as st

st.image('naturelogo.png')
st.logo('sidebarlogo.png', size = "large")

st.markdown("**Welcome to _Nurtured with Nature_ — your personal guide to mental wellness through naturopathy.**")
st.markdown("We offer two core features designed to support your well-being:")
st.markdown("### 1. Stress Level Analyzer 😩\nHarness the power of smart technology and nature-backed science. Our Stress Level Analyzer uses a refined K-Nearest Neighbor algorithm, trained on insights gathered from extensive research and real consultations with experienced naturopathic doctors. This means your stress levels are evaluated with a high degree of care and accuracy — just like a practitioner would.")
st.page_link("quiz.py", label = "Go to Stress Level Analyzer", icon = "📰")
st.markdown("### 2. Anahata AI 🍀\nMeet **Anahata AI**, your personal, in-house naturopathy companion. Whether you're dealing with a splitting headache, feeling anxious, or just need a natural remedy to unwind, Anahata is here to help — anytime, anywhere. No more second-guessing or Googling symptoms. Just ask, and let Anahata guide you toward balance and peace.")
st.page_link("bot.py", label = "Talk to Anahata AI", icon = "🍃")
st.markdown("*To get started, simply click on a feature from the sidebar or click the icons above*")
st.markdown("*--- Your journey to a calmer, healthier you begins now.*")