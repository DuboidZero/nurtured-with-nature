import streamlit as st

main_page = st.Page("main_page.py", title="Home", icon="🏠")
page_2 = st.Page("quiz.py", title="Stress Quiz", icon="📰")
page_3 = st.Page("bot.py", title="AnahataAI", icon="🍃")

pg = st.navigation([main_page, page_2, page_3])

pg.run()
