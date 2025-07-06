import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2', device = 'cpu')

# --------------------------------------------
# Remedies and Wellness Packs
# --------------------------------------------

naturopathy_remedies = {
    "headache": [
        "Drink ginger tea to reduce inflammation.",
        "Apply peppermint oil on temples.",
        "Soak feet in hot water to draw blood away from the head."
    ],
    "cold": [
        "Steam inhalation with eucalyptus oil.",
        "Tulsi and honey tea.",
        "Gargle with warm salt water."
    ],
    "indigestion": [
        "Chew fennel seeds after meals.",
        "Drink warm water with lemon.",
        "Avoid cold or processed foods."
    ],
    "stress": [
        "Practice deep belly breathing.",
        "Take a lavender oil bath.",
        "Use Ashwagandha as a calming adaptogen."
    ],
    "acne": [
        "Apply neem paste to affected areas.",
        "Drink plenty of water with mint leaves.",
        "Avoid dairy and fried foods."
    ],
    "psoriasis": [
        "Apply aloe vera gel directly to skin.",
        "Take sun exposure in moderation.",
        "Consume turmeric with black pepper for anti-inflammatory effect."
    ],
    "IBS (Irritable Bowel Syndrome)": [
        "Eat cooked apples with cinnamon.",
        "Use triphala as a gentle digestive tonic.",
        "Practice yoga postures like Pavanamuktasana."
    ],
    "insomnia": [
        "Drink nutmeg and warm milk before bed.",
        "Massage soles of feet with sesame oil.",
        "Use Brahmi or Jatamansi herbs."
    ],
    "vertigo": [
        "Drink coriander seed and gooseberry (amla) infusion.",
        "Practice slow head-tilt exercises.",
        "Avoid excess salt intake."
    ],
    "fatty liver": [
        "Drink dandelion root tea.",
        "Include bitter foods like karela (bitter gourd) in diet.",
        "Avoid alcohol and refined sugar completely."
    ],
    "fatigue": [
        "Drink ashwagandha tea daily.",
        "Take short outdoor walks every 90 mins.",
        "Use Brahmi oil for scalp massage."
    ],
    "brain fog": [
        "Take Brahmi or Shankhpushpi syrup.",
        "Avoid processed sugar; try soaked almonds instead.",
        "Inhale rosemary essential oil for mental clarity."
    ],
    "back pain": [
        "Stretch using yoga poses like Bhujangasana and Marjaryasana.",
        "Apply hot castor oil on lower back.",
        "Use turmeric in warm milk at night."
    ],
    "low mood": [
        "Drink tulsi tea with honey and lemon.",
        "Use St. John’s Wort (under supervision).",
        "Sunbathe in early morning hours."
    ],
    "sugar cravings": [
        "Take cinnamon water before breakfast.",
        "Chew fennel and ajwain after meals.",
        "Avoid white sugar and switch to jaggery or raw honey."
    ]
}

wellness_packs = {
    "stress": [
        "🧘 10 minutes deep breathing",
        "🌿 Walk barefoot on grass",
        "☀️ Morning sunlight for 15 mins"
    ],
    "insomnia": [
        "🌙 Limit screens 1hr before bed",
        "🛁 Warm bath with Epsom salt",
        "📿 Listen to soft instrumental music"
    ],
    "indigestion": [
        "🥗 Light home-cooked meals",
        "🚶 Post-meal walking for 10 mins",
        "🧃 Herbal digestive tea at night"
    ],
    "cold": [
        "🫖 Steam once a day",
        "💧 Stay hydrated",
        "🌿 Use a room diffuser with eucalyptus"
    ],
    "acne": [
        "🧼 Wash face with neem water",
        "🚫 No dairy/fried foods",
        "🍃 Drink coriander & mint juice"
    ],
    "psoriasis": [
        "🌞 10-15 min sun exposure",
        "💧 Hydrate well",
        "🍵 Anti-inflammatory turmeric tea"
    ],
    "IBS (Irritable Bowel Syndrome)": [
        "🧘 Practice Pavanamuktasana daily",
        "🍏 Eat cooked apples in the morning",
        "💧 Drink cumin + fennel + ajwain water"
    ],
    "headache": [
        "🦶 Foot soak in warm water",
        "🛏️ Take power naps (15 mins)",
        "🧴 Peppermint balm on temples"
    ],
    "vertigo": [
        "🌀 Do Epley maneuver slowly",
        "🥒 Avoid salty snacks",
        "🛏️ Rest in dark quiet space"
    ],
    "fatty liver": [
        "🍵 Bitter gourd juice in morning",
        "🥬 Eat leafy greens",
        "🚶 Walk after each meal"
    ],
    "fatigue": [
        "☀️ Morning sunlight for 15 mins",
        "💤 Power nap (15–20 mins)",
        "🧘 5 minutes of deep breathing"
    ],
    "brain fog": [
        "🧘 Alternate nostril breathing",
        "🍵 Green tea with lemon",
        "🧴 Rosemary oil diffuser in workspace"
    ],
    "back pain": [
        "🧘 Cat-Cow pose every 2 hours",
        "🌿 Warm castor oil compress before bed",
        "💺 Sit on firm cushion, maintain lumbar support"
    ],
    "low mood": [
        "☀️ 15 min morning sun exposure",
        "🧘 Chant Om for 5 mins",
        "🎧 Listen to binaural beats"
    ],
    "sugar cravings": [
        "🧃 Cinnamon + amla water in morning",
        "🚶 Light walk post meals",
        "🥗 Include high-fiber snacks like roasted chana"
    ]
}

# Precompute sentence embeddings
ailments_list = list(naturopathy_remedies.keys())
ailment_embeddings = model.encode(ailments_list)

# --------------------------------------------
# Streamlit UI Setup
# --------------------------------------------

st.set_page_config(page_title="Anahata AI", layout="centered")
page = st.sidebar.radio("🧭 Navigate", ["Chatbot", "Saved Remedies"])

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_ailment" not in st.session_state:
    st.session_state.last_ailment = None
if "last_remedy" not in st.session_state:
    st.session_state.last_remedy = None
if "saved_remedies" not in st.session_state:
    st.session_state.saved_remedies = []

# --------------------------------------------
# Saved Remedies Page
# --------------------------------------------

if page == "Saved Remedies":
    st.title("💾 Saved Remedies")
    if st.session_state.saved_remedies:
        for i, (ailment, remedy) in enumerate(st.session_state.saved_remedies):
            with st.container():
                st.markdown(f"### {ailment}\n{remedy}")
                if st.button("🗑️ Delete", key=f"del_{i}"):
                    st.session_state.saved_remedies.pop(i)
                    st.rerun()
    else:
        st.info("You haven’t saved any remedies yet.")
    st.stop()

# --------------------------------------------
# Chatbot Page
# --------------------------------------------

st.title("🌿 Anahata AI")
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    st.session_state.last_ailment = None
    st.session_state.last_remedy = None
    st.success("Chat cleared.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Tell me how you're feeling...")

if user_input:
    user_query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # SentenceTransformer Matching
    user_embedding = model.encode(user_query)
    similarity_scores = util.cos_sim(user_embedding, ailment_embeddings)[0]
    top_idx = int(torch.argmax(similarity_scores))
    best_match = ailments_list[top_idx]
    similarity_score = similarity_scores[top_idx].item()

    if similarity_score >= 0.4:
        matched_ailment = best_match
        st.session_state.last_ailment = matched_ailment
        remedies = naturopathy_remedies[matched_ailment]
        wellness = wellness_packs.get(matched_ailment, [])

        response = f"Looks like you're experiencing **{matched_ailment}**.\n\n"
        response += "### 🌿 Remedies:\n"
        for item in remedies:
            response += f"- {item}\n"

        if wellness:
            response += "\n### 🧘 Daily Wellness Pack:\n"
            for item in wellness:
                response += f"- {item}\n"

        st.session_state.last_remedy = response
    else:
        response = "🤔 Hmm... I couldn't detect an ailment from that. Try saying something like *headache*, *tired*, or *can’t sleep*."

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

if st.session_state.get("last_ailment") and st.session_state.get("last_remedy"):
    if st.button("⭐ Save this remedy"):
        st.session_state.saved_remedies.append((
            st.session_state.last_ailment,
            st.session_state.last_remedy
        ))
        st.success("Remedy saved!")
