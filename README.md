# NURTURED WITH NATURE
#### Video Demo:  https://youtu.be/l-MOkPKU-oQ
#### Description:
    


# 🌿 Nurtured with Nature

**Nurtured with Nature** is a wellness-focused web application built using Streamlit. It blends mental health analysis and personalized naturopathy recommendations using AI — helping users reconnect with natural healing practices through:

- ✅ A machine-learning powered **Stress Analysis Quiz**
- 💬 A conversational **Naturopathy Chatbot** (Anahata AI)
- 🌱 A database of holistic remedies and wellness rituals

---

## 🔧 Tech Stack

- **Frontend & UI**: Streamlit
- **ML Model**: scikit-learn (KNN Classifier)
- **NLP Engine**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Testing**: pytest
- **Other**: pandas, torch

---

## 📁 Project Structure

nurtured-with-nature/
│
├── app.py # Main Streamlit app (quiz + chatbot)
├── predictor.py # ML logic for stress classification
├── anahata_core.py # NLP logic for chatbot detection & remedies
├── sample_data.xlsx # Training data for the stress quiz
├── test_project.py # Pytest-based unit tests
├── 2.png # Image used in UI
├── sidebarlogo.png # Sidebar logo for branding
└── README.md # This file


---

## 🧠 Features

### 1. 🧪 Stress Analysis Quiz
- 10 slider-based questions on anxiety, sleep, pain, etc.
- Internally scores responses as "low" and "high" stress contributors
- Uses a trained **K-Nearest Neighbors (KNN)** classifier to predict total stress level:
  - 😁 Low
  - 😐 Moderate
  - 😥 High

### 2. 🌿 Anahata AI Chatbot
- Users describe how they feel in natural language
- Chatbot uses **SentenceTransformer** to semantically match ailments
- Responds with:
  - ✅ 3 natural home remedies
  - 🧘 3 wellness practices
- Supports saving favorite remedies with a "⭐ Save" button

---

## 🧪 Testing

Tests are written using **pytest** and are located in `test_project.py`. Two main modules are covered:

### 🔹 Stress Prediction (`predictor.py`)
| Test Name                | Description                                  |
|-------------------------|----------------------------------------------|
| `test_preprocess_data`  | Checks stress score transformation logic     |
| `test_train_knn`        | Validates KNN training on input data         |
| `test_predict_stress`   | Ensures prediction on new input works        |

### 🔹 Ailment Chatbot (`anahata_core.py`)
| Test Name                          | Description                                     |
|-----------------------------------|-------------------------------------------------|
| `test_detect_ailment_exact_match` | Tests NLP matching on valid ailments            |
| `test_detect_ailment_unrelated_text` | Tests fallback when input doesn’t match    |
| `test_get_remedies_and_wellness_valid_ailment` | Checks remedy and wellness return     |
| `test_get_remedies_and_wellness_invalid_ailment` | Ensures graceful failure on unknowns |

---

## Design Decisions
- Modular Separation: Core logic (ML + NLP) is separated from the UI to enable testing.
  [Original code has logic built in. Core logic was simply split to cover the testing requirement of CS50P.]

- Semantic Matching: Unlike keyword-based chatbots, this app uses SentenceTransformer to allow flexible inputs like “my mind feels foggy” → matches "brain fog".
  
[Initially I wanted to use AI features for the chatbot, however due to A) monetary restrictions with AI APIs and B) hardware restrictions (im running and coding all of this on a 8 year old laptop), I had to shift to semantic matching. Initial code used large volumes of dictionaries that were painstakingly made to fit each and every response a user could make, however that proved difficult as you had to give the exact phrases instead of natural language, thus the choice to use semantic matching using sentencetransformer]

---

## 💚 Built With Purpose
- This project blends modern AI tools with ancient healing wisdom to help people slow down, reflect, and reconnect with nature.
