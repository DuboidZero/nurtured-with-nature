# anahata_core.py
from sentence_transformers import SentenceTransformer, util
import torch

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

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

ailments_list = list(naturopathy_remedies.keys())
ailment_embeddings = model.encode(ailments_list)

def detect_ailment(user_query: str, threshold=0.4) -> tuple[str, float]:
    user_embedding = model.encode(user_query)
    similarity_scores = util.cos_sim(user_embedding, ailment_embeddings)[0]
    top_idx = int(torch.argmax(similarity_scores))
    top_score = similarity_scores[top_idx].item()
    if top_score >= threshold:
        return ailments_list[top_idx], top_score
    return None, top_score

def get_remedies_and_wellness(ailment: str) -> dict:
    return {
        "remedies": naturopathy_remedies.get(ailment, []),
        "wellness": wellness_packs.get(ailment, [])
    }


# predictor.py
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'How often have you felt anxious for situations which were not important/major?': 'anxious',
        'How difficult is it to overcome the situations where you are anxious?': 'overcome_anxious',
        'How often do you face trouble in sleeping?': 'sleeping',
        'How often had you been idle/trying to sleep and having a chaotic mind full of thoughts?': 'try_sleeping',
        'How often have you not felt confident in your ability to handle challenge related to work,family or physical pain/disease?': 'noconfidence',
        'How often do you take/consider painkillers, instead of resting, while facing mild pain?': 'painkillers',
        'How often have you pushed yourself to be awake due to work or your lifestyle?': 'try2awake',
        'How varying is your wake-up time of weekends or holidays compared to weekdays?': 'sleep_pattern',
        'How often do you engage in relaxation techniques such as meditation or deep breathing?': 'relaxation',
        'How often do you took a break by going on a nature walk or any other nature related activity?': 'nature_Activity'
    }
    dataframe.rename(columns=rename_map, inplace=True)
    dataframe = pd.DataFrame(dataframe, columns=[
        'anxious','overcome_anxious','sleeping','try_sleeping','noconfidence',
        'painkillers','try2awake','sleep_pattern','relaxation','nature_Activity'
    ])
    
    dataframe['low'] = dataframe.iloc[:, :8].sum(axis=1)
    dataframe['high'] = dataframe.iloc[:, 8:].sum(axis=1)

    dataframe['low'] = dataframe['low'].apply(lambda i: 3 if 30 <= i <= 40 else 2 if 20 <= i <= 29 else 1 if 1 <= i <= 19 else 4)
    dataframe['high'] = dataframe['high'].apply(lambda i: 1 if 8 <= i <= 10 else 2 if 6 <= i <= 7 else 3 if 1 <= i <= 5 else 4)
    
    dataframe['overall'] = dataframe['low'] + dataframe['high']
    return dataframe

def train_knn(df: pd.DataFrame):
    x = df.iloc[:, :10].values
    y = df['overall']
    model = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
    model.fit(x, y)
    return model

def predict_stress(model, user_inputs: list) -> int:
    cols = ['anxious','overcome_anxious','sleeping','try_sleeping','noconfidence',
            'painkillers','try2awake','sleep_pattern','relaxation','nature_Activity']
    test = pd.DataFrame([user_inputs], columns=cols)
    return model.predict(test)[0]
