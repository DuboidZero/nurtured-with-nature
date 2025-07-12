import pytest
import pandas as pd
from anahata_core import detect_ailment, get_remedies_and_wellness
from predictor import preprocess_data, train_knn, predict_stress

# -------- Tests for anahata_core.py -------- #

def test_detect_ailment_exact_match():
    ailment, score = detect_ailment("I have a headache")
    assert ailment == "headache"
    assert score >= 0.4

def test_detect_ailment_partial_match():
    ailment, score = detect_ailment("My head hurts and feels heavy")
    assert ailment == "headache"
    assert score >= 0.4

def test_detect_ailment_no_match():
    ailment, score = detect_ailment("I love chocolate cake and playing guitar")
    assert ailment is None
    assert score < 0.4

def test_get_remedies_and_wellness_exists():
    data = get_remedies_and_wellness("cold")
    assert isinstance(data, dict)
    assert "remedies" in data and "wellness" in data
    assert len(data["remedies"]) > 0
    assert len(data["wellness"]) > 0

def test_get_remedies_and_wellness_not_exists():
    data = get_remedies_and_wellness("unknown_ailment")
    assert data["remedies"] == []
    assert data["wellness"] == []


# -------- Tests for predictor.py -------- #

@pytest.fixture
def sample_dataframe():
    data = {
        'How often have you felt anxious for situations which were not important/major?': [3],
        'How difficult is it to overcome the situations where you are anxious?': [3],
        'How often do you face trouble in sleeping?': [2],
        'How often had you been idle/trying to sleep and having a chaotic mind full of thoughts?': [3],
        'How often have you not felt confident in your ability to handle challenge related to work,family or physical pain/disease?': [2],
        'How often do you take/consider painkillers, instead of resting, while facing mild pain?': [2],
        'How often have you pushed yourself to be awake due to work or your lifestyle?': [3],
        'How varying is your wake-up time of weekends or holidays compared to weekdays?': [3],
        'How often do you engage in relaxation techniques such as meditation or deep breathing?': [1],
        'How often do you took a break by going on a nature walk or any other nature related activity?': [1],
    }
    return pd.DataFrame(data)

def test_preprocess_data(sample_dataframe):
    df_processed = preprocess_data(sample_dataframe.copy())
    assert 'low' in df_processed.columns
    assert 'high' in df_processed.columns
    assert 'overall' in df_processed.columns
    assert df_processed['overall'].iloc[0] in [2, 3, 4, 5, 6, 7]

def test_train_knn_and_predict(sample_dataframe):
    df = preprocess_data(sample_dataframe.copy())
    model = train_knn(df)
    user_inputs = sample_dataframe.iloc[0].tolist()
    prediction = predict_stress(model, user_inputs)
    # Ensure we convert prediction to int if it's a numpy value
    prediction = int(prediction)
    assert prediction in [2, 3, 4, 5, 6, 7]
