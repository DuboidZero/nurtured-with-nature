# test_project.py
import pytest
import pandas as pd
from predictor import preprocess_data, train_knn, predict_stress

@pytest.fixture
def sample_data():
    # Create a small sample dataframe matching the expected input structure
    data = {
        'How often have you felt anxious for situations which were not important/major?': [5],
        'How difficult is it to overcome the situations where you are anxious?': [4],
        'How often do you face trouble in sleeping?': [3],
        'How often had you been idle/trying to sleep and having a chaotic mind full of thoughts?': [3],
        'How often have you not felt confident in your ability to handle challenge related to work,family or physical pain/disease?': [4],
        'How often do you take/consider painkillers, instead of resting, while facing mild pain?': [2],
        'How often have you pushed yourself to be awake due to work or your lifestyle?': [4],
        'How varying is your wake-up time of weekends or holidays compared to weekdays?': [3],
        'How often do you engage in relaxation techniques such as meditation or deep breathing?': [2],
        'How often do you took a break by going on a nature walk or any other nature related activity?': [2],
    }
    return pd.DataFrame(data)

def test_preprocess_data(sample_data):
    processed = preprocess_data(sample_data.copy())
    assert 'overall' in processed.columns
    assert processed['low'].iloc[0] in [1, 2, 3, 4]
    assert processed['high'].iloc[0] in [1, 2, 3, 4]

def test_train_knn(sample_data):
    df = preprocess_data(sample_data.copy())
    model = train_knn(df)
    assert hasattr(model, "predict")

def test_predict_stress(sample_data):
    df = preprocess_data(sample_data.copy())
    model = train_knn(df)
    user_input = df.iloc[0, :10].tolist()
    result = predict_stress(model, user_input)
    assert isinstance(result, int)
    assert result in range(2, 8)  # expected overall range
