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
