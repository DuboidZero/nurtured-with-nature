import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Stress Quiz", layout="wide")
st.logo("sidebarlogo.png", size="large")

st.markdown("<h2 style='text-align:center;'>Stress Analysis Quiz:</h2>", unsafe_allow_html=True)
st.divider()

# 🎨 CSS for card-style layout
st.markdown("""
<style>
div[data-testid="stSlider"] {
    padding: 10px;
    margin-bottom: 20px;
    border: 1px solid #555;
    border-radius: 10px;
    background-color: #1e1e1e;
}
</style>
""", unsafe_allow_html=True)

# 🎯 Questions and sliders
with st.container():
    st.markdown("**1. How often have you felt anxious for situations which were not important/major?**")
    a = st.slider("On a scale from 1-5", 1, 5, key=1)

with st.container():
    st.markdown("**2. How difficult is it to overcome the situations where you are anxious?**")
    b = st.slider("On a scale from 1-5", 1, 5, key=2)

with st.container():
    st.markdown("**3. How often do you face trouble in sleeping?**")
    c = st.slider("On a scale from 1-5", 1, 5, key=3)

with st.container():
    st.markdown("**4. How often had you been idle/trying to sleep and have a chaotic mind full of thoughts?**")
    d = st.slider("On a scale from 1-5", 1, 5, key=4)

with st.container():
    st.markdown("**5. How often have you not felt confident in your ability to handle challenge related to work, family or physical pain/disease?**")
    e = st.slider("On a scale from 1-5", 1, 5, key=5)

with st.container():
    st.markdown("**6. How often do you take/consider painkillers, instead of resting, while facing mild pain?**")
    f = st.slider("On a scale from 1-5", 1, 5, key=6)

with st.container():
    st.markdown("**7. How often have you pushed yourself to be awake due to work or your lifestyle?**")
    g = st.slider("On a scale from 1-5", 1, 5, key=7)

with st.container():
    st.markdown("**8. How varying is your wake-up time of weekends or holidays compared to weekdays?**")
    h = st.slider("On a scale from 1-5", 1, 5, key=8)

with st.container():
    st.markdown("**9. How often do you engage in relaxation techniques such as meditation or deep breathing?**")
    j = st.slider("On a scale from 1-5", 1, 5, key=9)

with st.container():
    st.markdown("**10. How often do you take a break by going on a nature walk or any other nature-related activity?**")
    k = st.slider("On a scale from 1-5", 1, 5, key=10)

result = st.button("Go!")

# 💡 KEEPING PREDICTION CODE UNTOUCHED — just UI upgrade afterward
if result:
    dataframe = pd.read_excel('sample_data.xlsx')
    dataframe.rename(columns = {'How often have you felt anxious for situations which were not important/major?':'anxious'}, inplace = True)
    dataframe.rename(columns = {'How difficult is it to overcome the situations where you are anxious?':'overcome_anxious'}, inplace = True)
    dataframe.rename(columns = {'How often do you face trouble in sleeping?':'sleeping'}, inplace = True)
    dataframe.rename(columns = {'How often had you been idle/trying to sleep and having a chaotic mind full of thoughts?':'try_sleeping'}, inplace = True)
    dataframe.rename(columns = {'How often have you not felt confident in your ability to handle challenge related to work,family or physical pain/disease?':'noconfidence'}, inplace = True)
    dataframe.rename(columns = {'How often do you take/consider painkillers, instead of resting, while facing mild pain?':'painkillers'}, inplace = True)
    dataframe.rename(columns = {'How often have you pushed yourself to be awake due to work or your lifestyle?':'try2awake'}, inplace = True)
    dataframe.rename(columns = {'How varying is your wake-up time of weekends or holidays compared to weekdays?':'sleep_pattern'}, inplace = True)
    dataframe.rename(columns = {'How often do you engage in relaxation techniques such as meditation or deep breathing?':'relaxation'}, inplace = True)
    dataframe.rename(columns = {'How often do you took a break by going on a nature walk or any other nature related activity?':'nature_Activity'}, inplace = True)

    dataframe=pd.DataFrame(dataframe,columns=['Timestamp','anxious','overcome_anxious','sleeping','try_sleeping','noconfidence','painkillers','try2awake','sleep_pattern','relaxation','nature_Activity','low','high','overall'])
    dataframe['low']=dataframe['anxious']+dataframe['overcome_anxious']+dataframe['sleeping']+dataframe['try_sleeping']+dataframe['noconfidence']+dataframe['painkillers']+dataframe['try2awake']+dataframe['sleep_pattern']
    dataframe['high']=dataframe['relaxation']+dataframe['nature_Activity']         

    df1=pd.DataFrame(dataframe,columns=['low','high','overall'])

    lstress=[]
    for i in dataframe['low']:                                                 
        if i>=30 and i<=40:                                               
            lstress.append(3)                                                      
        elif i>=20 and i<=29:
            lstress.append(2)                                                      
        elif i>=1 and i<=19:
            lstress.append(1)                                                      
        else:
            lstress.append(4)
    dataframe['low']=lstress

    nstress=[]   
    for i in dataframe['high']:                                                  
        if i>=8 and i<=10:
            nstress.append(1)                                                    
        elif i>=6 and i<=7:
            nstress.append(2)                                                    
        elif i>=1 and i<=5:
            nstress.append(3)                                                    
        else:
            nstress.append(4)
    dataframe['high']=nstress
    dataframe['overall']=dataframe['low']+dataframe['high']

    x=dataframe.iloc[:,1:11].values
    y=dataframe['overall']
    classifier = KNeighborsClassifier(n_neighbors = 2, metric = 'minkowski', p = 2) 
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    classifier.fit(x,y)
    KNN = classifier.fit(x,y)
    test=pd.DataFrame()

    test['anxious']=[a]
    test['overcome_anxious']=[b]
    test['sleeping']=[c]
    test['try_sleeping']=[d]
    test['noconfidence']=[e]
    test['painkillers']=[f]
    test['try2awake']=[g]
    test['sleep_pattern']=[h]
    test['relaxation']=[j]
    test['nature_Activity']=[k]

    s2=KNN.predict(test)

    # 🎨 Color-coded output box
    if s2==1 or s2==2:
        result_text = "😁 You have low stress levels!"
        color = "#4CAF50"
    elif s2==3 or s2==4:
        result_text = "😐 You have moderate stress levels"
        color = "#FF9800"
    elif s2==5 or s2==6:
        result_text = "😥 You have high stress levels"
        color = "#F44336"
    else:
        result_text = "Please enter all of the values"
        color = "#777777"

    st.markdown(
        f"<div style='padding:20px; border-radius:10px; background-color:{color}; color:white; font-size:20px; text-align:center;'>{result_text}</div>",
        unsafe_allow_html=True
    )

st.divider()
st.image("2.png")
