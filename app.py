import streamlit as st
import pickle


# Load the saved VotingClassifier model
with open('voting_clas.pkl', 'rb') as f:
    model = pickle.load(f)

def classify(num):
    if num == 1:
        return 'Hazardous'
    else:
        return 'Not Hazardous'


def main():
    html_temp = """
<div style="
    background: linear-gradient(90deg, #d32f2f 0%, #b71c1c 100%);
    padding: 16px 10px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(180,40,40,0.15);">
    <h2 style="
        color: #fff;
        text-align: center;
        font-family: 'Segoe UI', Arial, sans-serif;
        letter-spacing: 1px;
        text-shadow: 1px 1px 8px #880808;
        margin: 0;">
        Hazardous Asteroid Prediction 🚀
    </h2>
</div>
"""


    st.markdown(html_temp, unsafe_allow_html=True)

    velocity = st.slider('Select Velocity (km/s)',0.437707, 33.429559, 4.953788)
    miss_distance = st.slider('Select Miss Distance (km)', 92277.000000, 74780120.0000000, 43523095.0000000)
    diameter_avg = st.slider('Select Average Diameter (km)', 0.001519, 0.627397, 0.283643)

    inputs = [[velocity, miss_distance, diameter_avg]]


    if st.button('Classify'):
        prediction = model.predict(inputs)[0]
        st.success(classify(prediction))

if __name__ == '__main__':
    main()