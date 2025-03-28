import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("f1_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load encoders for Driver and Team
with open("driver_encoder.pkl", "rb") as file:
    le_driver = pickle.load(file)

with open("team_encoder.pkl", "rb") as file:
    le_team = pickle.load(file)

# Streamlit UI
st.title("F1 Qualifying Time Predictor")
st.write("Predict a driver's qualifying lap time based on practice session data.")

# User inputs
driver = st.selectbox("Select Driver", le_driver.classes_)  # Ensure the list matches the trained encoder
team = st.selectbox("Select Team", le_team.classes_)  # Ensure the list matches the trained encoder
sector_1 = st.number_input("Sector1 Time (s)", min_value=10.0, max_value=40.0, step=0.1)
sector_2 = st.number_input("Sector2 Time (s)", min_value=10.0, max_value=40.0, step=0.1)
sector_3 = st.number_input("Sector3 Time (s)", min_value=10.0, max_value=40.0, step=0.1)

# Prediction button
if st.button("Predict Lap Time"):
    # Convert categorical inputs using the same encoding as training
    driver_encoded = le_driver.transform([driver])[0]
    team_encoded = le_team.transform([team])[0]

    # Create input dataframe with correct feature order
    input_data = pd.DataFrame([[driver_encoded, sector_1, sector_2, sector_3, team_encoded]], 
                              columns=["Driver", "Sector1", "Sector2", "Sector3", "Team"])

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Qualifying Lap Time: {prediction:.3f} seconds")
