import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('cloudburst.csv')  # Replace with your actual dataset path

# Streamlit UI
st.title("Cloudburst Prediction System")

# Sidebar for input features
location = st.selectbox("Select Location", sorted(data['Location'].unique()))
terrain_height = st.slider("Terrain Height", min_value=int(data['Terrain_Height'].min()), max_value=int(data['Terrain_Height'].max()), value=50)
temperature = st.slider("Temperature", min_value=int(data['Temperature'].min()), max_value=int(data['Temperature'].max()), value=20)
humidity = st.slider("Humidity", min_value=int(data['Humidity'].min()), max_value=int(data['Humidity'].max()), value=60)
moisture = st.slider("Moisture", min_value=int(data['Moisture'].min()), max_value=int(data['Moisture'].max()), value=5)
air_pollution_index = st.slider("Air Pollution Index", min_value=int(data['Air_Pollution_Index'].min()), max_value=int(data['Air_Pollution_Index'].max()), value=40)
wind_speed = st.slider("Wind Speed", min_value=int(data['Wind_Speed'].min()), max_value=int(data['Wind_Speed'].max()), value=15)

# Prepare the input data
input_data = pd.DataFrame({
    'Location': [location],
    'Terrain_Height': [terrain_height],
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Moisture': [moisture],
    'Air_Pollution_Index': [air_pollution_index],
    'Wind_Speed': [wind_speed]
})

# Load your model (replace with your trained model)
model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(data[['Location', 'Terrain_Height', 'Temperature', 'Humidity', 'Moisture', 'Air_Pollution_Index', 'Wind_Speed']], data['Cloud_Burst_Event'], test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Predict
prediction = model.predict(input_data)

# Display prediction
st.write(f"Prediction: {'Cloudburst Likely' if prediction == 1 else 'No Cloudburst'}")
