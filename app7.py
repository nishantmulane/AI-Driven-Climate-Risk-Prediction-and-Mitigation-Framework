import streamlit as st
import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load the trained model
model = load_model("first.h5")

# Load and preprocess the dataset
@st.cache_data
def load_new_data():
    df = pd.read_csv("2324final.csv", parse_dates=["Datetime"], index_col="Datetime")
    return df

# Prepare the data for prediction
def prepare_data(df, columns_to_predict, scaler, n_steps):
    df = df[columns_to_predict].dropna()
    scaled_data = scaler.transform(df.values)
    X = []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i + n_steps])
    X = np.array(X)
    prediction_dates = df.index[n_steps:]
    return X, prediction_dates

# Classify climate risks and suggest mitigations
# Classify climate risks and suggest mitigations
def classify_risks_and_mitigations(row):
    heatwave_risk = "High" if row["TempC"] > 40 else "Medium" if row["TempC"] > 30 else "Low"
    coldwave_risk = "High" if row["TempC"] < 12 else "Medium" if row["TempC"] <= 18 else "Low"
    flood_risk = "High" if row["PrecipMM"] > 5 else "Medium" if row["PrecipMM"] >= 2 else "Low"
    storm_risk = "High" if row["WindspeedKmph"] > 30 else "Medium" if row["WindspeedKmph"] >= 20 else "Low"
    
    strategies = {}
    if heatwave_risk in ['High', 'Medium']:
        strategies['Heatwave'] = ["Stay hydrated by drinking plenty of water", "Use air conditioning or fans to stay cool indoors", "Avoid outdoor activities, especially during peak heat hours"]
    if coldwave_risk in ['High', 'Medium']:
        strategies['Coldwave'] = ["Wear warm clothing, including layers, hats, and gloves", "Use heaters or other heating devices to maintain warmth indoors", "Minimize exposure to outdoor cold weather conditions"]
    if flood_risk in ['High', 'Medium']:
        strategies['Flood'] = ["Avoid low-lying or flood-prone areas during heavy rainfall", "Prepare emergency kits with essentials like food and water", "Stay updated with flood alerts and follow safety instructions"]
    if storm_risk in ['High', 'Medium']:
        strategies['Storm'] = ["Secure outdoor items like furniture and quipment to prevent damage", "Stay indoors and away from windows during storms", "Avoid unnecessary travel until the storm subsides"]
    
    return {
        "Heatwave Risk": heatwave_risk,
        "Coldwave Risk": coldwave_risk,
        "Flood Risk": flood_risk,
        "Storm Risk": storm_risk,
        "Mitigation Strategies": strategies
    }

# Update to improve presentation of mitigation strategies
def display_mitigation_strategies(strategies):
    if not strategies:
        st.markdown(
            "<h4 style='color: green;'>🌿 No immediate risks detected! Enjoy a safe day!</h4>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<h4 style='color: red;'>⚠️ Climate Risks Detected! Suggested Mitigation Strategies:</h4>", unsafe_allow_html=True)
        for risk, actions in strategies.items():
            st.markdown(f"**{risk} Risk**:")
            st.markdown("<ul style='color: white;'>", unsafe_allow_html=True)
            for action in actions:
                st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

# Predict the next 24 hours
def predict_next_24_hours(model, scaler, X_new, prediction_dates, selected_datetime, n_steps, columns_to_predict):
    selected_index = prediction_dates.get_loc(selected_datetime)
    input_sequence = X_new[selected_index].reshape(1, n_steps, len(columns_to_predict))
    prediction_sequence = input_sequence
    predicted_values_list = []
    
    for _ in range(24):
        predicted_values = model.predict(prediction_sequence)
        predicted_values_original = scaler.inverse_transform(predicted_values)
        predicted_values_list.append(np.rint(predicted_values_original[0]).astype(int))  # Convert to integers
        predicted_values_reshaped = predicted_values.reshape(1, 1, len(columns_to_predict))
        prediction_sequence = np.append(prediction_sequence[:, 1:, :], predicted_values_reshaped, axis=1)

    predicted_df = pd.DataFrame(
        predicted_values_list, 
        columns=columns_to_predict, 
        index=pd.date_range(selected_datetime, periods=24, freq="H")
    )
    return predicted_df

# Streamlit app
def main():
    st.set_page_config(page_title="🌤 Climate Risk Dashboard", layout="wide")
    st.markdown(
        """
        <style>
            body {background-color: #2c2f38; color: white;}
            h1, h2, h3, h4, h5, h6 {color: #f9f9f9;}
            .stButton button {background-color: #4CAF50; color: white; border: none; border-radius: 5px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("🌤 Climate Risk Prediction Dashboard")
    st.subheader("Hourly Weather Forecast and Climate Risk Classification")

    # Load data
    new_df = load_new_data()
    columns_to_predict = ["MaxTempC", "MinTempC", "AvgTempC", "TempC", "WindspeedKmph", "PrecipMM", "Humidity"]
    scaler = StandardScaler()
    scaler.fit(new_df[columns_to_predict].dropna())
    n_steps = 24

    # Input selection
    try:
        X_new, prediction_dates = prepare_data(new_df, columns_to_predict, scaler, n_steps)
    except ValueError as e:
        st.error(f"Data preparation error: {e}")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input(
            "Select a date for prediction:", 
            min_value=prediction_dates.min().date(), 
            max_value=prediction_dates.max().date()
        )
    with col2:
        selected_hour = st.selectbox("Select an hour for prediction:", [f"{i:02d}:00" for i in range(24)])
    
    selected_datetime = pd.to_datetime(f"{selected_date} {selected_hour}")

    if selected_datetime not in prediction_dates:
        st.error("Invalid date and time selected for prediction.")
        return

    # Predict and display results
    if st.button("🌟 Predict Weather"):
        predicted_df = predict_next_24_hours(
            model, scaler, X_new, prediction_dates, selected_datetime, n_steps, columns_to_predict
        )

        # Display parameters for the selected date and time
        selected_params = predicted_df.loc[selected_datetime]
        st.markdown(f"### 🌟 Weather Parameters for {selected_datetime.strftime('%A, %d %B %Y, %H:%M')}")
        st.write(selected_params.to_frame(name="Value").rename_axis("Parameter"))
        
        # Risk classification for the selected date and time
        selected_risk = classify_risks_and_mitigations(selected_params)
        display_mitigation_strategies(selected_risk["Mitigation Strategies"])

        # Hourly forecast visualization
        st.subheader(f"🌡️ Weather Forecast: {selected_datetime.strftime('%A, %d %B %Y')}")
        
        st.markdown("### 🌡️ Temperature (°C)")
  
        temp_fig = px.line(predicted_df, x=predicted_df.index, y="TempC", title="Hourly Temperature")
        temp_fig.update_layout(template="plotly_dark", title_font_size=20)
        st.plotly_chart(temp_fig, use_container_width=True)

        st.markdown("### 🌧️ Precipitation (mm)")
        precip_fig = px.line(predicted_df, x=predicted_df.index, y="PrecipMM", title="Hourly Precipitation")
        precip_fig.update_layout(template="plotly_dark", title_font_size=20)
        st.plotly_chart(precip_fig, use_container_width=True)

        st.markdown("### 🌬️ Windspeed (km/h)")
        wind_fig = px.line(predicted_df, x=predicted_df.index, y="WindspeedKmph", title="Hourly Windspeed")
        wind_fig.update_layout(template="plotly_dark", title_font_size=20)
        st.plotly_chart(wind_fig, use_container_width=True)

        # Display hourly predictions in a table
        st.subheader("📋 Detailed Predictions")
        st.dataframe(predicted_df)

if __name__ == "__main__":
    main()