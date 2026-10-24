import streamlit as st
import pandas as pd
import os
import joblib

# Page configuration
st.set_page_config(page_title="ML Data Drift Monitor", page_icon="🔍", layout="wide")

st.title("🔍 Real-Time ML Data Drift & Observability Dashboard")
st.markdown("""
This dashboard monitors production traffic against our baseline model to detect **Data Drift** 
caused by real-world economic or behavioral shifts.
""")

# Sidebar controls
st.sidebar.header("Control Panel")
app_mode = st.sidebar.selectbox("Choose Mode", ["📊 Live Drift Report", "🔮 Test Model Prediction"])

# Load Reference Data & Model safely
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl') if os.path.exists('model.pkl') else None
    ref_data = pd.read_csv('reference_data.csv') if os.path.exists('reference_data.csv') else None
    return model, ref_data

model, reference_data = load_artifacts()

if app_mode == "📊 Live Drift Report":
    st.header("Automated Data Drift Audit")
    
    data_choice = st.sidebar.radio("Select Production Batch to Audit:", ["Drifted Production Data", "Normal Production Data"])
    
    if data_choice == "Drifted Production Data":
        target_file = 'drifted_data.csv'
        st.warning("⚠️ Currently viewing **Drifted Data** batch (Simulating inflation/market shifts).")
    else:
        target_file = 'reference_data.csv'
        st.success("✅ Currently viewing **Normal Data** batch (Matches training baseline).")
        
    if os.path.exists(target_file):
        current_data = pd.read_csv(target_file)
        
        with st.expander("Preview Incoming Production Data"):
            st.dataframe(current_data.head(10))
            
        st.markdown("---")
        st.subheader("Data Drift Analysis Report")
        
        # Render the custom HTML report inside Streamlit
        if os.path.exists('drift_report.html'):
            with open('drift_report.html', 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
        else:
            st.error("Drift HTML report not found.")
    else:
        st.error(f"Could not find {target_file}.")

elif app_mode == "🔮 Test Model Prediction":
    st.header("Test the Deployed Model Live")
    st.markdown("Adjust feature values below to see how the Random Forest model predicts house values.")
    
    if model is not None and reference_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            med_inc = st.slider("Median Income (MedInc)", float(reference_data['MedInc'].min()), float(reference_data['MedInc'].max() * 2), float(reference_data['MedInc'].mean()))
            ave_rooms = st.slider("Average Rooms (AveRooms)", float(reference_data['AveRooms'].min()), float(reference_data['AveRooms'].max() * 2), float(reference_data['AveRooms'].mean()))
            
        with col2:
            ave_occup = st.slider("Average Occupancy (AveOccup)", float(reference_data['AveOccup'].min()), float(reference_data['AveOccup'].max() * 2), float(reference_data['AveOccup'].mean()))
            house_age = st.slider("House Age (HouseAge)", float(reference_data['HouseAge'].min()), float(reference_data['HouseAge'].max()), float(reference_data['HouseAge'].mean()))
            
        # Raw values array to bypass name/order validation errors completely
        input_values = [[
            med_inc, 
            house_age, 
            ave_rooms, 
            reference_data['AveBedrms'].mean(), 
            ave_occup, 
            reference_data['Population'].mean(), 
            reference_data['Latitude'].mean(), 
            reference_data['Longitude'].mean()
        ]]
        
        if st.button("Predict House Value"):
            prediction = model.predict(input_values)[0]
            st.success(f"🏡 Predicted Median House Value: **${prediction * 100000:,.2f}**")
    else:
        st.error("Model artifacts not loaded properly.")