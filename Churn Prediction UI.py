import streamlit as st
import pandas as pd
import numpy as np
import pickle # Changed from joblib to pickle
import os

# --- Helper function to load the model ---
@st.cache_resource # Cache the model loading for efficiency
def load_model(model_name):
    """Loads a trained churn prediction pipeline."""
    filename = f"{model_name.lower().replace(' ', '_')}_churn_model.pkl" # Changed extension to .pkl
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f: # Use 'rb' for binary read mode
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Error loading {model_name} model: {e}")
            return None
    else:
        st.warning(f"Model file '{filename}' not found. Please run the training script first to save the models.")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Spotify Churn Prediction", layout="centered")

st.title("🎧 Spotify Churn Prediction App")
st.markdown("""
This application predicts whether a Spotify user is likely to churn based on their demographic and usage patterns.
Please input the user's details below.
""")

# --- Load models (assuming they are saved in the same directory) ---
# You need to run the previous Python script once to generate these files.
dt_model = load_model('Decision Tree')
rf_model = load_model('Random Forest')
xgb_model = load_model('XGBoost')

# Check if models are loaded
if dt_model is None and rf_model is None and xgb_model is None:
    st.error("No models could be loaded. Please ensure the training script has been run and model files are present.")
    st.stop() # Stop the app if no models are available

# --- Model Selection ---
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a Prediction Model:",
    ['Decision Tree', 'Random Forest', 'XGBoost']
)

model = None
if selected_model_name == 'Decision Tree':
    model = dt_model
elif selected_model_name == 'Random Forest':
    model = rf_model
elif selected_model_name == 'XGBoost':
    model = xgb_model

if model is None:
    st.error(f"Selected model '{selected_model_name}' could not be loaded. Please check the console for errors.")
    st.stop()


st.header("User Information")

# --- Input Fields for Features ---

# Age
age_options = ['12-20', '20-35', '35-60', '60+']
age = st.selectbox("Age Group:", age_options)

# Gender
gender = st.radio("Gender:", ['Male', 'Female', 'Others'])

# Spotify Usage Period
spotify_usage_period = st.selectbox("Spotify Usage Period:",
                                     ['Less than 6 months', '6 months to 1 year', '1 year to 2 years', 'More than 2 years'])

# Spotify Listening Device (Multi-select)
spotify_listening_device_options = [
    'Smartphone', 'Computer or laptop', 'Smart speakers or voice assistants', 'Wearable devices'
]
spotify_listening_device = st.multiselect("Spotify Listening Device(s):", spotify_listening_device_options)
# Convert list of selected devices to comma-separated string for consistency with training data
spotify_listening_device_str = ", ".join(spotify_listening_device) if spotify_listening_device else ""


# Spotify Subscription Plan
spotify_subscription_plan = st.selectbox("Spotify Subscription Plan:",
                                         ['Free (ad-supported)', 'Premium (paid subscription)'])

# Premium Subscription Willingness
premium_sub_willingness = st.radio("Willingness to Pay for Premium:", ['Yes', 'No'])

# Preferred Premium Plan
preffered_premium_plan_options = [
    'None', 'Student Plan-Rs 59/month', 'Individual Plan- Rs 119/ month',
    'Duo plan- Rs 149/month', 'Family Plan-Rs 179/month'
]
preffered_premium_plan = st.selectbox("Preferred Premium Plan (if willing to pay):", preffered_premium_plan_options)

# Preferred Listening Content
preferred_listening_content = st.selectbox("Preferred Listening Content:", ['Music', 'Podcast'])

# Favorite Music Genre
fav_music_genre_options = [
    'Melody', 'Pop', 'Rap', 'classical', 'Rock', 'Electronic/Dance', 'Old songs',
    'trending songs random', 'All', 'Kpop', 'None' # 'None' for podcast listeners
]
fav_music_genre = st.selectbox("Favorite Music Genre:", fav_music_genre_options)

# Music Time Slot
music_time_slot_options = ['Morning', 'Afternoon', 'Night']
music_time_slot = st.selectbox("Preferred Music Listening Time Slot:", music_time_slot_options)

# Music Influential Mood (Multi-select)
music_Influencial_mood_options = [
    'Relaxation and stress relief', 'Uplifting and motivational', 'Sadness or melancholy',
    'Social gatherings or parties'
]
music_Influencial_mood = st.multiselect("Music Influential Mood(s):", music_Influencial_mood_options)
music_Influencial_mood_str = ", ".join(music_Influencial_mood) if music_Influencial_mood else ""


# Music Listening Frequency
music_lis_frequency_options = [
    'Daily', 'Several times a week', 'Once a week', 'Rarely', 'Never'
]
music_lis_frequency = st.selectbox("Music Listening Frequency:", music_lis_frequency_options)

# Music Exploration Method (Multi-select)
music_expl_method_options = [
    'recommendations', 'Playlists', 'Radio', 'Others', 'Social media', 'Friends', 'Search', 'Random'
]
music_expl_method = st.multiselect("Music Exploration Method(s):", music_expl_method_options)
music_expl_method_str = ", ".join(music_expl_method) if music_expl_method else ""


# Music Recommendation Rating (1-5)
music_recc_rating = st.slider("Music Recommendation Rating (1-5, 1=Very Dissatisfied, 5=Very Satisfied):", 1, 5, 3)

# Podcast Listening Frequency
pod_lis_frequency_options = [
    'Daily', 'Several times a week', 'Once a week', 'Rarely', 'Never'
]
pod_lis_frequency = st.selectbox("Podcast Listening Frequency:", pod_lis_frequency_options)

# Favorite Podcast Genre
fav_pod_genre_options = [
    'Comedy', 'Lifestyle and Health', 'Sports', 'Food and cooking', 'Health and Fitness',
    'Business', 'Informative stuff', 'Technology', 'Spiritual and devotional',
    'General knowledge', 'Murder Mystery', 'Educational', 'Stories',
    'Political, informative, topics that interests me', 'Dance and Relevant cases', 'Novels', 'Everything', 'None'
]
fav_pod_genre = st.selectbox("Favorite Podcast Genre:", fav_pod_genre_options)

# Preferred Podcast Format
preffered_pod_format_options = [
    'Interview', 'Story telling', 'Conversational', 'Educational', 'None'
]
preffered_pod_format = st.selectbox("Preferred Podcast Format:", preffered_pod_format_options)

# Podcast Host Preference
pod_host_preference_options = [
    'Both', 'Well known individuals', 'unknown Podcasters', 'None'
]
pod_host_preference = st.selectbox("Podcast Host Preference:", pod_host_preference_options)

# Preferred Podcast Duration
preffered_pod_duration_options = ['Shorter', 'Longer', 'Both', 'None']
preffered_pod_duration = st.selectbox("Preferred Podcast Duration:", preffered_pod_duration_options)

# Podcast Variety Satisfaction
pod_variety_satisfaction_options = ['Very Satisfied', 'Satisfied', 'Ok', 'Dissatisfied', 'Very Dissatisfied', 'None']
pod_variety_satisfaction = st.selectbox("Podcast Variety Satisfaction:", pod_variety_satisfaction_options)


# --- Prediction Button ---
if st.button("Predict Churn"):
    if model is not None:
        # Create a DataFrame from the input features
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'spotify_usage_period': spotify_usage_period,
            'spotify_listening_device': spotify_listening_device_str,
            'spotify_subscription_plan': spotify_subscription_plan,
            'premium_sub_willingness': premium_sub_willingness,
            'preffered_premium_plan': preffered_premium_plan,
            'preferred_listening_content': preferred_listening_content,
            'fav_music_genre': fav_music_genre,
            'music_time_slot': music_time_slot,
            'music_Influencial_mood': music_Influencial_mood_str,
            'music_lis_frequency': music_lis_frequency,
            'music_expl_method': music_expl_method_str,
            'music_recc_rating': music_recc_rating,
            'pod_lis_frequency': pod_lis_frequency,
            'fav_pod_genre': fav_pod_genre,
            'preffered_pod_format': preffered_pod_format,
            'pod_host_preference': pod_host_preference,
            'preffered_pod_duration': preffered_pod_duration,
            'pod_variety_satisfaction': pod_variety_satisfaction
        }])

        # Apply the same age conversion as in training
        def convert_age_to_numeric_for_prediction(age_range):
            if isinstance(age_range, str):
                if '-' in age_range:
                    parts = age_range.split('-')
                    return (int(parts[0]) + int(parts[1])) / 2
                elif '+' in age_range:
                    return int(age_range.replace('+', '')) + 5
            return np.nan # Should not happen with selectbox, but for safety
        input_data['Age_Numeric'] = input_data['Age'].apply(convert_age_to_numeric_for_prediction)
        # Drop the original 'Age' column as the model expects 'Age_Numeric'
        input_data = input_data.drop('Age', axis=1)

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"⚠️ This user is likely to CHURN!")
            st.write(f"Probability of Churn: {prediction_proba[1]:.2f}")
        else:
            st.success(f"✅ This user is likely NOT to churn.")
            st.write(f"Probability of No Churn: {prediction_proba[0]:.2f}")
    else:
        st.warning("Please ensure a model is selected and loaded successfully.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Scikit-learn.")