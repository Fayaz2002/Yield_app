import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import base64

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Encode your local background image
main_bg_img = get_img_as_base64("sfield.jpg")  # Replace with your image file name

# Add this CSS for the main background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{main_bg_img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
# Initialize page state if it doesn't exist
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Load Saved Models
yearly_model = joblib.load('arimax_yearly_model.pkl')
user_input_model = joblib.load('arimax_yearly_model.pkl')
scaler = joblib.load('ENscaler.pkl')
lr_model = joblib.load('elastic_net_model.pkl')

# Define Features and Readable Names
features = ['TEMP', 'RAINFALL', 'SIZE', 'AUI', 'NITROGEN', 'PHOSPHORUS', 'POTASSIUM']
readable_features = {
    
    'TEMP': 'Temperature (in °C) ☀️',
    'RAINFALL': ' Rainfall (in mm) 🌧️',
    'SIZE': ' Size of Area (in million hectares) 🏞️',
    'AUI': 'Area Under Irrigation (in %) 🚰',
    'NITROGEN': ' Nitrogen Fertilizer Amount (in thousand tonnes) 🥬🧪',
    'PHOSPHORUS': 'Phosphorus Fertilizer Amount (in thousand tonnes) 🫚🧪 ',
    'POTASSIUM': ' Potassium Fertilizer Amount (in thousand tonnes) 🌱🧪',
}


# Minimum allowable values for features
min_values = {
    'TEMP': 22,
    'RAINFALL': 1000,
    'SIZE': 30,
    'AUI': 30,
    'NITROGEN': 500,
    'PHOSPHORUS': 500,
    'POTASSIUM': 500
}

# Helper Functions
def load_forecast_models(features):
    models = {}
    for feature in features:
        with open(f'{feature}_arima.pkl', 'rb') as f:
            models[feature] = joblib.load(f)
    return models

def forecast_features_with_saved_models(models, years=7):
    forecasted = {}
    for feature, model in models.items():
        forecasted[feature] = model.forecast(steps=years)
    return pd.DataFrame(forecasted)

def predict_future_yields(yearly_model, forecasted_features):
    return pd.Series(yearly_model.forecast(steps=len(forecasted_features), exog=forecasted_features))

def format_predictions_table(forecasted_features_df, future_predictions):
    results_df = forecasted_features_df.copy()
    results_df['Predicted Yield (kg/hectare)'] = future_predictions.values
    results_df['Year'] = range(2024, 2024 + len(future_predictions))
    cols_order = ['Year'] + [col for col in results_df.columns if col != 'Year']
    return results_df[cols_order]

def analyze_factor_impact(input_data, lr_model, scaler, features, readable_features):
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    contributions = input_scaled[0] * lr_model.coef_


    # Define feature categories
    climatic_features = ['TEMP', 'RAINFALL']
    agronomic_features = ['SIZE', 'AUI', 'NITROGEN', 'PHOSPHORUS', 'POTASSIUM']

    # Store impact analysis results
    climatic_impact = []
    agronomic_impact = []

    # Process individual climatic and agronomic factors
    for i, feature in enumerate(features):
        impact_type = "Climatic" if feature in climatic_features else "Agronomic"
        impact_text = {
            'feature': readable_features[feature],
            'contribution': contributions[i],
            'impact_text': f"{readable_features[feature]} has a <b style='color: {'lightgreen' if contributions[i] > 0 else 'red'};'>{'positive impact📈' if contributions[i] > 0 else 'negative impact📉'}</b> over the predicted yield."
        }

        if impact_type == "Climatic":
            climatic_impact.append(impact_text)
        else:
            agronomic_impact.append(impact_text)

    # Sort impacts by absolute contribution
    climatic_impact = sorted(climatic_impact, key=lambda x: -abs(x['contribution']))
    agronomic_impact = sorted(agronomic_impact, key=lambda x: -abs(x['contribution']))

    return climatic_impact, agronomic_impact


import numpy as np

def analyze_yearly_factor_impact(forecasted_features_df, lr_model, scaler, features, readable_features):
    # Scale the forecasted features
    forecasted_scaled = scaler.transform(forecasted_features_df)
    contributions = forecasted_scaled * lr_model.coef_

    # Sum contributions across the forecasted years
    total_contributions = contributions.sum(axis=0)


    # Define feature categories
    climatic_features = ['TEMP', 'RAINFALL']
    agronomic_features = ['SIZE', 'AUI', 'NITROGEN', 'PHOSPHORUS', 'POTASSIUM']

    # Store impact results
    climatic_impact = []
    agronomic_impact = []

    # Process individual climatic and agronomic factors
    for i, feature in enumerate(features):
        impact_type = "Climatic" if feature in climatic_features else "Agronomic"
        total_contribution = total_contributions[i]

        impact_text = {
            'feature': readable_features[feature],
            'total_contribution': total_contribution,
            'impact_text': f"{readable_features[feature]} has a <b style='color: {'lightgreen' if total_contribution > 0 else 'red'};'>{'positive impact📈' if total_contribution > 0 else 'negative impact📉'}</b> on yield over the projected years."
        }

        if impact_type == "Climatic":
            climatic_impact.append(impact_text)
        else:
            agronomic_impact.append(impact_text)


    # Sort impacts by absolute contribution
    climatic_impact = sorted(climatic_impact, key=lambda x: -abs(x['total_contribution']))
    agronomic_impact = sorted(agronomic_impact, key=lambda x: -abs(x['total_contribution']))

    return climatic_impact, agronomic_impact


# Load models and forecast data
forecast_models = load_forecast_models(features)
forecasted_features_df = forecast_features_with_saved_models(forecast_models)
future_predictions = predict_future_yields(yearly_model, forecasted_features_df)
results_table = format_predictions_table(forecasted_features_df, future_predictions)

# recommendations
def generate_recommendations(input_df, lr_model, scaler, features, readable_features):
    input_scaled = scaler.transform(input_df)  # Scale user input
    contributions = input_scaled[0] * lr_model.coef_  # Compute individual contributions

    recommendations = []

    for i, feature in enumerate(features):
        # For AUI (Area Under Irrigation)
        if feature == 'AUI':
            if contributions[i] > 0:
                recommendations.append(f"➕ Expanding {readable_features[feature]} has shown to improve yield. Consider optimizing irrigation strategies for better results.")
            else:
                recommendations.append(f"⬆️ Increasing {readable_features[feature]} may be necessary to counter potential yield reduction. Evaluate water management for efficiency.")

        # For SIZE (Cultivation Area)
        elif feature == 'SIZE':
            if contributions[i] > 0:
                recommendations.append(f"➕ Expanding {readable_features[feature]} could further enhance yield output. Assess feasibility for increasing land use.")
            else:
                recommendations.append(f"⬆️ Current {readable_features[feature]} may not be optimal for maximizing yield. Consider revising land utilization strategies.")

        # For NITROGEN, PHOSPHORUS, and POTASSIUM (Individual Fertilizer Impact)
        elif feature in ['NITROGEN', 'PHOSPHORUS', 'POTASSIUM']:
            if contributions[i] > 0:
                recommendations.append(f"➕ Increasing {readable_features[feature]} seems beneficial for yield.")
            else:
                recommendations.append(f"⬆️ Consider boosting {readable_features[feature]} to improve yield.")

    return recommendations


# Streamlit Pages
# Home Page
if st.session_state.page == 'Home':
    with st.container():
        st.markdown("""
            <style>
            /* Streamlit default font */
            body, h1, h2, p, ul, blockquote {
                font-family: "Source Sans Pro", sans-serif;
                line-height: 1.6;
            }
            .transparent-box {
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 20px;
                padding: 30px;
                margin: 30px 0;
                color: #333;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                position: relative;
            }
            .darker-box {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 45px;
                padding: 20px;
                margin: 20px 0;
                color: #333;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            }
            h1 {
                color: rgb(251, 193, 2);
                margin-bottom: 20px;
                text-align: center;
                font-family: 'Agency FB', sans-serif;
                font-size: 50px;
            }
            h2 {
                color:rgb(0, 8, 153);
                margin-bottom: 5px;
                font-family: 'Agency FB', sans-serif;
                font-size: 35px;
            }
            ul {
                color: #333333;
                padding-left: 20px;
                margin-bottom: 20px;
                font-size: 19px;
            }
            
            blockquote {
                font-style: italic;
                color: #555555;
                border-left: 4px solid #dddddd;
                margin: 10px 0;
                padding-left: 20px;
                font-size: 18px;
            }
            </style>
            <div class="transparent-box">
                <h1>Rice Yield Prediction App🌾</h1>
                <p style="text-align: center;font-size: 19px;">
                    This app helps you predict rice yield using advanced forecasting techniques. It considers climatic 
                    and agronomic factors such as temperature, rainfall, fertilizer usage, and irrigation. Explore future 
                    rice yield predictions or input your data for personalized analysis and actionable insights.
                </p>
                <div class="darker-box">
                    <h2>Key Statistics 📊</h2>
                    <ul>
                        <li> 🌾 India is the <span style="color: black;font-weight: bold">world's 🥈 second-largest producer of rice</span>, contributing significantly to global rice production.</li>
                        <li> 📦 India contributes approximately <span style="color: black;font-weight: bold">37% ⬆️ of the world's rice exports</span>.</li>
                        <li> 🍚 Rice is a staple food in India, <span style="color: black;font-weight: bold">feeding over 60% of the population 🌍</span>.</li>
                    </ul>
                    <h2>Quote ✍🏼</h2>
                    <blockquote>
                        <span style="color: black;font-weight: bold">" The ultimate goal of farming is not the growing of crops, but the cultivation and perfection of human beings." </span>
                        - <span style="color: brown;">Masanobu Fukuoka</span>
                    </blockquote>
                </div>
                <p style="text-align: center; font-weight: bold; color: red; font-size: 19px;">
                    ⏬ Click the below button for Future Prediction or Prediction by Input ⏬
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Creating two columns for the buttons
    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Future Rice Yield Predictions 🔮"):
                st.session_state.page = 'Future'
                st.experimental_rerun()

        # Adding space before the second button
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

        with col2:
            if st.button("Predict Rice Yield by Input 💻"):
                st.session_state.page = 'User Input'
                st.experimental_rerun()

# Future page
elif st.session_state.page == 'Future':
    # Darker Box Styling
    st.markdown(
        """
        <style>
        .darker-box {
            background-color: rgba(0, 0, 0, 0.9);
            border-radius: 15px;
            padding: 15px;
            margin-top: 20px;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        .darker-box h3 {
        color: #FFD700; /* Gold color for headings */
        font-size: 2.0 rem; /* Larger font size */
        font-weight: bold; /* Bold text */
    }
    * Climatic impact color */
.climatic-impact {
    color: #ADD8E6; /* Light blue */
}

/* Agronomic impact color */
.agronomic-impact {
    color:rgb(209, 254, 74); /* Light green */
}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Page Title
    # Use Markdown for styling the title
    st.markdown('<h1 style="color:rgb(207, 5, 5); text-align: center;font-size:35px; font-weight: bold;background-color:  rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); ">Rice Yield Predictions (2024–2030) 🔮</h1>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add line breaks

    # Plot Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(results_table['Year'], results_table['Predicted Yield (kg/hectare)'], marker='o', color='b')
    plt.title("Predicted Rice Yield (2024–2030)")
    plt.xlabel("Year")
    plt.ylabel("Predicted Yield (kg/hectare)")
    plt.grid()
    st.pyplot(plt)

    # Show DataFrame
    st.markdown('<h3 style="color:rgb(0, 0, 0,); text-align: center; font-weight: bold; background-color:  rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5); ">Predicted Yield and Forecasted Factors </h3>', unsafe_allow_html=True)
    st.dataframe(
        results_table.reset_index(drop=True).style.format({
            "Year": "{:.0f}",
            "TEMP": "{:.2f}",
            "RAINFALL": "{:.2f}",
            "SIZE": "{:.2f}",
            "AUI": "{:.2f}",
            "NITROGEN": "{:.2f}",
            "PHOSPHORUS": "{:.2f}",
            "POTASSIUM": "{:.2f}",
            "Predicted Yield (kg/hectare)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

        # Darker Box: Impact Analysis
    climatic_impact, agronomic_impact = analyze_yearly_factor_impact(
        forecasted_features_df, lr_model, scaler, features, readable_features
    )

    # Separate agronomic impact into **individual** and **combined fertilizers**
    individual_agronomic = [impact for impact in agronomic_impact if impact['feature'] in readable_features.values()]
    combined_fertilizer_impact = [impact for impact in agronomic_impact if impact['feature'] not in readable_features.values()]

    # Generate Climatic Impact HTML
    impact_analysis_html = "<h4 class='climatic-impact'>Climatic Impact ⛅:</h4>"
    impact_analysis_html += "".join(f"<li>{impact['impact_text']}</li>" for impact in climatic_impact)

    # Generate Agronomic Impact HTML (Individual Factors)
    impact_analysis_html += "<h4 class='agronomic-impact'>Agronomic Impact 🌽:</h4>"
    impact_analysis_html += "".join(f"<li>{impact['impact_text']}</li>" for impact in individual_agronomic)


    # Display Impact Analysis
    darker_box_html = f"""
    <div class="darker-box">
        <h3>Impact Analysis of Climatic and Agronomic Factors 💯</h3>
        <ul>
            {impact_analysis_html}
        </ul>
    </div>
    """
    st.markdown(darker_box_html, unsafe_allow_html=True)


    with st.container():
     st.markdown('<div class="back-button"></div>', unsafe_allow_html=True)
    if st.button("Back to Home 🏠"):
        st.session_state.page = 'Home'
        st.experimental_rerun()

# CSS for custom containers and button alignment
# User Input Page
custom_css = """
<style>
/* Dark background for Impact Analysis container */
.impact-container {
    background-color: rgba(0, 0, 0, 0.9);
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    color: white;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    font-size: 1.1rem;
}

/* Climatic impact color */
.climatic-impact {
    color: #ADD8E6; /* Light blue */
}

/* Agronomic impact color */
.agronomic-impact {
    color:rgb(209, 254, 74); /* Light green */
}

/* Transparent white background for Recommended Actions container */
.recommendation-container {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    margin-bottom: 20px; /* Add spacing below the container */
    color: #333;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    font-size: 1.1rem;
}

/* Title color for Recommended Actions */
.recommendation-title {
    color: #FF0000; /* Red */
    font-weight: bold;
}

/* Title color for Impact Analysis */
.impact-title {
    color: #FFD700; /* Gold */
    font-weight: bold;
}

/* Green transparent background for Predicted Yield */
.predicted-yield-box {
    background-color: rgba(0, 255, 0, 0.4);
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    color: #333;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    font-weight: bold;
    text-align: center;
    font-size: 1.2rem;
}

/* Align the back button to the right */
.back-button-container {
    text-align: right;
    margin-top: 20px;
}
</style>
"""

# Add CSS to the app
st.markdown(custom_css, unsafe_allow_html=True)

if st.session_state.page == 'User Input':
    st.markdown('<h1 style="color:rgb(17, 8, 103); text-align: center;font-size:40px; font-weight: bold;background-color:  rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-bottom: 20px; ">Predict Rice Yield Through Input 👨🏽‍💻</h1>', unsafe_allow_html=True)

    # Ensure session state exists
    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = {feature: None for feature in features}

    input_data = []
    with st.form("user_input_form"):
        for feature in features:
            min_value = min_values[feature]
            
            # Feature label
            st.markdown(f"""
            <p style="
                font-size: 20px; 
                font-weight: bold; 
                margin-bottom: 1px;
                padding: 0px;
            ">
                Enter {readable_features[feature]}:
            </p>
            """, unsafe_allow_html=True)

            # Prefill with previous input
            value = st.number_input(
                "", 
                value=st.session_state.user_inputs.get(feature, None),  
                step=1.0,
                format="%.2f",
                key=feature
            )

            # Store input in session state
            st.session_state.user_inputs[feature] = value

            # Validation
            if value is not None and value < min_value:
                st.error(f"The value for {readable_features[feature]} must be at least {min_value}.")

            input_data.append(value if value is not None else 0.0)

        submitted = st.form_submit_button("Predict Rice Yield 🌾")

    if submitted:
        if all(value >= min_values[feature] for feature, value in zip(features, input_data)):
            input_df = pd.DataFrame([input_data], columns=features)
            pred_yield = user_input_model.forecast(steps=1, exog=input_df).iloc[0]

            # Predicted Yield
            st.markdown(f"<div class='predicted-yield-box'>Predicted Yield: {pred_yield:.2f} kg/hectare</div>", unsafe_allow_html=True)

            # Impact Analysis
            climatic_impact, agronomic_impact = analyze_factor_impact(input_df, lr_model, scaler, features, readable_features)
            individual_agronomic = [impact for impact in agronomic_impact if impact['feature'] in readable_features.values()]
            combined_fertilizer_impact = [impact for impact in agronomic_impact if impact['feature'] not in readable_features.values()]

            impact_html = "<h4 class='climatic-impact'>Climatic Impact ⛅:</h4>"
            impact_html += "".join(f"<li>{impact['impact_text']}</li>" for impact in climatic_impact)
            impact_html += "<h4 class='agronomic-impact'>Agronomic Impact 🌽:</h4>"
            impact_html += "".join(f"<li>{impact['impact_text']}</li>" for impact in individual_agronomic)

            st.markdown(f"""
            <div class='impact-container'>
                <h3 class='impact-title'>Impact Analysis of Input Factors 💯</h3>
                <ul>{impact_html}</ul>
            </div>
            """, unsafe_allow_html=True)

            recommendations = generate_recommendations(input_df, lr_model, scaler, features, readable_features)
            recommendations_html = "".join(f"<li>{rec}</li>" for rec in recommendations)
            st.markdown(f"""
            <div class='recommendation-container'>
                <h3 class='recommendation-title'>Recommended Actions 💡</h3>
                <ul>{recommendations_html}</ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Please correct the inputs and ensure all values meet the minimum requirements.")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Back to Home 🏠"):
            st.session_state.page = 'Home'
            st.experimental_rerun()
