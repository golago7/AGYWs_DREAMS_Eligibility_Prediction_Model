import streamlit as st
import pandas as pd
import numpy as np
import sklearn
#from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the trained model (replace 'rf_model.pkl' with your model file)
model = joblib.load('vs_xgb_model.pkl')

# Define the feature columns expected by the model
expected_features = [
    'age_at_screening','marital_status',
    'has_disability', 'out_of_school', 'ever_had_sex', 'is_head',
    'undergone_gbv_last_12mnths', 'sexual_partners_last_12mnths',
    'received_gifts_for_sex', 'ever_had_sti', 'no_condom_use',
    'is_orphan', 'has_child', 'used_drugs_last_12mnths'
]

def process_input_data(age, marital_status, has_disability, out_of_school, ever_had_sex, 
                       is_head, undergone_gbv_last_12mnths, sexual_partners_last_12mnths,
                       received_gifts_for_sex, ever_had_sti, no_condom_use, 
                       is_orphan, has_child, used_drugs_last_12mnths):
    # Create DataFrame
    input_data = pd.DataFrame({
        'age_at_screening': [age],
        'marital_status': [marital_status],
        'has_disability': [has_disability],
        'out_of_school': [out_of_school],
        'ever_had_sex': [ever_had_sex],
        'is_head': [is_head],
        'undergone_gbv_last_12mnths': [undergone_gbv_last_12mnths],
        'sexual_partners_last_12mnths': [sexual_partners_last_12mnths],
        'received_gifts_for_sex': [received_gifts_for_sex],
        'ever_had_sti': [ever_had_sti],
        'no_condom_use': [no_condom_use],
        'is_orphan': [is_orphan],
        'has_child': [has_child],
        'used_drugs_last_12mnths': [used_drugs_last_12mnths]
    })

    # Convert binary attributes from 'Yes'/'No' to 1/0
    binary_attributes = [
        'has_disability', 'out_of_school', 'ever_had_sex', 'is_head', 
        'undergone_gbv_last_12mnths', 'sexual_partners_last_12mnths', 
        'received_gifts_for_sex', 'ever_had_sti', 'no_condom_use', 
        'is_orphan', 'has_child', 'used_drugs_last_12mnths'
    ]

    for attr in binary_attributes:
        input_data[attr] = input_data[attr].map({'Yes': 1, 'No': 0})

    # One-hot encode the 'marital_status' column
    input_data = pd.get_dummies(input_data, columns=['marital_status'], prefix='status')

    # Ensure all expected features are present
    aligned_data = pd.DataFrame(
        {feature: input_data.get(feature, [0]) for feature in expected_features}
    )

    # Convert to NumPy array for prediction
    return aligned_data.values

def main():
    # Streamlit UI
    st.title("AGYW Vulnerability Screening")

    # Age input
    age_at_screening = st.text_input("AGYW Age at Screening")

    # Validate age input
    try:
        age_at_screening = int(age_at_screening)
        if age_at_screening < 10 or age_at_screening > 24:
            st.error("Please enter an age between 10 and 24.")
            st.stop()
    except ValueError:
        st.error("Please enter a valid number for age.")
        st.stop()

    # Collect other inputs
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widowed", "Cohabiting"])
    has_disability = st.selectbox("Has Disability", ["Yes", "No"])
    out_of_school = st.selectbox("Out of School", ["Yes", "No"])
    ever_had_sex = st.selectbox("Ever Had Sex", ["Yes", "No"])
    is_head = st.selectbox("Is Head", ["Yes", "No"])
    undergone_gbv_last_12mnths = st.selectbox("Undergone GBV Last 12 Months", ["Yes", "No"])
    sexual_partners_last_12mnths = st.selectbox("Sexual Partners Last 12 Months", ["Yes", "No"])
    received_gifts_for_sex = st.selectbox("Received Gifts for Sex", ["Yes", "No"])
    ever_had_sti = st.selectbox("Ever Had STI", ["Yes", "No"])
    no_condom_use = st.selectbox("No Condom Use", ["Yes", "No"])
    is_orphan = st.selectbox("Is Orphan", ["Yes", "No"])
    has_child = st.selectbox("Has Child", ["Yes", "No"])
    used_drugs_last_12mnths = st.selectbox("Used Drugs Last 12 Months", ["Yes", "No"])

    # Make prediction when button is clicked
    if st.button("Check Eligibility"):
        input_data_as_numpy_array = process_input_data(
            age_at_screening, marital_status, has_disability, out_of_school, ever_had_sex, 
            is_head, undergone_gbv_last_12mnths, sexual_partners_last_12mnths,
            received_gifts_for_sex, ever_had_sti, no_condom_use, is_orphan, has_child, 
            used_drugs_last_12mnths
        )
      # Make prediction
        prediction = model.predict(input_data_as_numpy_array)

        # Display result
        if prediction[0] == 1:
            st.write('AGYW is Not Eligible')
        else:
            st.write('AGYW is Eligible')
        
       
# Run the app
if __name__ == "__main__":
    main()
