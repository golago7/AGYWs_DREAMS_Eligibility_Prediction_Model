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

    # Collect other inputs with no pre-loaded options
    marital_status = st.selectbox("Marital Status", ["Select an option", "Single", "Married", "Widowed", "Cohabiting"])
    if marital_status == "Select an option":
        st.error("Please select a marital status.")
        st.stop()

    has_disability = st.selectbox("Do you have any form of disability?", ["Select an option", "Yes", "No"])
    if has_disability == "Select an option":
        st.error("Please select an option for disability status.")
        st.stop()

    out_of_school = st.selectbox("Out of School", ["Select an option", "Yes", "No"])
    if out_of_school == "Select an option":
        st.error("Please select an option for out of school status.")
        st.stop()

    ever_had_sex = st.selectbox("Ever Had Sex", ["Select an option", "Yes", "No"])
    if ever_had_sex == "Select an option":
        st.error("Please select an option for sexual activity status.")
        st.stop()

    is_head = st.selectbox("Is the head of the household or in a child headed household", ["Select an option", "Yes", "No"])
    if is_head == "Select an option":
        st.error("Please select an option for household head status.")
        st.stop()

    undergone_gbv_last_12mnths = st.selectbox("Undergoing violence or has undergone violence in the last 12 Months? (Physical, Emotional, Sexual, Social economic Violence)", ["Select an option", "Yes", "No"])
    if undergone_gbv_last_12mnths == "Select an option":
        st.error("Please select an option for violence status.")
        st.stop()

    sexual_partners_last_12mnths = st.selectbox("Has had more than one sexual partner in the last 12 months?", ["Select an option", "Yes", "No"])
    if sexual_partners_last_12mnths == "Select an option":
        st.error("Please select an option for sexual partners.")
        st.stop()

    received_gifts_for_sex = st.selectbox("Has ever received money gifts or favors in exchange for sex?", ["Select an option", "Yes", "No"])
    if received_gifts_for_sex == "Select an option":
        st.error("Please select an option for receiving gifts for sex.")
        st.stop()

    ever_had_sti = st.selectbox("Have been diagnosed or treated for STI?", ["Select an option", "Yes", "No"])
    if ever_had_sti == "Select an option":
        st.error("Please select an option for STI history.")
        st.stop()

    no_condom_use = st.selectbox("No or irregular condom use with a non-marital /non-cohabiting partner?", ["Select an option", "Yes", "No"])
    if no_condom_use == "Select an option":
        st.error("Please select an option for condom use.")
        st.stop()

    is_orphan = st.selectbox("Is an orphan (partial or total)", ["Select an option", "Yes", "No"])
    if is_orphan == "Select an option":
        st.error("Please select an option for orphan status.")
        st.stop()

    has_child = st.selectbox("Has a child of her own/is pregnant/has been pregnant?", ["Select an option", "Yes", "No"])
    if has_child == "Select an option":
        st.error("Please select an option for child/pregnancy status.")
        st.stop()

    used_drugs_last_12mnths = st.selectbox("Has used alcohol/drugs or abused or struggled with addiction in the last 12 months?", ["Select an option", "Yes", "No"])
    if used_drugs_last_12mnths == "Select an option":
        st.error("Please select an option for drug/alcohol use.")
        st.stop()

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
            st.write('AGYW is Not Eligible for Enrollment')
        else:
            st.write('AGYW is Eligible for Enrollment')
        
# Run the app
if __name__ == "__main__":
    main()
