from functions import *
import streamlit as st
import numpy
import pickle
from joblib import load
import base64
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title = 'Credit Score Classification Prediction Tool')

# Tüm model bileşenlerini yükleyin
model_components = load('credit_score_model_components.joblib')

# Bileşenleri ayrı değişkenlere atayın
model = model_components['model']
scaler = model_components['scaler']
oe_features = model_components['oe_features']
oe_target = model_components['oe_target']

def add_logo(logo_path, width, height):
    """Logonun base64 kodlanmış versiyonunu döndürür"""
    with open(logo_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"

# Logo ekleyin
logo_html = f'<img src="{add_logo("credit_score.png", width=200, height=100)}" style="display:block; margin-left:auto; margin-right:auto; width:500px; height:250px;">'

# CSS ile stillendirme yapın
st.markdown(
    """
    <style>
    .logo-container {
        text-align: center;
        padding: 20px 0;
    }
    .title {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo ve başlığı ekleyin
st.markdown(f'<div class="logo-container">{logo_html}</div>', unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center;'>Credit Score Classification Prediction Tool</h2>", unsafe_allow_html=True)

month_encode = {'January':1, 'February':2, 'March':3, 'April':4,
                'May':5, 'June':6, 'July':7, 'August':8}

occ_encode = {'Lawyer':1, 'Engineer':2, 'Mechanic':3, 'Teacher':4, 'Architect':5,
                  'Scientist':6, 'Accountant':7, 'Entrepreneur':8, 'Developer':9,
                  'Doctor':10, 'Media Manager':11, 'Journalist':12, 'Manager':13,
                  'Musician':14, 'Writer':15, 'Other':16}

payment_of_encode = {'No':0, 'Yes':1, 'NM':2}

credit_mix_encode = {'Bad':0, 'Standard':1, 'Good':2}

credit_score_encode = {'Poor':0, 'Standard':1, 'Good':2}

payment_spent_encode = {'Low':0, 'High':1}

payments_value_encode = {'Small':0, 'Medium':1, 'Large':2}

#Input Variables
Age = st.number_input('Age', min_value = 18, max_value = 75, step = 1)
Occupation = st.selectbox('Occupation', list(occ_encode.keys()))

Monthly_Inhand_Salary = st.number_input('Monthly Income', min_value = 0.0)
Monthly_Balance = st.number_input('Monthly Balance Amount', min_value = 0.0)
Amount_invested_monthly = st.number_input('Monthly Investment Amount', min_value = 0.0)

Num_Bank_Accounts = st.number_input('Number of Bank Accounts Owned by the Person', min_value = 0, step = 1)
Num_Credit_Card = st.number_input('Number of Credit Cards Owned by the Person', min_value = 0, step = 1)
Num_of_Loan = st.number_input('Number of Loans Owned by the Person', min_value = 0, step = 1)
Num_Credit_Inquiries = st.number_input('Number of Credit Card Applications Made by the Person', min_value = 0, step = 1)
Credit_History_Months = st.number_input('Credit History (in Months)', min_value = 0, step = 1)
Total_EMI_per_month = st.number_input('Monthly Loan Payment of the Person', min_value = 0.0)

Credit_Utilization_Ratio = st.slider('Credit Card Utilization Rate as a Percentage', 0.0, 100.0, 0.0, step=0.1)
Interest_Rate = st.slider("Monthly Interest Rate Applied to the Person's Credit Card", 0.0, 100.0, 0.0, step=0.1)
Changed_Credit_Limit = st.slider("Percentage Change in the Person's Credit Card Limit", 0.0, 37.0, 0.0, step=0.1)

Credit_Mix = st.selectbox("Person's Credit Mix", list(credit_mix_encode.keys()))
Payment_spent = st.selectbox("Person's Spending Behavior", list(payment_spent_encode.keys()))
Payments_value = st.selectbox("Person's Payment Behavior", list(payments_value_encode.keys()))

Delay_from_due_date = st.number_input('Average Number of Days Overdue Since Payment Due Date', min_value = 0, step = 1)
Num_of_Delayed_Payment = st.number_input("Number of Payments Delayed by the Person", min_value = 0, step = 1)
Payment_of_Min_Amount = st.selectbox("Has the Person Previously Paid Only the Minimum Amount of the Credit?", list(payment_of_encode.keys()))

Month = st.selectbox('Month', list(month_encode.keys()))
Outstanding_Debt = st.number_input('Outstanding Debt', min_value = 0.0)

if st.button('Predict'):

    Occupation_encoded = occ_encode[Occupation]
    Payment_of_Min_Amount_encoded = payment_of_encode[Payment_of_Min_Amount]
    Month_encoded = month_encode[Month]
    Credit_Mix_encoded, Payment_spent_encoded, Payments_value_encoded = oe_features.transform([[Credit_Mix, Payment_spent, Payments_value]])[0]

    features_to_scale = [Age, Monthly_Inhand_Salary, Monthly_Balance, Amount_invested_monthly,
                         Num_Bank_Accounts, Num_Credit_Card, Num_of_Loan, Num_Credit_Inquiries,
                         Credit_History_Months, Total_EMI_per_month, Credit_Utilization_Ratio,
                         Interest_Rate, Changed_Credit_Limit, Delay_from_due_date,
                         Num_of_Delayed_Payment, Outstanding_Debt]
    
    scaled_features = scaler.transform([features_to_scale])[0]

    input_features = [scaled_features[0], Occupation_encoded, scaled_features[1], scaled_features[2], scaled_features[3],
                      scaled_features[4], scaled_features[5], scaled_features[6], scaled_features[7],
                      scaled_features[8], scaled_features[9], scaled_features[10],
                      scaled_features[11], scaled_features[12], Credit_Mix_encoded, Payment_spent_encoded, Payments_value_encoded,
                      scaled_features[13], scaled_features[14], Payment_of_Min_Amount_encoded,
                      Month_encoded, scaled_features[15]]

    makeprediction = model.predict(input_features)

    target_map = {
        0: "Poor",
        1: "Standard",
        2: "Good"}
 
    original_prediction = makeprediction[0]

    predicted_label = target_map.get(original_prediction, "Unknown")
    
    st.success(f"Person's Credit Score: {predicted_label}")
