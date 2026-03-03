import streamlit as st
import joblib
import pandas as pd
import numpy as np

import plotly.express as px
from utils import get_feature_importance, churn_cause, action_priority


st.set_page_config(layout="wide") 
st.title("Telecom Churn Dashboard")

# **Key Features:**
# - Calculates total revenue at risk
# - Displays feature importance to highlight key churn drivers
# - Predicts churn probability for each customer
# - Identifies risk levels of customers with action recommendation
# """)


df_test = pd.read_csv("data/X_test.csv")

model = joblib.load('churn_model.pkl')

features = joblib.load('features.pkl')

# KPIs
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    col1.metric("Number of Customers", f"{len(df_test)}")
    col1.metric("High-Risk Customers", f"{df_test['High Risk Customer'].sum()}")

with col2:
    col2.metric("Current Churn Rate", f"{df_test['Churn'].mean():.2%}")
    col2.metric("Total Revenue at Risk", f"${df_test['Revenue at Risk'].sum():,.2f}")

with col3:
    col3.metric("Predicted Churn Rate", f"{df_test['Churn Prediction'].mean():.2%}")
    col3.metric("Revenue at Risk (High-Risk)", 
            f"${df_test.loc[df_test['High Risk Customer']==1,'Revenue at Risk'].sum():,.2f}")

st.subheader("Customer Prediction Panel")
col1, col2= st.columns(2)

with col1:
    call_fail = st.number_input("Call Failure", min_value=0)
    charge = st.selectbox("Charge Amount", list(range(0, 10)))
    value = st.number_input("Customer Value", min_value=0.0)


with col2:
    complains = st.selectbox("Complain in last 9 months?", ["No", "Yes"])
    freq_use = st.number_input("Frequency of Use", min_value=0)


complains = 0 if complains == "No" else 1

# Default Values for Hidden Features (No Impact on prediction)
plan = df_test['Tariff Plan'].mode()[0]
age = df_test['Age'].median()
sub_length = df_test['Subscription  Length'].median()



if st.button("Predict Churn"):
    # Create a dataframe from the inputs
    input_df = pd.DataFrame([[call_fail, complains, charge, freq_use, plan, 
                              age, value, sub_length]], 
                            columns=features) 
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if probability >=0.75:
        risk_level = "High"
    elif probability >= 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    if risk_level == "High":
        st.error(f"High Risk! This customer is likely to churn. (Probability: {probability:.2%})")

    elif risk_level == "Moderate":
        st.warning(f"Moderate Risk. This customer is likely to churn. (Probability: {probability:.2%})")

    else:
        st.success(f"Low Risk. This customer is likely to stay. (Probability: {1-probability:.2%})")

    if risk_level != "Low":    
        churn_cause = churn_cause(input_df, probability)
        st.write(f"**Predicted Cause:** {churn_cause}")

    action = action_priority(churn_cause, risk_level)

    st.write(f"**Recommended Action:** {action}")


st.subheader("Feature Explanation")
st.markdown("""
Visualise the important drivers of churn and how much they impact the churn probability prediction.
""")

df = pd.read_csv("data/cleaned_churn_data.csv")

feature_importance = get_feature_importance(model, features).sort_values(by='abs_coefficient', ascending=True)

col1, col2 = st.columns(2)

fig1 = px.bar(
    feature_importance,
    x='coefficient',
    y='feature',
    orientation='h',
    color='coefficient',
    color_continuous_scale='RdBu',
    title="Feature Importance (coefficient & direction)"
)


fig2 = px.bar(
    feature_importance.sort_values(by='odds_ratio', ascending=False),
    x='feature',
    y='odds_ratio',
    orientation='v',
    color='feature',
    color_continuous_scale='RdBu',
    title="Odds Ratio"
)

with col1:
    st.plotly_chart(fig1, width='stretch')

with col2:
    st.plotly_chart(fig2, width='stretch')


st.subheader("Global Data Analysis")
st.markdown("""
This plots show the proportion of customers who churned versus those retained.
This will help the company understand the overall churn levels and identify risk segments.
""")


col1, col2 = st.columns(2)
df['Status'] = df['Churn'].map({0: 'Retained', 1: 'Churn'})
df['Tariff Plan'] = df['Tariff Plan'].map({1: 'Pay As You Go', 2: 'Contractual'})

fig1 = px.histogram(
    df,
    x='Frequency of use',
    color='Status',
    barmode='group',
    title="Churn by Frequency of Use",
    color_discrete_map={'Retained': 'green', 'Churn': 'red'}
)
with col1:
    st.plotly_chart(fig1, width='stretch')

df['Complains'] = df['Complains'].map({0: "No", 1: "Yes"})

fig2 = px.histogram(
    df,
    x='Customer Value',
    color='Status',
    barmode='group',
    title="Churn by Customer Value",
    color_discrete_map={'Retained': 'green', 'Churn': 'red'}
)
with col2:
    st.plotly_chart(fig2, width='stretch')

fig3 = px.histogram(
    df,
    x='Complains',
    color='Status',
    barmode='group',
    title="Churn by Complains",
    color_discrete_map={'Retained': 'green', 'Churn': 'red'}
)
with col1:
    st.plotly_chart(fig3, width='stretch')

fig4 = px.histogram(
    df,
    x='Charge  Amount',
    color='Status',
    barmode='group',
    title="Churn by Charge Amounts",
    color_discrete_map={'Retained': 'green', 'Churn': 'red'}
)
with col2:
    st.plotly_chart(fig4, width='stretch')



