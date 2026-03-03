import numpy as np
import pandas as pd

def get_feature_importance(model, features):
    coefficients = model.named_steps['classifier'].coef_[0]

    feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
    }).sort_values(by='abs_coefficient', ascending=False)

    feature_importance['odds_ratio'] = round(np.exp(feature_importance['coefficient']), 2)
    feature_importance = feature_importance.sort_values(by='abs_coefficient', ascending=False)

    return feature_importance

def churn_cause(data, churn_prob):
    data = data.iloc[0]
    if churn_prob >= 0.6:
        if  data['Frequency of use'] < 30 or data['Customer Value'] < 500  or data['Charge  Amount'] < 3:
            churn_cause = "Low Commitment and Engagement"
        elif data['Complains'] == 1:
            churn_cause = "Dissatisfaction"
        elif data['Call  Failure'] > 5:
            churn_cause = "Network Quality"
        else:
            churn_cause = "General Risk"
    else:
        churn_cause = "Low Probability of Churn."

    return churn_cause

def action_priority(churn_cause, risk_level):
    if risk_level == 'High':
        if churn_cause == "Low Commitment and Engagement":
            action = "Offer retention support and incentives"
        elif churn_cause == "Dissatifaction":
            action = "Priority customer support call and immediate follow-up"
        elif churn_cause == "Network Quality":
            action = "Urgent technical investigation and compensation"
        
        else:
            action = "High-touch retention action (e.g., call or email)"

    elif risk_level == "Moderate":
        if churn_cause == "Low Commitment and Engagement":
            action = "Send engagement nudges or small incentive"
        elif churn_cause == "Dissatisfaction":
            action = "Send follow-up email + monitor complaints"
        elif churn_cause == "Network Quality":
            action = "Notify customer of monitoring / minor compensation"
        else:
            action = "Send reminder email / app notification"
    else:  
        action = None
        
    return action