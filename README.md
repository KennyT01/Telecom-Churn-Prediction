# Telecom Churn Prediction
Analysis and prediction of customer churn in the telecom industry

## Executive Summary
This project identifies high-risk customers likely to cancel their telecom subscriptions. By analysing customer experience, usage, service types and demographics, I developed a machine learning mdoel that achieves a **93% Recall rate**, potentially saving the company $6,000 in revenue by enabling proactive retention offers.

## The Business Problem
Customer churn is a major challenge for a majority of companies, specifically for telecom companies, leading to significant revenue loss. Customer churning is defined as when a customer stops using a service or product by a company. This metric is often portray as a percentage of total customers and is a key performance indicator in evaluating the performance of a product or service. Being able to identify the customers that have a high risk of churning before they actually churn will allow companies to take action and mitigate loss. This goal of the project is to predict:

- Identify the main factors driving churn
- Predict the customers which are at high risk of cancelling
- Provide targeted retention actions to mitigate loss

By analysing historical customer data and building an interactive dashboard, the company can proactively reduce churn and improve customer satisfaction.

## Streamlit Dashboard
![KPI-and-Prediction-Panel](./Images/KPI%20and%20Prediction%20Panel.png)
![Feature Explanationl](./Images/Feature_Explanation.png)
![KPI-and-Prediction-Panel](./Images/KPI-and-Prediction-Panel.png)



## Key Insights from EDA
- The distribution of customers that churn or not is quite imbalanced, with 15% churning and 85% staying.
- Almost 80% of customers that churned had complaints about their service, suggesting that poor customer experience is a driving factor in churning.
- Customers that churn have significantly lower customer value and frequency of use.

## Model Pipeline
1. **Preprocessing**: Handled missing values and outliers, and removed duplicate rows.
   
2. **Model Selection**: Tested Logistic Regression, Random Forest, and XGBoost. Handled imbalanced data using SMOTE and a weighted model. Logistic Regression performed the best after hyperparameter tuning via RandomizedSearchCV. Logistic Regression provided the best of interpretability and generalisation, likely because the relationships between 'Churn' and the features are strong linear, which Logistic Regression performs well on.
   
3. **Model Training and Evaluation**: Models were trained on the training set and evaluated on a validation set. Performance was assessed usins recall as the main metric and confusion matrices to analyse model errors. Recall was used instead of accuracy as missing a churning customer is more costly than sending a discount to a non-churning customer. We were also identified the features that had the most influence on whether a customer will churn or not.

## Business Impacts 
Based on our model and analysis, we were able to achieve a **Recall of 89%**, which means that we were able to identify 90% of the customers that are likely to churn. Based on the customer test set of 570 customers, we were able to identify 51 of them as high-risk customers, with a **Total Revenue at Risk of $6,340.13**. In addition, we also identified the features that had the most impact on customer churning. These are frequency of use, customer value, complaints, charge amount and call failures. With the model also providing the main reason for a customer to likely churn, we are able take direct action for our retention strategies. These strategies can include providing retention support and incentives, priority support call, troubleshooting technical difficulties, direct monitoring and reminder emails/app notifications. 

