import pickle
import streamlit as st
import numpy as np
import pandas as pd

# -----------------------------
# Load trained model & scaler
# -----------------------------
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
num_cols = pickle.load(open("num_cols.pkl","rb"))
features_order = pickle.load(open("features.pkl","rb"))

# -----------------------------
# Title
# -----------------------------
st.title("HR Employee Attrition Prediction")

st.write("Enter employee details to predict Attrition")

# -----------------------------
# Input Fields
# -----------------------------

Age = st.number_input("Age", 18, 60, 30)

BusinessTravel = st.selectbox(
    "Business Travel",
    ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
)

DailyRate = st.number_input("Daily Rate", 100, 1500, 800)

Department = st.selectbox(
    "Department",
    ["Sales", "Research & Development", "Human Resources"]
)

DistanceFromHome = st.number_input("Distance From Home", 1, 30, 5)

Education = st.selectbox(
    "Education Level",
    [1, 2, 3, 4, 5]
)

EnvironmentSatisfaction = st.selectbox(
    "Environment Satisfaction",
    [1, 2, 3, 4]
)

Gender = st.selectbox("Gender", ["Male", "Female"])

JobInvolvement = st.selectbox("Job Involvement", [1,2,3,4])

JobLevel = st.selectbox("Job Level", [1,2,3,4,5])

JobSatisfaction = st.selectbox("Job Satisfaction", [1,2,3,4])

MonthlyIncome = st.number_input("Monthly Income", 1000, 50000, 5000)

NumCompaniesWorked = st.number_input("Companies Worked", 0, 10, 1)

OverTime = st.selectbox("OverTime", ["Yes", "No"])

PercentSalaryHike = st.number_input("Salary Hike %", 10, 30, 15)

TotalWorkingYears = st.number_input("Total Working Years", 0, 40, 5)

TrainingTimesLastYear = st.number_input("Training Times Last Year", 0, 10, 2)

WorkLifeBalance = st.selectbox("Work Life Balance", [1,2,3,4])

YearsAtCompany = st.number_input("Years At Company", 0, 40, 5)

YearsInCurrentRole = st.number_input("Years In Current Role", 0, 20, 3)

YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", 0, 15, 1)

YearsWithCurrManager = st.number_input("Years With Current Manager", 0, 20, 3)

# -----------------------------
# Encode Categorical Values
# -----------------------------

BusinessTravel = {
    "Travel_Rarely":0,
    "Travel_Frequently":1,
    "Non-Travel":2
}[BusinessTravel]

Department = {
    "Sales":0,
    "Research & Development":1,
    "Human Resources":2
}[Department]

Gender = {
    "Male":1,
    "Female":0
}[Gender]

OverTime = {
    "Yes":1,
    "No":0
}[OverTime]

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Attrition"):

    # Create dataframe
    input_df = pd.DataFrame({
        "Age":[Age],
        "BusinessTravel":[BusinessTravel],
        "DailyRate":[DailyRate],
        "Department":[Department],
        "DistanceFromHome":[DistanceFromHome],
        "Education":[Education],
        "EnvironmentSatisfaction":[EnvironmentSatisfaction],
        "Gender":[Gender],
        "JobInvolvement":[JobInvolvement],
        "JobLevel":[JobLevel],
        "JobSatisfaction":[JobSatisfaction],
        "MonthlyIncome":[MonthlyIncome],
        "NumCompaniesWorked":[NumCompaniesWorked],
        "OverTime":[OverTime],
        "PercentSalaryHike":[PercentSalaryHike],
        "TotalWorkingYears":[TotalWorkingYears],
        "TrainingTimesLastYear":[TrainingTimesLastYear],
        "WorkLifeBalance":[WorkLifeBalance],
        "YearsAtCompany":[YearsAtCompany],
        "YearsInCurrentRole":[YearsInCurrentRole],
        "YearsSinceLastPromotion":[YearsSinceLastPromotion],
        "YearsWithCurrManager":[YearsWithCurrManager]
    })

    # Maintain same column order
    input_df = input_df.reindex(columns=features_order, fill_value=0)

    # Scale ONLY numeric columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Prediction
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Employee Likely to Leave")
    else:
        st.success("✅ Employee Likely to Stay")