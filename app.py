import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("📊 Customer Churn Analytics Platform")
st.caption("Predict • Analyze • Retain Customers using Machine Learning")
st.divider()

st.markdown("""
<style>
div[data-testid="metric-container"]{
    background:#F8F9FA;
    border:2px solid #E5E7EB;
    padding:15px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("data/dataset.csv")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.title("🎛 Dashboard Filters")

gender = st.sidebar.multiselect(
    "Gender",
    data["gender"].unique(),
    default=data["gender"].unique()
)

contract = st.sidebar.multiselect(
    "Contract Type",
    data["Contract"].unique(),
    default=data["Contract"].unique()
)

internet = st.sidebar.multiselect(
    "Internet Service",
    data["InternetService"].unique(),
    default=data["InternetService"].unique()
)

payment = st.sidebar.multiselect(
    "Payment Method",
    data["PaymentMethod"].unique(),
    default=data["PaymentMethod"].unique()
)

filtered_data = data[
    (data["gender"].isin(gender)) &
    (data["Contract"].isin(contract)) &
    (data["InternetService"].isin(internet)) &
    (data["PaymentMethod"].isin(payment))
]

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("model/churn_model.pkl")
st.success("✅ Model Loaded Successfully!")

# -------------------------------
# KPI Metrics
# -------------------------------
if "Churn" in filtered_data.columns:
    churn_yes = len(filtered_data[filtered_data["Churn"] == "Yes"])
    churn_no = len(filtered_data[filtered_data["Churn"] == "No"])
else:
    churn_yes = 0
    churn_no = 0

total = len(filtered_data)
churn_rate = (churn_yes / total) * 100 if total > 0 else 0
avg_bill = filtered_data["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Customers", total)
col2.metric("❌ Churn", churn_yes)
col3.metric("✅ Retained", churn_no)
col4.metric("📈 Churn Rate", f"{churn_rate:.2f}%")

col5, col6, col7, col8 = st.columns(4)

col5.metric("💰 Avg Bill", f"${avg_bill:.2f}")
col6.metric("📅 Avg Tenure", f"{filtered_data['tenure'].mean():.1f}")
col7.metric("🌐 Internet Users", len(filtered_data))
col8.metric("💳 Payment Methods", filtered_data["PaymentMethod"].nunique())

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(filtered_data.head())

# -------------------------------
# Dataset Statistics
# -------------------------------
st.subheader("📋 Dataset Statistics")
st.dataframe(filtered_data.describe())
# -------------------------------
# Charts
# -------------------------------
if "Churn" in filtered_data.columns:

    churn_counts = filtered_data["Churn"].value_counts()

    # Pie Chart
    st.subheader("🥧 Customer Churn Distribution")

    pie = px.pie(
        values=churn_counts.values,
        names=churn_counts.index,
        hole=0.45,
        title="Customer Churn Distribution"
    )
    st.plotly_chart(pie, use_container_width=True)

    # Bar Chart
    st.subheader("📊 Churn Count")

    bar = px.bar(
        x=churn_counts.index,
        y=churn_counts.values,
        color=churn_counts.index,
        labels={"x": "Churn", "y": "Customers"},
        title="Customer Churn Count"
    )
    st.plotly_chart(bar, use_container_width=True)

    # Tenure Distribution
    st.subheader("📈 Customer Tenure Distribution")

    tenure = px.histogram(
        filtered_data,
        x="tenure",
        color="Churn",
        title="Customer Tenure Distribution"
    )
    st.plotly_chart(tenure, use_container_width=True)

    # Monthly Charges
    if "MonthlyCharges" in filtered_data.columns:

        st.subheader("💰 Monthly Charges vs Churn")

        monthly = px.box(
            filtered_data,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Monthly Charges vs Churn"
        )

        st.plotly_chart(monthly, use_container_width=True)

    # Contract Analysis
    if "Contract" in filtered_data.columns:

        st.subheader("📑 Contract Type Analysis")

        contract_chart = px.bar(
            filtered_data,
            x="Contract",
            color="Churn",
            title="Contract Type Analysis",
            barmode="group"
        )

        st.plotly_chart(contract_chart, use_container_width=True)

    # Internet Service Distribution
    if "InternetService" in filtered_data.columns:

        st.subheader("🌐 Internet Service Distribution")

        internet_chart = px.pie(
            filtered_data,
            names="InternetService",
            title="Internet Service Distribution"
        )

        st.plotly_chart(internet_chart, use_container_width=True)

else:
    st.error("❌ Churn column not found!")
# -------------------------------
# Sample Button
# -------------------------------
if st.button("Predict Sample"):
    st.success("✅ Dashboard is working successfully!")