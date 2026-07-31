import streamlit as st
import plotly.express as px

fig = px.pie(
    values=[10, 20],
    names=["Yes", "No"],
    title="Test Pie Chart"
)

st.plotly_chart(fig)