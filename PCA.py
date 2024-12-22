#######################
# app.py
#######################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.optimize import minimize

# 1) MUST be first Streamlit command
st.set_page_config(page_title="PCA Concrete Mix Optimizer", layout="wide")

# -----------------------
# 2) LOAD & TRAIN MODEL
# -----------------------
@st.cache_data
def load_and_train_model():
    df = pd.read_excel('cleaned.xlsx')

    X = df[[
        'Cement (Kg/m3)',
        'Blast Furnace Slag (Kg/m3)',
        'Silica Fume (Kg/m3)',
        'Fly Ash (Kg/m3)',
        'Water (Kg/m3)',
        'Coarse Aggregate (Kg/m3)',
        'Fine Aggregate (Kg/m3)'
    ]]
    y = df["F'c (MPa)"]

    model = RandomForestRegressor(
        max_depth=None,
        max_features='sqrt',
        n_estimators=50,
        random_state=42
    )
    model.fit(X, y)
    return model

model = load_and_train_model()

# -----------------------
# 3) DEFAULT COEFFICIENTS
# -----------------------
default_co2_coefficients_metric = {
    "Cement (Kg/m3)": 0.795,
    "Blast Furnace Slag (Kg/m3)": 0.135,
    "Silica Fume (Kg/m3)": 0.024,
    "Fly Ash (Kg/m3)": 0.0235,
    "Water (Kg/m3)": 0.00025,
    "Coarse Aggregate (Kg/m3)": 0.026,
    "Fine Aggregate (Kg/m3)": 0.01545,
}

default_cost_coefficients_metric = {
    "Cement (Kg/m3)": 0.10,
    "Blast Furnace Slag (Kg/m3)": 0.05,
    "Silica Fume (Kg/m3)": 0.40,
    "Fly Ash (Kg/m3)": 0.03,
    "Water (Kg/m3)": 0.0005,
    "Coarse Aggregate (Kg/m3)": 0.02,
    "Fine Aggregate (Kg/m3)": 0.015,
}

# US coefficients are derived dynamically by conversion
KG_TO_LB = 2.20462  # Kilogram to pound
COST_CONVERSION = 35.3147  # m³ to ft³ for cost conversion

# Function to convert coefficients to US

def convert_to_us(coefficients_metric, is_cost=False):
    return {
        key: value * (KG_TO_LB if not is_cost else 1 / COST_CONVERSION)
        for key, value in coefficients_metric.items()
    }

default_co2_coefficients_us = convert_to_us(default_co2_coefficients_metric)
default_cost_coefficients_us = convert_to_us(default_cost_coefficients_metric, is_cost=True)

# Initialize coefficients
co2_coefficients = default_co2_coefficients_metric.copy()
cost_coefficients = default_cost_coefficients_metric.copy()

# -----------------------
# 4) FUNCTIONS
# -----------------------
def predict_strength(mix_metric):
    df_input = pd.DataFrame([mix_metric], columns=co2_coefficients.keys())
    return model.predict(df_input)[0]

def calculate_co2_kg(mix_metric):
    return sum(
        mix_metric[i] * list(co2_coefficients.values())[i]
        for i in range(len(mix_metric))
    )

def calculate_cost(mix_metric):
    total_cost = 0.0
    cost_list = list(cost_coefficients.values())
    for i, amt in enumerate(mix_metric):
        total_cost += amt * cost_list[i]
    return total_cost  # in $/m³

# -----------------------
# 5) STREAMLIT APP
# -----------------------
st.markdown(
    """
    <h2 style="background-color:#003366; color:white; padding:10px; text-align:center;">
      Portland Cement Association - Concrete Mix Optimizer
    </h2>
    """,
    unsafe_allow_html=True
)

# Let user pick Metric or US
unit_system = st.radio("Select Unit System:", ["Metric", "US"], index=0)

# Adjust coefficients based on unit system
if unit_system == "Metric":
    co2_coefficients = default_co2_coefficients_metric.copy()
    cost_coefficients = default_cost_coefficients_metric.copy()
else:
    co2_coefficients = default_co2_coefficients_us.copy()
    cost_coefficients = default_cost_coefficients_us.copy()

# Display default options
st.markdown("### Default Optimization Options")
st.write("The app uses the following default optimization options:")
st.write("**Bayesian Optimization Acquisition Function:** EI")
st.write("**Minimize Optimization Method:** Trust-Constr")
st.write("**Default CO₂ Coefficients (based on selected unit):**")
st.json(co2_coefficients)
st.write("**Default Cost Coefficients (based on selected unit):**")
st.json(cost_coefficients)

# Advanced Options Toggle
advanced_options = st.checkbox("Enable Advanced Options")

if advanced_options:
    st.markdown("### Customize Coefficients")

    # Editable CO₂ Coefficients
    st.subheader("Edit CO₂ Coefficients")
    for key in co2_coefficients.keys():
        co2_coefficients[key] = st.number_input(
            f"{key}", value=co2_coefficients[key], key=f"co2_{key}")

    # Editable Cost Coefficients
    st.subheader("Edit Cost Coefficients")
    for key in cost_coefficients.keys():
        cost_coefficients[key] = st.number_input(
            f"{key}", value=cost_coefficients[key], key=f"cost_{key}")

st.success("Setup Complete! Modify the coefficients if necessary and proceed with optimization.")
