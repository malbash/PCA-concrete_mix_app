#######################
# app.py
#######################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from scipy.optimize import minimize

# 1) LOAD & TRAIN MODEL (in Metric)
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
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)
    return model

model = load_and_train_model()

# 2) CO₂ & COST COEFFICIENTS, ETC.
co2_coefficients = {
    "Cement (Kg/m3)": 0.795,
    "Blast Furnace Slag (Kg/m3)": 0.135,
    "Silica Fume (Kg/m3)": 0.024,
    "Fly Ash (Kg/m3)": 0.0235,
    "Water (Kg/m3)": 0.00025,
    "Coarse Aggregate (Kg/m3)": 0.026,
    "Fine Aggregate (Kg/m3)": 0.01545,
}

cost_coefficients = {
    "Cement (Kg/m3)": 0.10,
    "Blast Furnace Slag (Kg/m3)": 0.05,
    "Silica Fume (Kg/m3)": 0.40,
    "Fly Ash (Kg/m3)": 0.03,
    "Water (Kg/m3)": 0.0005,
    "Coarse Aggregate (Kg/m3)": 0.02,
    "Fine Aggregate (Kg/m3)": 0.015,
}

cleaned_names = {
    "Cement (Kg/m3)": "Cement",
    "Blast Furnace Slag (Kg/m3)": "Blast Furnace Slag",
    "Silica Fume (Kg/m3)": "Silica Fume",
    "Fly Ash (Kg/m3)": "Fly Ash",
    "Water (Kg/m3)": "Water",
    "Coarse Aggregate (Kg/m3)": "Coarse Aggregate",
    "Fine Aggregate (Kg/m3)": "Fine Aggregate",
}

# Conversion factors
KG_TO_LB = 2.20462
MPA_TO_PSI = 145.038
KG_PER_M3_TO_LB_PER_FT3 = 0.06242796
M3_TO_FT3 = 35.3146667

# Strength ranges -> used for bounding
strength_ranges = {
    "Normal Strength": "NSC",
    "High Strength": "HSC",
    "Ultra-High performance": "UHPC"
}

# Minimization methods
minimize_methods_display = {
    "SLSQP": "SLSQP",
    "COBYLA": "COBYLA",
    "L-BFGS-B": "L-BFGS-B",
    "Trust-Constr": "trust-constr",
}

def get_bounds(concrete_type):
    """Same logic as your code, returning skopt Real(...) bounds."""
    if concrete_type == "NSC":
        return [
            Real(140, 400, "Cement (Kg/m3)"),
            Real(0, 150,   "Blast Furnace Slag (Kg/m3)"),
            Real(0, 1,     "Silica Fume (Kg/m3)"),
            Real(0, 100,   "Fly Ash (Kg/m3)"),
            Real(120, 200, "Water (Kg/m3)"),
            Real(800, 1200,"Coarse Aggregate (Kg/m3)"),
            Real(600, 700, "Fine Aggregate (Kg/m3)"),
        ]
    elif concrete_type == "HSC":
        return [
            Real(240, 550, "Cement (Kg/m3)"),
            Real(0, 150,   "Blast Furnace Slag (Kg/m3)"),
            Real(0, 50,    "Silica Fume (Kg/m3)"),
            Real(0, 150,   "Fly Ash (Kg/m3)"),
            Real(105, 160, "Water (Kg/m3)"),
            Real(700, 1000,"Coarse Aggregate (Kg/m3)"),
            Real(600, 800, "Fine Aggregate (Kg/m3)"),
        ]
    else:  # UHPC
        return [
            Real(350, 1000,"Cement (Kg/m3)"),
            Real(0, 150,   "Blast Furnace Slag (Kg/m3)"),
            Real(140, 300, "Silica Fume (Kg/m3)"),
            Real(0, 200,   "Fly Ash (Kg/m3)"),
            Real(125, 150, "Water (Kg/m3)"),
            Real(0, 1,     "Coarse Aggregate (Kg/m3)"),
            Real(650, 1200,"Fine Aggregate (Kg/m3)"),
        ]

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
    return total_cost  # $ per 1 m³

# Bayesian Optimization
@use_named_args([])
def objective_bayes(**params):
    pass

def optimize_mix_bayesian(target_strength_mpa, concrete_type, acq_func="EI"):
    bounds = get_bounds(concrete_type)
    iteration_values = []

    @use_named_args(bounds)
    def objective(**params):
        mix = list(params.values())
        strength = predict_strength(mix)
        penalty = max(0, (1.1 * target_strength_mpa - strength)) ** 2
        val = calculate_co2_kg(mix) + 10 * penalty
        iteration_values.append(val)
        return val

    res = gp_minimize(
        objective,
        dimensions=bounds,
        n_calls=50,
        random_state=42,
        acq_func=acq_func
    )
    return res, iteration_values

# SciPy Minimize
def optimize_mix_minimize(target_strength_mpa, concrete_type, method="SLSQP"):
    bounds = [(b.low, b.high) for b in get_bounds(concrete_type)]
    iteration_values = []

    def objective(mix):
        mix = np.clip(mix, [b[0] for b in bounds], [b[1] for b in bounds])
        val = calculate_co2_kg(mix)
        iteration_values.append(val)
        return val

    def constraint(mix):
        mix = np.clip(mix, [b[0] for b in bounds], [b[1] for b in bounds])
        return predict_strength(mix) - 1.1 * target_strength_mpa

    cons = [{"type": "ineq", "fun": constraint}]
    x0 = np.mean(bounds, axis=1)

    res = minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        constraints=cons
    )
    return res, iteration_values

# 3) STREAMLIT APP
st.set_page_config(page_title="PCA Concrete Mix Optimizer", layout="wide")

st.markdown(
    """
    <h2 style="background-color:#003366; color:white; padding:10px; text-align:center;">
      Portland Cement Association - Concrete Mix Optimizer
    </h2>
    """,
    unsafe_allow_html=True
)

# Sidebar / or main panel for unit toggle
unit_system = st.radio("Select Unit System:", ["Metric", "US"], index=0)

st.write("## Input Type")
input_type = st.radio("", ["Target Strength", "Ingredients"], index=0, horizontal=True)

if input_type == "Target Strength":
    label_str = "Target Strength (MPa)" if unit_system=="Metric" else "Target Strength (psi)"
    target_strength_val = st.number_input(label_str, min_value=0.0, value=30.0)
else:
    st.write("### Input Ingredients")
    user_mix = []
    for col_key in co2_coefficients.keys():
        if unit_system=="Metric":
            val = st.number_input(f"{cleaned_names[col_key]} (Kg/m³)", min_value=0.0, value=100.0)
            user_mix.append(val)
        else:
            # if in US, user is providing lb/ft³
            val = st.number_input(f"{cleaned_names[col_key]} (lb/ft³)", min_value=0.0, value=10.0)
            # convert to kg/m³ internally
            val_metric = val / KG_PER_M3_TO_LB_PER_FT3
            user_mix.append(val_metric)

st.write("## Concrete Type")
concrete_choice = st.selectbox("", list(strength_ranges.keys()), index=0)
concrete_code = strength_ranges[concrete_choice]

st.write("## Optimization Settings")
acq_choice = st.selectbox("Bayesian acq_func:", ["EI", "PI"], index=0)
minimize_choice = st.selectbox("Minimize Method:", list(minimize_methods_display.keys()), index=0)

# Press "Optimize" button
if st.button("Optimize"):
    if input_type == "Target Strength":
        # convert user input to MPa if in US
        if unit_system == "Metric":
            target_strength_mpa = target_strength_val
        else:
            # user typed psi, convert to MPa
            target_strength_mpa = target_strength_val / MPA_TO_PSI
    else:
        # user typed a mix
        # predict strength in MPa (already in metric behind the scenes)
        predicted_strength_mpa = predict_strength(user_mix)
        target_strength_mpa = predicted_strength_mpa
        st.write(f"**Predicted Strength:** {predicted_strength_mpa:.2f} MPa")

    # 1) Bayesian
    res_bayes, iters_bayes = optimize_mix_bayesian(target_strength_mpa, concrete_code, acq_choice)
    # 2) Minimize
    res_min, iters_min = optimize_mix_minimize(target_strength_mpa, concrete_code, minimize_methods_display[minimize_choice])

    # Extract final solutions
    bayes_mix = res_bayes.x
    bayes_co2 = calculate_co2_kg(bayes_mix)
    bayes_str = predict_strength(bayes_mix)
    bayes_cost = calculate_cost(bayes_mix)

    min_mix = res_min.x
    min_co2 = calculate_co2_kg(min_mix)
    min_str = predict_strength(min_mix)
    min_cost = calculate_cost(min_mix)

    # Summarize iteration values
    def co2_for_display(c):
        return c if unit_system=="Metric" else c * KG_TO_LB

    # Layout in columns
    c1, c2 = st.columns(2)

    with c1:
        # Bayesian result
        st.markdown("### Bayesian Optimization Result")
        # Convert CO2, Strength, Cost for display
        if unit_system=="Metric":
            co2_disp = bayes_co2
            str_disp = bayes_str
            cost_disp = bayes_cost
            co2_unit = "kg/m³"
            str_unit = "MPa"
            cost_unit = "$/m³"
            mix_unit = "Kg/m³"
        else:
            co2_disp = bayes_co2 * KG_TO_LB  # lb per
            str_disp = bayes_str * MPA_TO_PSI
            cost_disp = bayes_cost / M3_TO_FT3
            co2_unit = "lb/m³"  # Not strictly correct dimension, but matches user approach
            str_unit = "psi"
            cost_unit = "$/ft³"
            mix_unit = "lb/ft³"

        st.write(f"**CO₂ Emissions:** {co2_disp:.2f} {co2_unit}")
        st.write(f"**Strength:** {str_disp:.2f} {str_unit}")
        st.write(f"**Cost:** ${cost_disp:.2f} per {cost_unit}")
        st.write("**Mix Proportions:**")
        for ingr_name, val_metric in zip(co2_coefficients.keys(), bayes_mix):
            if unit_system=="Metric":
                final_val = val_metric
            else:
                final_val = val_metric * KG_PER_M3_TO_LB_PER_FT3
            st.write(f"- {cleaned_names[ingr_name]}: {final_val:.2f} {mix_unit}")

    with c2:
        # Minimize result
        st.markdown("### Minimize Optimization Result")
        if unit_system=="Metric":
            co2_disp = min_co2
            str_disp = min_str
            cost_disp = min_cost
            co2_unit = "kg/m³"
            str_unit = "MPa"
            cost_unit = "$/m³"
            mix_unit = "Kg/m³"
        else:
            co2_disp = min_co2 * KG_TO_LB
            str_disp = min_str * MPA_TO_PSI
            cost_disp = min_cost / M3_TO_FT3
            co2_unit = "lb/m³"
            str_unit = "psi"
            cost_unit = "$/ft³"
            mix_unit = "lb/ft³"

        st.write(f"**CO₂ Emissions:** {co2_disp:.2f} {co2_unit}")
        st.write(f"**Strength:** {str_disp:.2f} {str_unit}")
        st.write(f"**Cost:** ${cost_disp:.2f} per {cost_unit}")
        st.write("**Mix Proportions:**")
        for ingr_name, val_metric in zip(co2_coefficients.keys(), min_mix):
            if unit_system=="Metric":
                final_val = val_metric
            else:
                final_val = val_metric * KG_PER_M3_TO_LB_PER_FT3
            st.write(f"- {cleaned_names[ingr_name]}: {final_val:.2f} {mix_unit}")

    # Plot iteration values side by side
    st.markdown("---")
    st.write("### Iteration Values")

    fig, ax = plt.subplots()
    bayes_plot = [co2_for_display(c) for c in iters_bayes]
    min_plot = [co2_for_display(c) for c in iters_min]
    ax.plot(range(1, len(bayes_plot)+1), bayes_plot, marker="o", label="Bayesian")
    ax.plot(range(1, len(min_plot)+1), min_plot, marker="x", label="Minimize")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("CO₂ Emissions" + (" (lb)" if unit_system=="US" else " (kg)"))
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    st.success("Optimization Complete!")
