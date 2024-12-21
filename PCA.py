#pip install streamlit scikit-learn scipy scikit-optimize matplotlib pandas openpyxl
##############################
# app.py
##############################

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
from io import BytesIO

# =========================
# 1) LOAD AND TRAIN MODEL
# =========================
@st.cache_data  # Caches the result so it doesn't re-train on every run
def load_and_train_model():
    # Adapt these lines if your Excel columns differ
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
    y = df["F'c (MPa)"]  # Strength in MPa

    model = RandomForestRegressor(
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)
    return model

model = load_and_train_model()

# Optional: Evaluate quickly (printed to console/log)
# st.write("Model trained. Evaluate on loaded data:")
# y_pred = model.predict(X)
# st.write("MSE:", mean_squared_error(y, y_pred))
# st.write("R^2:", r2_score(y, y_pred))

# =========================
# 2) DICTIONARIES & FUNCTIONS
# =========================

# Example CO₂ coefficients (kg CO₂ per kg of material)
co2_coefficients = {
    "Cement (Kg/m3)": 0.795,
    "Blast Furnace Slag (Kg/m3)": 0.135,
    "Silica Fume (Kg/m3)": 0.024,
    "Fly Ash (Kg/m3)": 0.0235,
    "Water (Kg/m3)": 0.00025,
    "Coarse Aggregate (Kg/m3)": 0.026,
    "Fine Aggregate (Kg/m3)": 0.01545,
}

# Example cost coefficients ($ per kg)
cost_coefficients = {
    "Cement (Kg/m3)": 0.10,
    "Blast Furnace Slag (Kg/m3)": 0.05,
    "Silica Fume (Kg/m3)": 0.40,
    "Fly Ash (Kg/m3)": 0.03,
    "Water (Kg/m3)": 0.0005,
    "Coarse Aggregate (Kg/m3)": 0.02,
    "Fine Aggregate (Kg/m3)": 0.015,
}

def calculate_co2_kg(mix):
    """Sum up CO₂ (kg) for each ingredient in the mix (in kg/m³)."""
    return sum(
        mix[i] * list(co2_coefficients.values())[i]
        for i in range(len(mix))
    )

def calculate_cost(mix):
    """Calculate total cost ($) for 1 m³ of the given mix."""
    total_cost = 0.0
    cost_vals = list(cost_coefficients.values())
    for i, amount in enumerate(mix):
        total_cost += amount * cost_vals[i]
    return total_cost  # in $ per 1 m³

def predict_strength(mix):
    """Use the trained RandomForest to predict strength (MPa) for the given mix."""
    df_input = pd.DataFrame([mix], columns=co2_coefficients.keys())
    return model.predict(df_input)[0]

# ---------------------------------
# Different concrete type bounds:
# ---------------------------------
strength_ranges = {
    "Normal Strength": "NSC",
    "High Strength": "HSC",
    "Ultra-High performance": "UHPC"
}

def get_bounds(concrete_type):
    """Return skopt bounds for the selected concrete type code."""
    if concrete_type == "NSC":
        return [
            Real(140, 400, name="Cement (Kg/m3)"),
            Real(0, 150,   name="Blast Furnace Slag (Kg/m3)"),
            Real(0, 1,     name="Silica Fume (Kg/m3)"),
            Real(0, 100,   name="Fly Ash (Kg/m3)"),
            Real(120, 200, name="Water (Kg/m3)"),
            Real(800, 1200, name="Coarse Aggregate (Kg/m3)"),
            Real(600, 700,  name="Fine Aggregate (Kg/m3)"),
        ]
    elif concrete_type == "HSC":
        return [
            Real(240, 550, name="Cement (Kg/m3)"),
            Real(0, 150,   name="Blast Furnace Slag (Kg/m3)"),
            Real(0, 50,    name="Silica Fume (Kg/m3)"),
            Real(0, 150,   name="Fly Ash (Kg/m3)"),
            Real(105, 160, name="Water (Kg/m3)"),
            Real(700, 1000, name="Coarse Aggregate (Kg/m3)"),
            Real(600, 800,  name="Fine Aggregate (Kg/m3)"),
        ]
    else: # "UHPC"
        return [
            Real(350, 1000, name="Cement (Kg/m3)"),
            Real(0, 150,    name="Blast Furnace Slag (Kg/m3)"),
            Real(140, 300,  name="Silica Fume (Kg/m3)"),
            Real(0, 200,    name="Fly Ash (Kg/m3)"),
            Real(125, 150,  name="Water (Kg/m3)"),
            Real(0, 1,      name="Coarse Aggregate (Kg/m3)"),
            Real(650, 1200, name="Fine Aggregate (Kg/m3)"),
        ]

# =====================================
# 3) OPTIMIZATION FUNCTIONS
# =====================================

from skopt.utils import use_named_args

def optimize_mix_bayesian(target_strength_mpa, concrete_type_code, acq_func="EI"):
    """Bayesian optimization minimizing CO2 + penalty if strength < 1.1 * target."""
    bounds = get_bounds(concrete_type_code)
    iteration_values = []

    @use_named_args(bounds)
    def objective(**params):
        mix = list(params.values())
        predicted = predict_strength(mix)
        # penalty if strength < 1.1 * target
        penalty = max(0, (1.1 * target_strength_mpa - predicted)) ** 2
        val = calculate_co2_kg(mix) + 10 * penalty
        iteration_values.append(val)
        return val

    result = gp_minimize(
        objective,
        dimensions=bounds,
        n_calls=30,       # can adjust
        random_state=42,
        acq_func=acq_func # "EI" or "PI"
    )
    return result, iteration_values

def optimize_mix_minimize(target_strength_mpa, concrete_type_code, method="SLSQP"):
    """SciPy minimize (CO2) with constraint: strength >= 1.1 * target."""
    bounds = [(b.low, b.high) for b in get_bounds(concrete_type_code)]
    iteration_vals = []

    def objective(mix):
        # clip just in case
        mix = np.clip(mix, [b[0] for b in bounds], [b[1] for b in bounds])
        val = calculate_co2_kg(mix)
        iteration_vals.append(val)
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
    return res, iteration_vals

# =========================
# 4) STREAMLIT APP
# =========================

st.title("Concrete Mix Optimizer (Streamlit)")
st.write("This web application uses a RandomForest model and two optimization methods (Bayesian + SciPy) to find a low-CO₂ concrete mix that meets a target strength.")

# ---------- Input Type ----------
input_type = st.radio("Select Input Type:", ["Target Strength", "Ingredients"])

if input_type == "Target Strength":
    target_strength = st.number_input("Target Strength (MPa)", min_value=0.0, value=30.0)
else:
    st.subheader("Enter Ingredient Values (Kg/m³)")
    ingredient_columns = list(co2_coefficients.keys())  # same order
    user_mix = []
    for ingr in ingredient_columns:
        val = st.number_input(f"{ingr}", min_value=0.0, value=100.0)
        user_mix.append(val)

# ---------- Concrete Type ----------
concrete_type_options = list(strength_ranges.keys())  # ["Normal Strength", ...]
selected_type_name = st.selectbox("Concrete Type:", concrete_type_options)
selected_type_code = strength_ranges[selected_type_name]  # e.g. "NSC"

# ---------- Bayesian acq_func ----------
acq_func = st.selectbox("Bayesian acq_func:", ["EI", "PI"])

# ---------- Minimization Method ----------
method_options = ["SLSQP", "COBYLA", "L-BFGS-B", "Trust-Constr"]
min_method = st.selectbox("Minimize Method:", method_options)

# ---------- Optimize Button ----------
if st.button("Optimize"):
    if input_type == "Target Strength":
        # user gave a target in MPa
        strength_mpa = target_strength
    else:
        # user gave ingredient values, so predict the strength
        strength_mpa = predict_strength(user_mix)
        st.write(f"Predicted Strength of your mix is ~{strength_mpa:.2f} MPa. Using this as the target.")

    # 1) Run Bayesian
    bayes_res, bayes_iters = optimize_mix_bayesian(strength_mpa, selected_type_code, acq_func)
    # 2) Run Minimize
    min_res, min_iters = optimize_mix_minimize(strength_mpa, selected_type_code, min_method)

    # ---------------------
    # Extract Bayesian results
    bayes_mix = bayes_res.x
    bayes_co2 = calculate_co2_kg(bayes_mix)
    bayes_strength = predict_strength(bayes_mix)
    bayes_cost = calculate_cost(bayes_mix)

    # Extract Minimize results
    min_mix = min_res.x
    min_co2 = calculate_co2_kg(min_mix)
    min_strength = predict_strength(min_mix)
    min_cost = calculate_cost(min_mix)

    # ---------------------
    # Display Bayesian results
    st.write("## Bayesian Optimization Result")
    st.write(f"**CO₂ Emissions (kg/m³):** {bayes_co2:.2f}")
    st.write(f"**Strength (MPa):** {bayes_strength:.2f}")
    st.write(f"**Cost ($/m³):** {bayes_cost:.2f}")
    st.write(f"**Mix Proportions (Kg/m³):** {bayes_mix}")

    # Display Minimize results
    st.write("## Minimize Optimization Result")
    st.write(f"**CO₂ Emissions (kg/m³):** {min_co2:.2f}")
    st.write(f"**Strength (MPa):** {min_strength:.2f}")
    st.write(f"**Cost ($/m³):** {min_cost:.2f}")
    st.write(f"**Mix Proportions (Kg/m³):** {min_mix}")

    # Plot iteration values
    fig, ax = plt.subplots()
    ax.plot(bayes_iters, marker='o', label='Bayesian')
    ax.plot(min_iters, marker='x', label='Minimize')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("CO₂ Emissions (kg/m³)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.success("Optimization Complete!")
