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

# 1) MUST be first Streamlit command
st.set_page_config(page_title="PCA - Carboon Emission Concrete Mix Optimizer", layout="wide")

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
# 3) DICTIONARIES & FUNCTIONS
# -----------------------
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

# Conversions
KG_TO_LB = 2.20462
MPA_TO_PSI = 145.038
KG_PER_M3_TO_LB_PER_FT3 = 0.06242796
M3_TO_FT3 = 35.3146667

strength_ranges = {
    "Normal Strength": "NSC",
    "High Strength": "HSC",
    "Ultra-High performance": "UHPC"
}

# --- NEW user-facing vs. internal codes for Bayesian & Minimize ---

# For Bayesian acquisition functions
bayesian_ui_dict = {
    "Expected Improvement (EI)": "EI",
    "Probability of Improvement (PI)": "PI",
}

# For SciPy minimize methods
minimize_ui_dict = {
    "SLSQP (Sequential Least Squares)": "SLSQP",
    "COBYLA (Constrained Optimization)": "COBYLA",
    "L-BFGS-B (Limited-memory BFGS)": "L-BFGS-B",
    "Trust-Constr (Trust Region)": "trust-constr",
}

def get_bounds(concrete_type):
    if concrete_type == "NSC":
        return [
            Real(140, 400, name="Cement (Kg/m3)"),
            Real(0, 150,   name="Blast Furnace Slag (Kg/m3)"),
            Real(0, 1,     name="Silica Fume (Kg/m3)"),
            Real(0, 100,   name="Fly Ash (Kg/m3)"),
            Real(120, 200, name="Water (Kg/m3)"),
            Real(800, 1200,name="Coarse Aggregate (Kg/m3)"),
            Real(600, 700, name="Fine Aggregate (Kg/m3)"),
        ]
    elif concrete_type == "HSC":
        return [
            Real(240, 550, name="Cement (Kg/m3)"),
            Real(0, 150,   name="Blast Furnace Slag (Kg/m3)"),
            Real(0, 50,    name="Silica Fume (Kg/m3)"),
            Real(0, 150,   name="Fly Ash (Kg/m3)"),
            Real(105, 160, name="Water (Kg/m3)"),
            Real(700, 1000,name="Coarse Aggregate (Kg/m3)"),
            Real(600, 800, name="Fine Aggregate (Kg/m3)"),
        ]
    else:  # UHPC
        return [
            Real(350, 1000,name="Cement (Kg/m3)"),
            Real(0, 150,   name="Blast Furnace Slag (Kg/m3)"),
            Real(140, 300, name="Silica Fume (Kg/m3)"),
            Real(0, 200,   name="Fly Ash (Kg/m3)"),
            Real(125, 150, name="Water (Kg/m3)"),
            Real(0, 1,     name="Coarse Aggregate (Kg/m3)"),
            Real(650, 1200,name="Fine Aggregate (Kg/m3)"),
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
    return total_cost  # in $/m³

# Bayesian
def optimize_mix_bayesian(target_strength_mpa, concrete_type, acq_func="EI"):
    bounds = get_bounds(concrete_type)
    iteration_values = []

    @use_named_args(bounds)
    def objective(**params):
        mix = list(params.values())
        strength = predict_strength(mix)
        # Add penalty if strength is below 110% of target
        penalty = max(0, (1.0 * target_strength_mpa - strength)) ** 2
        val = calculate_co2_kg(mix) + 10 * penalty
        iteration_values.append(val)
        return val

    res = gp_minimize(
        objective,
        dimensions=bounds,
        n_calls=20,
        random_state=42,
        acq_func=acq_func
    )
    return res, iteration_values

# SciPy
def optimize_mix_minimize(target_strength_mpa, concrete_type, method="SLSQP"):
    """
    method="SLSQP" is just the fallback default if none is provided.
    The actual method used is determined by 'method_code' that we pass in
    from the Streamlit selectbox.
    """
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
        constraints=cons,
        options={"maxiter": 800}  # limit # of iterations

    )
    return res, iteration_values


# 4) BUILD THE STREAMLIT APP
# --------------------------------
# Fancy Header
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

# Let user pick "Target Strength" or "Ingredients"
input_type = st.radio("Select Input Type:", ["Target Strength", "Ingredients"], index=0)

# Main input area
if input_type == "Target Strength":
    # If user picks target strength
    if unit_system == "Metric":
        target_strength_val = st.number_input("Target Strength (MPa)", min_value=0.0, value=30.0)
    else:
        target_strength_val = st.number_input("Target Strength (psi)", min_value=0.0, value=4351.0)  # ~30 MPa
else:
    st.subheader("Provide Ingredient Amounts")
    user_mix = []
    for col_key in co2_coefficients.keys():
        if unit_system == "Metric":
            val = st.number_input(f"{cleaned_names[col_key]} (Kg/m³)", min_value=0.0, value=100.0)
            user_mix.append(val)
        else:
            val = st.number_input(f"{cleaned_names[col_key]} (lb/ft³)", min_value=0.0, value=10.0)
            # convert to metric
            val_metric = val / KG_PER_M3_TO_LB_PER_FT3
            user_mix.append(val_metric)

st.write("---")

# Concrete Type
conc_type_name = st.selectbox("Concrete Type:", list(strength_ranges.keys()), index=0)
conc_type_code = strength_ranges[conc_type_name]

# Bayesian acq_func selection (user-friendly)
acq_display = st.selectbox(
    "Bayesian Acquisition Function:",
    list(bayesian_ui_dict.keys()),
    index=0  # default to "Expected Improvement (EI)"
)
acq_func = bayesian_ui_dict[acq_display]  # actual code "EI" or "PI"

# Minimize method selection (user-friendly)
method_display = st.selectbox(
    "Minimize Method:",
    list(minimize_ui_dict.keys()),
    index=3  # default to "Trust-Constr (Trust Region)"
)
method_code = minimize_ui_dict[method_display]  # actual code "SLSQP", etc.

if st.button("Optimize"):
    # Convert to MPa if user is in US
    if input_type == "Target Strength":
        if unit_system == "Metric":
            target_strength_mpa = target_strength_val
        else:
            target_strength_mpa = target_strength_val / MPA_TO_PSI
    else:
        # user typed a custom mix, let's predict the strength in MPa
        predicted_strength = predict_strength(user_mix)
        target_strength_mpa = predicted_strength
        st.write(f"**Predicted Strength** from your ingredients: {predicted_strength:.2f} MPa")

    # 1) Bayesian
    res_bayes, iter_bayes = optimize_mix_bayesian(target_strength_mpa, conc_type_code, acq_func)
    # 2) Minimize
    res_min, iter_min = optimize_mix_minimize(target_strength_mpa, conc_type_code, method_code)

    # Bayesian results
    bayes_mix = res_bayes.x
    bayes_co2 = calculate_co2_kg(bayes_mix)
    bayes_str = predict_strength(bayes_mix)
    bayes_cost = calculate_cost(bayes_mix)

    # Minimize results
    min_mix = res_min.x
    min_co2 = calculate_co2_kg(min_mix)
    min_str = predict_strength(min_mix)
    min_cost = calculate_cost(min_mix)

    # Display side-by-side
    col1, col2 = st.columns(2)

    # ----------------------
    #  BAYESIAN RESULTS
    # ----------------------
    with col1:
        st.markdown("### Bayesian Optimization")
        if unit_system == "Metric":
            co2_disp = bayes_co2
            str_disp = bayes_str
            cost_disp = bayes_cost
            co2_label = "kg/m³"
            str_label = "MPa"
            cost_label = "$/m³"
            mix_unit = "Kg/m³"
        else:
            co2_disp = bayes_co2 * KG_TO_LB
            str_disp = bayes_str * MPA_TO_PSI
            cost_disp = bayes_cost / M3_TO_FT3
            co2_label = "lb/ft³"
            str_label = "psi"
            cost_label = "$/ft³"
            mix_unit = "lb/ft³"

        st.write(f"**CO₂ Emissions:** {co2_disp:.2f} {co2_label}")
        st.write(f"**Strength:** {str_disp:.2f} {str_label}")
        st.write(f"**Cost:** ${cost_disp:.2f} per {cost_label}")
        st.write("**Mix Proportions:**")
        for k, val_metric in zip(co2_coefficients.keys(), bayes_mix):
            if unit_system == "Metric":
                final_val = val_metric
            else:
                final_val = val_metric * KG_PER_M3_TO_LB_PER_FT3
            st.write(f"- {cleaned_names[k]}: {final_val:.2f} {mix_unit}")

    # ----------------------
    #  MINIMIZE RESULTS
    # ----------------------
    with col2:
        st.markdown("### Minimize Optimization")
        if unit_system == "Metric":
            co2_disp = min_co2
            str_disp = min_str
            cost_disp = min_cost
            co2_label = "kg/m³"
            str_label = "MPa"
            cost_label = "$/m³"
            mix_unit = "Kg/m³"
        else:
            co2_disp = min_co2 * KG_TO_LB
            str_disp = min_str * MPA_TO_PSI
            cost_disp = min_cost / M3_TO_FT3
            co2_label = "lb/ft³"
            str_label = "psi"
            cost_label = "$/ft³"
            mix_unit = "lb/ft³"

        st.write(f"**CO₂ Emissions:** {co2_disp:.2f} {co2_label}")
        st.write(f"**Strength:** {str_disp:.2f} {str_label}")
        st.write(f"**Cost:** ${cost_disp:.2f} per {cost_label}")
        st.write("**Mix Proportions:**")
        for k, val_metric in zip(co2_coefficients.keys(), min_mix):
            if unit_system == "Metric":
                final_val = val_metric
            else:
                final_val = val_metric * KG_PER_M3_TO_LB_PER_FT3
            st.write(f"- {cleaned_names[k]}: {final_val:.2f} {mix_unit}")

    st.write("---")
    st.markdown("### Iteration Values")

    # Helper to convert CO2 to lb if US system
    def co2_for_plot(c):
        return c if unit_system == "Metric" else c * KG_TO_LB

    # Plot Bayesian iteration values
    fig_bayes, ax_bayes = plt.subplots()
    bayes_iter_values = [co2_for_plot(v) for v in iter_bayes]
    ax_bayes.plot(range(1, len(bayes_iter_values) + 1), bayes_iter_values, marker='o', label='Bayesian')
    ax_bayes.set_xlabel("Iteration")
    ylab_bayes = "CO₂ (kg)" if unit_system == "Metric" else "CO₂ (lb)"
    ax_bayes.set_ylabel(ylab_bayes)
    ax_bayes.set_title("Bayesian Iterations")
    ax_bayes.grid(True)
    st.pyplot(fig_bayes)

    # Plot Minimize iteration values
    fig_min, ax_min = plt.subplots()
    min_iter_values = [co2_for_plot(v) for v in iter_min]
    ax_min.plot(range(1, len(min_iter_values) + 1), min_iter_values, marker='x', label='Minimize')
    ax_min.set_xlabel("Iteration")
    ylab_min = "CO₂ (kg)" if unit_system == "Metric" else "CO₂ (lb)"
    ax_min.set_ylabel(ylab_min)
    ax_min.set_title("Minimize Iterations")
    ax_min.grid(True)
    st.pyplot(fig_min)

    st.success("Optimization Complete!")
