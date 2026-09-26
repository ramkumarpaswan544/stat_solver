import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import rpy2.robjects as robjects
from rpy2.robjects import conversion, default_converter

st.set_page_config(page_title="Advanced Stat AI", layout="wide")

# 1. Load the external R engine
with open("engine.R", "r") as file:
    r_code = file.read()

with conversion.localconverter(default_converter):
    robjects.r(r_code)
    run_t_test = robjects.globalenv['run_t_test']
    calc_binom = robjects.globalenv['calc_binom_prob']
    calc_norm = robjects.globalenv['calc_norm_prob']
    run_reg = robjects.globalenv['run_simple_regression']

# 2. Sidebar Navigation
st.sidebar.title("Stat Stream Modules")
app_mode = st.sidebar.radio("Select Engine:", [
    "Inference (Hypothesis Testing)", 
    "Probability Distributions",
    "Simple Linear Regression",
    "Theorem & Calculus Engine"
])

# 3. Module 1: Hypothesis Testing
if app_mode == "Inference (Hypothesis Testing)":
    st.title("📊 The Inference Engine")
    st.header("One-Sample Hypothesis Testing")
    user_input = st.text_input("Enter sample data (comma-separated):", "12, 18, 22, 28, 35")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        mu_input = st.number_input("Null Hypothesis Mean (μ₀):", value=20.0)
    with col2:
        conf_input = st.slider("Confidence Level (1 - α):", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
    with col3:
        alt_input = st.selectbox("Alternative Hypothesis:", ["two.sided", "less", "greater"])
        
    if st.button("Run Exact T-Test"):
        try:
            data_list = [float(x.strip()) for x in user_input.split(',')]
            with conversion.localconverter(default_converter):
                r_vector = robjects.FloatVector(data_list)
                with st.spinner("Computing via local R engine..."):
                    r_result = run_t_test(r_vector, mu_val=mu_input, conf_level=conf_input, alt=alt_input)
                    
                    st.success("Test executed flawlessly!")
                    st.markdown(f"**Test Statistic ($t$):** {r_result.rx2('statistic')[0]:.4f}")
                    st.markdown(f"**Degrees of Freedom ($df$):** {r_result.rx2('df')[0]}")
                    st.markdown(f"**$p$-value:** {r_result.rx2('p_value')[0]:.5f}")
                    st.markdown(f"**Sample Mean ($\hat{{\mu}}$):** {r_result.rx2('mean_est')[0]:.4f}")
                    st.markdown(f"**{int(conf_input*100)}% Confidence Interval:** [{r_result.rx2('conf_lower')[0]:.4f}, {r_result.rx2('conf_upper')[0]:.4f}]")
        except Exception as e:
            st.error(f"Error calculating test: {e}")

# 4. Module 2: Probability Distributions
elif app_mode == "Probability Distributions":
    st.title("📈 The Distribution Engine")
    dist_type = st.selectbox("Select Distribution Family:", ["Binomial (Discrete)", "Normal (Continuous)"])
    
    if dist_type == "Binomial (Discrete)":
        st.subheader("Binomial Probability: $X \sim B(n, p)$")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_val = st.number_input("Number of Trials ($n$):", min_value=1, value=10)
        with col2:
            p_val = st.number_input("Probability of Success ($p$):", min_value=0.0, max_value=1.0, value=0.5)
        with col3:
            k_val = st.number_input("Target Successes ($k$):", min_value=0, max_value=n_val, value=5)
            
        tail = st.radio("Direction:", ["$P(X \le k)$ (Lower Tail)", "$P(X > k)$ (Upper Tail)"])
        is_lower = True if "Lower" in tail else False
        
        if st.button("Calculate Exact Probability"):
            with conversion.localconverter(default_converter):
                prob = calc_binom(k_val, n_val, p_val, is_lower)[0]
                st.success(f"**Result:** {prob:.5f}")

    elif dist_type == "Normal (Continuous)":
        st.subheader("Normal Probability: $X \sim N(\mu, \sigma^2)$")
        col1, col2, col3 = st.columns(3)
        with col1:
            mu_val = st.number_input("Mean ($\mu$):", value=0.0)
        with col2:
            sd_val = st.number_input("Standard Deviation ($\sigma$):", min_value=0.01, value=1.0)
        with col3:
            x_val = st.number_input("Target Value ($x$):", value=1.96)
            
        tail = st.radio("Direction:", ["$P(X \le x)$ (Lower Tail)", "$P(X > x)$ (Upper Tail)"])
        is_lower = True if "Lower" in tail else False
        
        if st.button("Calculate Exact Probability"):
            with conversion.localconverter(default_converter):
                prob = calc_norm(x_val, mu_val, sd_val, is_lower)[0]
                st.success(f"**Result:** {prob:.5f}")

# 5. Module 3: Linear Regression
elif app_mode == "Simple Linear Regression":
    st.title("📉 The Regression Engine")
    st.write("Upload a CSV file to fit an exact Ordinary Least Squares (OLS) model.")
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write("### Configure Model")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select Independent Variable ($X$):", df.columns)
        with col2:
            y_col = st.selectbox("Select Dependent Variable ($Y$):", df.columns)
            
        if st.button("Compute Exact OLS Model"):
            try:
                x_data = df[x_col].dropna().tolist()
                y_data = df[y_col].dropna().tolist()
                
                if len(x_data) != len(y_data):
                    st.error("Error: X and Y variables must have the same number of valid rows.")
                else:
                    with conversion.localconverter(default_converter):
                        r_x = robjects.FloatVector(x_data)
                        r_y = robjects.FloatVector(y_data)
                        
                        with st.spinner("Fitting model in local R environment..."):
                            res = run_reg(r_x, r_y)
                            
                            intercept = res.rx2('intercept')[0]
                            slope = res.rx2('slope')[0]
                            
                            st.success("Model fitted flawlessly!")
                            
                            col_res1, col_res2 = st.columns(2)
                            with col_res1:
                                st.markdown(f"**Intercept ($\\beta_0$):** {intercept:.4f}")
                                st.markdown(f"**Slope ($\\beta_1$):** {slope:.4f}")
                            with col_res2:
                                st.markdown(f"**Coefficient of Determination ($R^2$):** {res.rx2('r_squared')[0]:.4f}")
                                st.markdown(f"**$p$-value (for slope):** {res.rx2('p_value')[0]:.4e}")
                            
                            st.divider()
                            st.write("### OLS Regression Fit")
                            fig, ax = plt.subplots(figsize=(8, 5))
                            
                            ax.scatter(x_data, y_data, color='#1f77b4', alpha=0.7, label='Observed Data')
                            x_array = np.array(x_data)
                            y_pred = intercept + slope * x_array
                            ax.plot(x_array, y_pred, color='#d62728', linewidth=2.5, label='Fitted Line')
                            
                            ax.set_xlabel(x_col)
                            ax.set_ylabel(y_col)
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.5)
                            
                            st.pyplot(fig)
                            
            except Exception as e:
                st.error(f"Error computing regression: {e}")

# 6. Module 4: Theorem & Calculus Engine
elif app_mode == "Theorem & Calculus Engine":
    st.title("🧮 Theorem & Calculus Engine")
    st.write("Solve theoretical statistics problems using symbolic mathematics.")
    
    calc_type = st.selectbox("Select Operation:", ["Differentiation (MLEs, Gradients)", "Integration (Continuous Probabilities)"])
    x = sp.Symbol('x')
    
    if calc_type == "Differentiation (MLEs, Gradients)":
        st.subheader("Symbolic Differentiation")
        expr_input = st.text_input("Enter mathematical expression (in terms of x):", "x**2 * exp(-x)")
        
        if st.button("Compute Derivative"):
            try:
                expr = sp.sympify(expr_input)
                derivative = sp.diff(expr, x)
                st.success("Derivative computed successfully!")
                st.latex(rf"\frac{{d}}{{dx}} \left( {sp.latex(expr)} \right) = {sp.latex(derivative)}")
            except Exception as e:
                st.error(f"Error parsing expression: {e}")
                
    elif calc_type == "Integration (Continuous Probabilities)":
        st.subheader("Definite Integration")
        expr_input = st.text_input("Enter Probability Density Function (in terms of x):", "exp(-x)")
        
        col1, col2 = st.columns(2)
        with col1:
            lower_limit = st.text_input("Lower Limit:", "0")
        with col2:
            upper_limit = st.text_input("Upper Limit (use 'oo' for infinity):", "oo")
            
        if st.button("Compute Definite Integral"):
            try:
                expr = sp.sympify(expr_input)
                lower = sp.sympify(lower_limit)
                upper = sp.sympify(upper_limit)
                
                integral = sp.integrate(expr, (x, lower, upper))
                st.success("Integral computed successfully!")
                st.latex(rf"\int_{{{sp.latex(lower)}}}^{{{sp.latex(upper)}}} \left( {sp.latex(expr)} \right) dx = {sp.latex(integral)}")
            except Exception as e:
                st.error(f"Error computing integral: {e}")