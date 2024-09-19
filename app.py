import streamlit as st
import pandas as pd
import numpy as np
import math
from fractions import Fraction

# Function to calculate entropy and show intermediate steps with fractions
def calculate_entropy(column, show_steps=False):
    value_counts = column.value_counts()
    total_count = len(column)
    probabilities = value_counts / total_count

    if show_steps:
        st.write(f"Counts for each class: {value_counts.to_dict()}")
        # Display probabilities as fractions
        probabilities_fraction = {value: Fraction(count, total_count) for value, count in value_counts.items()}
        st.write(f"Probabilities (fractions): {probabilities_fraction}")

    # Calculate entropy using fractions
    entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])

    if show_steps:
        for value, count in value_counts.items():
            p = Fraction(count, total_count)
            if p > 0:
                entropy_contribution = -(p * math.log2(float(p)))
                st.write(f"- ({p} * log2({float(p):.4f})) = {entropy_contribution:.4f}")
        st.write(f"Final entropy: {entropy:.4f}")
    
    return entropy

# Function to calculate information gain and show step-by-step with fractions
def information_gain(data, feature, target):
    st.write(f"### Calculating Information Gain for feature '{feature}'")
    
    # Step 1: Calculate total entropy for the target column
    st.write(f"**Step 1: Calculate Entropy of the target variable ({target})**")
    total_entropy = calculate_entropy(data[target], show_steps=True)
    st.write(f"Entropy(S) = {total_entropy:.4f}")
    
    # Step 2: Calculate weighted entropy for each value of the feature
    st.write(f"**Step 2: Calculate the weighted entropy for each value of the feature ({feature})**")
    values = data[feature].unique()
    weighted_entropy = 0
    total_count = len(data)
    
    for value in values:
        subset = data[data[feature] == value][target]
        subset_entropy = calculate_entropy(subset, show_steps=True)
        weight_fraction = Fraction(len(subset), total_count)  # Use fraction for weight
        weighted_entropy += float(weight_fraction) * subset_entropy
        
        st.write(f"For value '{value}':")
        st.write(f"- Subset entropy = {subset_entropy:.4f}")
        st.write(f"- Weight (fraction) = {weight_fraction}")
        st.write(f"- Contribution to weighted entropy = {weight_fraction} * {subset_entropy:.4f} = {float(weight_fraction) * subset_entropy:.4f}")
    
    st.write(f"Weighted Entropy of {feature} = {weighted_entropy:.4f}")
    
    # Step 3: Calculate Information Gain
    st.write(f"**Step 3: Calculate Information Gain**")
    gain = total_entropy - weighted_entropy
    st.write(f"Information Gain for {feature} = {total_entropy:.4f} - {weighted_entropy:.4f} = {gain:.4f}")
    
    return gain

# Streamlit UI
st.title("Step-by-Step Decision Tree Constructor (Information Gain) with Fractions")
st.write("This app walks through each step of constructing a decision tree using information gain. Upload your CSV dataset and follow along.")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Here is a preview of your dataset:")
    st.write(df)

    # Step 2: Select the target column (Y) and features
    column_names = df.columns.tolist()
    
    target_col = st.selectbox("Select the target column", options=column_names)
    
    # Allow the user to select which columns to use as features
    feature_cols = st.multiselect("Select the feature columns", options=[col for col in column_names if col != target_col])
    
    if len(feature_cols) > 0:
        # Step 3: Compute Information Gain for each feature
        st.write("### Calculating Information Gain for each feature")
        gains = {}
        for col in feature_cols:
            gain = information_gain(df, col, target_col)
            gains[col] = gain

        # Step 4: Display best feature for split
        best_feature = max(gains, key=gains.get)
        st.write(f"**The best feature to split on is: {best_feature}**")
    else:
        st.write("Please select at least one feature column.")
