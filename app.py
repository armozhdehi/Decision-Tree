import streamlit as st
import pandas as pd
import numpy as np
import math
from fractions import Fraction
from graphviz import Digraph

# Function to calculate entropy for a column and return the formula
def calculate_entropy(column):
    value_counts = column.value_counts()
    total_count = len(column)
    entropy = 0
    entropy_term_strings = []

    for value, count in value_counts.items():
        p = Fraction(count, total_count)
        if p > 0:
            log_p = math.log2(float(p))
            term = float(p) * log_p
            entropy_contribution = -(term)
            entropy_term_strings.append(f"- \\frac{{{p.numerator}}}{{{p.denominator}}} \\log_2\\left(\\frac{{{p.numerator}}}{{{p.denominator}}}\\right)")

            entropy += entropy_contribution

    entropy_formula = " \\\\ ".join(entropy_term_strings)  # Break the formula into multiple lines
    return entropy, entropy_formula

# Function to calculate Gini index for a column
def calculate_gini(column):
    value_counts = column.value_counts()
    total_count = len(column)
    gini = 1.0
    gini_term_strings = []

    for value, count in value_counts.items():
        p = Fraction(count, total_count)
        gini -= float(p) ** 2
        gini_term_strings.append(f"\\left(\\frac{{{p.numerator}}}{{{p.denominator}}}\\right)^2")

    gini_formula = " + \\\\ ".join(gini_term_strings)  # Break the formula into multiple lines
    return gini, gini_formula

# Function to calculate intrinsic value (used for gain ratio)
def calculate_intrinsic_value(data, feature):
    total_count = len(data)
    intrinsic_value = 0
    iv_term_strings = []

    for value in data[feature].unique():
        subset_count = len(data[data[feature] == value])
        weight = Fraction(subset_count, total_count)
        if weight > 0:
            log_weight = math.log2(float(weight))
            term = float(weight) * log_weight
            intrinsic_value -= term
            iv_term_strings.append(f"- \\frac{{{weight.numerator}}}{{{weight.denominator}}} \\log_2\\left(\\frac{{{weight.numerator}}}{{{weight.denominator}}}\\right)")

    iv_formula = " \\\\ ".join(iv_term_strings)  # Break the formula into multiple lines
    return intrinsic_value, iv_formula

# Function to calculate the information gain, gain ratio, or gini index
def calculate_split_criterion(data, feature, target, criterion="Information Gain"):
    total_count = len(data)
    
    if criterion == "Information Gain":
        total_entropy, total_entropy_formula = calculate_entropy(data[target])
        weighted_entropy_total = 0
        weighted_entropy_terms = []

        # Show detailed entropy calculation for the target
        st.latex(f"Entropy({target}) = {total_entropy_formula} = {total_entropy:.4f}")

        for value in data[feature].unique():
            subset = data[data[feature] == value]
            subset_entropy, subset_entropy_formula = calculate_entropy(subset[target])
            weight = len(subset) / total_count
            weighted_entropy_total += weight * subset_entropy
            weighted_entropy_terms.append(f"\\frac{{{len(subset)}}}{{{total_count}}} \\times \\left({subset_entropy_formula}\\right)")

        weighted_entropy_formula = " + \\\\ ".join(weighted_entropy_terms)  # Break the formula into multiple lines
        info_gain = total_entropy - weighted_entropy_total

        # Show the formulas for the weighted entropy and information gain
        st.latex(f"Weighted\\ Entropy({feature}) = {weighted_entropy_formula} = {weighted_entropy_total:.4f}")
        st.latex(f"Information\\ Gain({feature}) = {total_entropy:.4f} - {weighted_entropy_total:.4f} = {info_gain:.4f}")
        return info_gain

    elif criterion == "Gain Ratio":
        # Compute Information Gain first
        info_gain = calculate_split_criterion(data, feature, target, "Information Gain")
        intrinsic_value, iv_formula = calculate_intrinsic_value(data, feature)
        
        if intrinsic_value == 0:
            return 0  # To avoid division by zero
        gain_ratio = info_gain / intrinsic_value

        # Show the formulas for Intrinsic Value and Gain Ratio
        st.latex(f"Intrinsic\\ Value({feature}) = {iv_formula} = {intrinsic_value:.4f}")
        st.latex(f"Gain\\ Ratio({feature}) = \\frac{{{info_gain:.4f}}}{{{intrinsic_value:.4f}}} = {gain_ratio:.4f}")
        return gain_ratio

    elif criterion == "Gini Index":
        total_gini, gini_formula = calculate_gini(data[target])
        weighted_gini_total = 0
        weighted_gini_terms = []

        for value in data[feature].unique():
            subset = data[data[feature] == value]
            subset_gini, subset_gini_formula = calculate_gini(subset[target])
            weight = len(subset) / total_count
            weighted_gini_total += weight * subset_gini
            weighted_gini_terms.append(f"\\frac{{{len(subset)}}}{{{total_count}}} \\times \\left({subset_gini_formula}\\right)")

        weighted_gini_formula = " + \\\\ ".join(weighted_gini_terms)  # Break the formula into multiple lines
        gini_gain = total_gini - weighted_gini_total

        # Show the formulas for the Gini index
        st.latex(f"Gini({target}) = {gini_formula} = {total_gini:.4f}")
        st.latex(f"Weighted\\ Gini({feature}) = {weighted_gini_formula} = {weighted_gini_total:.4f}")
        st.latex(f"Gini\\ Gain({feature}) = {total_gini:.4f} - {weighted_gini_total:.4f} = {gini_gain:.4f}")
        return gini_gain

# Function to recursively build the decision tree
def build_decision_tree(data, target, features, criterion, depth=0, dot=None, node_name="root"):
    if dot is None:
        dot = Digraph(comment="Decision Tree")

    # Stop recursion if the entropy or gini index is 0 (pure node), max depth is reached, or no features left
    if len(data[target].unique()) == 1 or len(features) == 0 or depth == 3:
        class_value = data[target].mode()[0]
        st.write(f"Leaf Node at depth {depth}: All samples belong to class '{class_value}'")
        dot.node(node_name, f"Class: {class_value}", shape='box')
        return dot

    # Select the best feature to split on
    gains = {}
    for feature in features:
        gains[feature] = calculate_split_criterion(data, feature, target, criterion)

    # Check if there are no gains, stop recursion
    if not gains:
        class_value = data[target].mode()[0]
        st.write(f"Leaf Node at depth {depth}: All samples belong to class '{class_value}' (No features left to split on)")
        dot.node(node_name, f"Class: {class_value}", shape='box')
        return dot

    best_feature = max(gains, key=gains.get)
    st.write(f"**Best feature to split on at depth {depth}: {best_feature}**")

    # Remove the selected feature from further splits
    remaining_features = [f for f in features if f != best_feature]

    # Add the node to the graph
    dot.node(node_name, f"Split: {best_feature}")

    # Recurse for each value of the best feature
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        st.write(f"### Splitting on '{best_feature}' with value '{value}' at depth {depth + 1}")

        # Show the subset of data at this split
        st.write(f"**Subset of data for value '{value}' at depth {depth + 1}:**")
        st.dataframe(subset)  # Display the subset of the data that corresponds to this split

        new_node_name = f"{best_feature}_{value}_{depth + 1}"
        dot.edge(node_name, new_node_name, label=str(value))
        build_decision_tree(subset, target, remaining_features, criterion, depth + 1, dot, new_node_name)

    return dot

# Streamlit UI
st.title("Recursive Decision Tree Constructor with Detailed Calculations (Information Gain, Gain Ratio, Gini Index)")
st.write("This app recursively constructs a decision tree based on different split criteria and displays detailed calculations at each level.")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Here is a preview of your dataset:")
    st.write(df)

    # Step 2: Select the target column (Y) and features
    column_names = df.columns.tolist()
    target_col = st.selectbox("Select the target column", options=column_names)
    feature_cols = st.multiselect("Select the feature columns", options=[col for col in column_names if col != target_col])

    # Step 3: Select the criterion (Information Gain, Gain Ratio, Gini Index)
    criterion = st.selectbox("Select the criterion", options=["Information Gain", "Gain Ratio", "Gini Index"])

    if feature_cols:
        # Step 4: Build and display the decision tree
        st.write(f"### Building the decision tree using {criterion} criterion...")
        decision_tree_graph = build_decision_tree(df, target_col, feature_cols, criterion)

        # Render the graph as an image
        st.graphviz_chart(decision_tree_graph.source)
    else:
        st.write("Please select at least one feature column.")
