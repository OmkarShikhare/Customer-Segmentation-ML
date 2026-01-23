import streamlit as st
import pandas as pd
import pickle


# Load Model & Scaler

kmeans = pickle.load(open("kmeans_cluster_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Cluster label mapping
cluster_labels = {
    0: "Regular Customers",
    1: "Price-Sensitive Customers",
    2: "High-Value Customers"}


# App Title

st.set_page_config(page_title="Customer Segmentation App")
st.title(" Customer Segmentation System")
st.write("Predict customer segment using ML clustering")


# Sidebar Inputs

st.sidebar.header("Enter Customer Details")

income = st.sidebar.number_input("Income", min_value=0.0, value=50000.0)
recency = st.sidebar.number_input("Recency (days)", min_value=0, value=30)
web = st.sidebar.number_input("Web Purchases", min_value=0, value=5)
store = st.sidebar.number_input("Store Purchases", min_value=0, value=6)
spending = st.sidebar.number_input("Total Spending", min_value=0.0, value=1000.0)


# Prediction

if st.sidebar.button("Predict Segment"):
    input_data = [[income, recency, web, store, spending]]
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]
    segment = cluster_labels[cluster]

    st.success(f" Customer Segment: **{segment}**")


# Show Segmented Dataset

st.subheader(" Segmented Customer Data")

df = pd.read_csv("clean_clusters.csv")

# Feature engineering (same as training)
df["Total_Spending"] = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] +
                        df["MntGoldProds"])

X = df[["Income","Recency","NumWebPurchases",
    "NumStorePurchases","Total_Spending"]]

X_scaled = scaler.transform(X)
df["Cluster"] = kmeans.predict(X_scaled)
df["Customer_Segment"] = df["Cluster"].map(cluster_labels)

st.dataframe(df[["Income","Recency","Total_Spending",
        "NumWebPurchases","NumStorePurchases",
        "Customer_Segment"]].head(20))
