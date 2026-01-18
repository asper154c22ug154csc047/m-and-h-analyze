import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import io

# --- Page Config ---
st.set_page_config(page_title="Human Stress Analysis", layout="wide", page_icon="🧠")

# --- Custom UI Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Human Stress Level Clustering Analysis")
st.markdown("Comparing **K-Means**, **K-Medoids**, and Density/Grid-based algorithms.")

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        # Load the CSV file provided in the directory
        df = pd.read_csv("country_wise_latest.csv")
    except:
        st.error("Dataset 'country_wise_latest.csv' not found in repository!")
        st.stop()
    
    # Filter needed columns
    df = df[['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active']].dropna()
    
    # Stress Index Logic
    df['Stress_Index'] = (df['Confirmed'] * 0.4) + (df['Deaths'] * 0.4) + (df['Active'] * 0.2)
    return df

df_data = load_data()

# --- Sidebar ---
st.sidebar.header("Configuration")
algo_choice = st.sidebar.selectbox(
    "Select Algorithm",
    ("K-Means", "K-Medoids", "DBSCAN", "OPTICS", "HDBSCAN", "Gaussian Mixture", "BIRCH")
)

# --- Preprocessing ---
scaler = MinMaxScaler()
cols = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Stress_Index']
df_scaled = scaler.fit_transform(df_data[cols])

# --- Clustering Logic ---
if algo_choice in ["K-Means", "K-Medoids", "Gaussian Mixture", "BIRCH"]:
    k = st.sidebar.slider("Number of Clusters", 2, 5, 3)

if algo_choice == "K-Means":
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(df_scaled)
elif algo_choice == "K-Medoids":
    model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(df_scaled)
elif algo_choice == "DBSCAN":
    model = DBSCAN(eps=0.1, min_samples=2)
    labels = model.fit_predict(df_scaled)
elif algo_choice == "OPTICS":
    model = OPTICS(min_samples=3)
    labels = model.fit_predict(df_scaled)
elif algo_choice == "HDBSCAN":
    model = HDBSCAN(min_cluster_size=2)
    labels = model.fit_predict(df_scaled)
elif algo_choice == "Gaussian Mixture":
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(df_scaled)
elif algo_choice == "BIRCH":
    model = Birch(n_clusters=k)
    labels = model.fit_predict(df_scaled)

df_data['Cluster'] = labels.astype(str)

# Map labels to Stress Levels based on Cluster Mean
means = df_data.groupby('Cluster')['Stress_Index'].mean().sort_values()
stress_map = {cluster: "Low Stress" for cluster in means.index}
if len(means) > 1:
    stress_map[means.index[-1]] = "High Stress"
if len(means) > 2:
    stress_map[means.index[len(means)//2]] = "Medium Stress"

df_data['Stress_Level'] = df_data['Cluster'].map(stress_map)

# --- Layout ---
c1, c2 = st.columns([3, 1])

with c1:
    fig = px.scatter(
        df_data, x="Confirmed", y="Deaths", color="Stress_Level",
        size="Stress_Index", hover_name="Country/Region",
        title=f"Clustering Result: {algo_choice}",
        template="plotly_white",
        color_discrete_map={"Low Stress": "green", "Medium Stress": "orange", "High Stress": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Metrics")
    try:
        score = silhouette_score(df_scaled, labels)
        st.metric("Silhouette Score", f"{score:.3f}")
    except:
        st.write("Score N/A")

st.divider()
st.subheader("Data Analysis Table")
st.dataframe(df_data[['Country/Region', 'Stress_Index', 'Stress_Level']].sort_values(by="Stress_Index", ascending=False), use_container_width=True)
