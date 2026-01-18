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

# --- Professional Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- Title Section ---
st.title("🧠 Human Stress Level Clustering Analysis")
st.markdown("""
**Research Objective:** To categorize human stress levels based on environmental demands using 
various clustering techniques. This index uses COVID-19 impact data as a proxy for societal pressure.
""")

# --- Data Loading ---
@st.cache_data
def get_dataframe():
    # Attempt to load local file, fallback to internal sample if missing
    try:
        df = pd.read_csv("country_wise_latest.csv")
    except:
        # Mini sample of your data for emergency fallback
        data = """Country/Region,Confirmed,Deaths,Recovered,Active
Afghanistan,36263,1269,25198,9796
Albania,4880,144,2745,1991
Algeria,27973,1163,18837,7973
Argentina,167416,3059,72575,91782
Brazil,2442375,87618,1846641,508116
US,4290259,148011,1325804,2816444
India,1480073,33408,951166,495499
UK,301708,45844,1437,254427
"""
        df = pd.read_csv(io.StringIO(data))
    
    # Preprocessing
    df = df[['Country/Region', 'Confirmed', 'Deaths', 'Recovered', 'Active']].dropna()
    
    # Stress Index Formula: (Confirmed*0.4) + (Deaths*0.4) + (Active*0.2)
    df['Stress_Index'] = (df['Confirmed'] * 0.4) + (df['Deaths'] * 0.4) + (df['Active'] * 0.2)
    return df

df = get_dataframe()

# --- Sidebar ---
st.sidebar.header("🕹️ Control Panel")
st.sidebar.subheader("Algorithm Selection")
algo = st.sidebar.selectbox(
    "Choose Algorithm",
    ("K-Means", "K-Medoids", "DBSCAN", "OPTICS", "HDBSCAN", "Gaussian Mixture", "BIRCH (Grid-based)")
)

if algo in ["K-Means", "K-Medoids", "Gaussian Mixture", "BIRCH (Grid-based)"]:
    k = st.sidebar.slider("Number of Clusters (k)", 2, 6, 3)
else:
    st.sidebar.info("This algorithm determines cluster count automatically.")

# --- Data Preparation ---
scaler = MinMaxScaler()
features = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Stress_Index']
df_scaled = scaler.fit_transform(df[features])

# --- Clustering Execution ---
model_name = algo
labels = None

if algo == "K-Means":
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(df_scaled)
elif algo == "K-Medoids":
    model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(df_scaled)
elif algo == "DBSCAN":
    model = DBSCAN(eps=0.15, min_samples=2)
    labels = model.fit_predict(df_scaled)
elif algo == "OPTICS":
    model = OPTICS(min_samples=3)
    labels = model.fit_predict(df_scaled)
elif algo == "HDBSCAN":
    model = HDBSCAN(min_cluster_size=2)
    labels = model.fit_predict(df_scaled)
elif algo == "Gaussian Mixture":
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(df_scaled)
elif algo == "BIRCH (Grid-based)":
    model = Birch(n_clusters=k)
    labels = model.fit_predict(df_scaled)

df['Cluster'] = labels.astype(str)

# --- Logic: Auto-Label Stress Levels ---
# Calculate mean stress index per cluster to rank them
rank = df.groupby('Cluster')['Stress_Index'].mean().sort_values().index
label_map = {rank[i]: ["Low Stress", "Medium Stress", "High Stress", "Extreme"][min(i, 3)] for i in range(len(rank))}
df['Stress_Level'] = df['Cluster'].map(label_map)

# --- Visualization Section ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"Results: {algo} Analysis")
    fig = px.scatter(
        df, x="Confirmed", y="Deaths", color="Stress_Level",
        size="Stress_Index", hover_name="Country/Region",
        color_discrete_map={
            "Low Stress": "#22C55E",
            "Medium Stress": "#EAB308",
            "High Stress": "#EF4444",
            "Extreme": "#7F1D1D"
        },
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Efficiency")
    try:
        score = silhouette_score(df_scaled, labels)
        st.metric("Silhouette Score", f"{score:.3f}")
    except:
        st.write("Score N/A")
    
    st.write("**Quick Stats:**")
    st.write(f"Total Countries: {len(df)}")
    st.write(f"Detected Clusters: {len(df['Cluster'].unique())}")

# --- Data Table ---
st.divider()
st.subheader("Detailed stress Assessment Report")
st.dataframe(
    df[['Country/Region', 'Stress_Index', 'Stress_Level', 'Cluster']].sort_values("Stress_Index", ascending=False),
    use_container_width=True
)

# --- Download ---
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Analysis Report", data=csv, file_name="stress_analysis.csv", mime="text/csv")
