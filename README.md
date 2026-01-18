📊 Human Stress Level Clustering Analysis
Perbandingan K-Means & K-Medoids untuk Klastering Tingkat Stress
![alt text](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)

![alt text](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)

![alt text](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?logo=scikit-learn)

![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)
📌 Project Overview
This research-based application addresses the challenge of managing and understanding human stress levels caused by societal demands. By utilizing COVID-19 global statistics as a proxy for environmental stressors, the system evaluates and compares multiple clustering techniques to categorize stress into Low, Medium, and High levels.
The Problem
Societal demands and unmet expectations are primary drivers of human stress. This tool provides an automated way to cluster these stressors to improve management strategies and decision-making.
🧪 Methodology & Stress Index
The core of this analysis is the Stress Index (SI), a weighted formula designed to quantify the impact of environmental factors:
S
t
r
e
s
s
I
n
d
e
x
=
(
C
o
n
f
i
r
m
e
d
×
0.4
)
+
(
D
e
a
t
h
s
×
0.4
)
+
(
A
c
t
i
v
e
×
0.2
)
StressIndex=(Confirmed×0.4)+(Deaths×0.4)+(Active×0.2)
Supported Algorithms
The application provides a comparative environment for several clustering paradigms:
Centroid-based: K-Means (Baseline) vs. K-Medoids (Robust to outliers).
Density-based: DBSCAN & OPTICS (Identifies noise).
Hierarchical: HDBSCAN.
Distribution-based: Gaussian Mixture Models (GMM).
Grid-based: BIRCH.
✨ Key Features
Interactive Dashboard: A professional UI built with Streamlit for real-time parameter tuning.
Comparative Metrics: Automatically calculates Silhouette Scores to identify the most efficient algorithm for the dataset.
Dynamic Visualizations: High-quality, interactive scatter plots and histograms using Plotly.
Data Management: Normalization via MinMaxScaler and automated stress-level mapping.
Exportable Results: Download the clustered results directly as a CSV file.
🚀 Getting Started
Prerequisites
Python 3.8 or higher
Pip (Python package manager)
Installation
Clone the repository:
code
Bash
git clone https://github.com/yourusername/stress-clustering-analysis.git
cd stress-clustering-analysis
Install dependencies:
code
Bash
pip install streamlit pandas numpy scikit-learn scikit-learn-extra plotly hdbscan
Run the Application:
code
Bash
streamlit run app.py
📂 Project Structure
code
Text
├── app.py                   # Main Streamlit application
├── country_wise_latest.csv  # Dataset file
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
📊 Evaluation
The application evaluates models based on:
Silhouette Coefficient: Measuring how tightly grouped the clusters are.
Distribution Analysis: Analyzing the spread of "High Stress" vs "Low Stress" regions.
Algorithm Robustness: Comparing how K-Medoids handles outliers differently than K-Means.
📷 Screenshots
(Add your application screenshots here to make the README more attractive)
🤝 Contributing
Contributions are welcome! If you have suggestions for new algorithms or improved stress index formulas:
Fork the Project
Create your Feature Branch (git checkout -b feature/NewAlgorithm)
Commit your Changes (git commit -m 'Add some NewAlgorithm')
Push to the Branch (git push origin feature/NewAlgorithm)
Open a Pull Request
📄 License
Distributed under the MIT License. See LICENSE for more information.
