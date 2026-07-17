# Market Regime Classifier

A Python-based machine learning pipeline that uses unsupervised learning (K-Means Clustering) to identify, classify, and visualize distinct market regimes in financial data (e.g., S&P 500). 

This project processes raw historical financial data, applies clustering algorithms from scratch to segment market phases, and visualizes the results by overlaying colored regime bands onto asset price charts.

---

## 📁 Project Structure

```text
regime_classifier/
├── data/                  # Raw financial datasets (e.g., S&P 500 CSVs)
├── outputs/               # Saved plots and visualization results
├── notebooks/             # Jupyter Notebooks for experimentation
├── data_loader.py         # Script to fetch/load and clean financial data
├── visualizer.py          # Functions for plotting and saving regime charts
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation