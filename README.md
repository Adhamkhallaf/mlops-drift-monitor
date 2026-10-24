# 🔍 Real-Time ML Data Drift & Observability Monitor

A production-grade MLOps monitoring dashboard built with **Python**, **Streamlit**, and **Scikit-Learn** designed to detect real-world data drift, concept decay, and model degradation over time.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

---

## 🚀 Project Overview
In modern machine learning pipelines, training a high-performing model is only half the battle. Once deployed into production, real-world economic shifts, seasonal changes, or behavioral variations cause **Data Drift** (changes in incoming feature distributions), which silently degrades model accuracy.

This project simulates a live production monitoring system using the **California Housing Dataset**. It tracks production traffic against a baseline training distribution, automatically audits discrepancies, generates HTML diagnostic reports, and provides an interactive inference sandbox.

---

## ✨ Key Features
* **Automated Data Drift Audit:** Compares live production data against training baselines and flags statistical anomalies at the feature level (e.g., Median Income shifts).
* **Interactive HTML Reporting:** Embeds dynamic, self-contained auditing reports directly into the observability dashboard.
* **Live Model Prediction Sandbox:** Allows engineers to test model behavior via interactive sliders under changing parameters, managing strict input sequencing and schema alignment.
* **Dual Simulation Modes:** Toggle instantly between *Normal Production Data* and *Drifted Production Data* (simulating market inflation and shifts).

---

## 📂 Project Architecture
```text
mlops-drift-monitor/
├── app.py              # Main Streamlit monitoring dashboard & inference app
├── requirements.txt    # Project dependencies
├── Dockerfile          # Containerization configuration for cloud deployment
├── model.pkl           # Pre-trained baseline Random Forest Regressor
├── reference_data.csv  # Training baseline reference dataset
├── drifted_data.csv    # Simulated drifted production data batch
└── drift_report.html   # Automated feature-level drift audit report