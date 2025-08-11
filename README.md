# 🛡️ FraudSniper  
**AI-Powered Fraud Detection & Analysis Engine** with real-time dashboarding, anomaly scoring, and optional Splunk integration.  

---

## 🚀 Overview  
FraudSniper is a **cyber + AI** fraud detection system designed to identify and score suspicious transactions in real time.  
Built with **Isolation Forest** for anomaly detection, **Streamlit** for interactive visualization, and a clean, modular architecture for easy deployment.  

**Core Features:**  
- ⚡ **Real-Time Fraud Detection** — Isolation Forest ML model optimized for anomaly detection  
- 📊 **Interactive Dashboard** — Filter, search, and visualize anomalies in an intuitive Streamlit UI  
- 🎯 **Severity Scoring** — Risk-based scoring logic to prioritize high-impact anomalies  
- 📈 **KPI Metrics** — Track fraud trends with dynamic Plotly visualizations  
- 🛠 **Modular Design** — Ready for AWS Lambda or other deployment environments  
- 🔌 **Optional Splunk Integration** — Send flagged anomalies to Splunk for centralized monitoring (currently commented out for security)  

---

## 🧠 Tech Stack  
- **Python** — Core ML + data processing  
- **Pandas / NumPy** — Data manipulation  
- **Scikit-learn** — Isolation Forest model  
- **Streamlit** — Interactive UI  
- **Plotly** — Visualizations  
- **AWS Lambda** — Optional serverless deployment  
- **Splunk HEC** — Optional log ingestion  

---

## 📂 Project Structure  
```plaintext
FraudSniper/
│── data/                   # Sample datasets / training data
│── models/                 # Saved Isolation Forest models
│── src/
│   ├── preprocessing.py    # Data cleaning & feature engineering
│   ├── detection.py        # ML model logic
│   ├── scoring.py          # Severity scoring functions
│   ├── dashboard.py        # Streamlit dashboard
│   └── utils.py            # Helper functions
│── requirements.txt        # Dependencies
│── README.md               # This file
│── LICENSE



## ⚙️ Installation  
```bash
# Clone the repository
git clone https://github.com/yourusername/FraudSniper.git
cd FraudSniper

# Install dependencies
pip install -r requirements.txt

## ▶️ Usage  
Run the dashboard locally:  
```bash
streamlit run src/dashboard.py

## 📸 Screenshots / Demo  
*(Add your own GIFs or images here to make the repo pop)*  

Example placeholders:  
![Dashboard Overview](docs/images/dashboard_example.pn)  
![Anomaly Detection View](docs/images/anomaly_detection.png)  
![Severity Scoring Chart](docs/images/severity_scoring.png)  

 ## 🔒 Security Note  
- Splunk integration is commented out by default to prevent credential leaks.  
- If enabling, store tokens in **environment variables** or a **secrets manager** — never in plain code.  

## 📸 Screenshots / Demo  
*(Add your own GIFs or images here to make the repo pop)*  

Example placeholders:  
![Dashboard Overview]()  
![Anomaly Detection View]()  
![Severity Scoring Chart]()  
