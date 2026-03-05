

# 🔍 Graph-Based AML Risk Detection System

A **Graph Machine Learning–based Anti-Money Laundering (AML) system** that detects suspicious financial accounts by combining **network analytics, traditional machine learning, and graph neural networks**.

This project builds a **transaction network of bank accounts**, extracts graph-based behavioral features, and predicts **high-risk accounts** using an **ensemble of Logistic Regression and Graph Neural Networks (GraphSAGE)**.

The results are visualized through an **interactive Streamlit dashboard** designed for financial analysts.

---

# 📌 Project Overview

Financial institutions must monitor millions of transactions to detect **money laundering activities**.

Traditional systems rely on rule-based detection. However, money laundering often occurs through **complex transaction networks**, making it difficult to detect using standard methods.

This project introduces a **graph-based AML detection pipeline** that:

* Builds a **directed transaction network**
* Extracts **network behavior features**
* Applies **machine learning models**
* Computes **risk scores for accounts**
* Provides an **interactive risk monitoring dashboard**

---

# 🧠 Key Features

✔ Graph-based transaction modeling
✔ Network feature extraction (centrality metrics)
✔ Logistic Regression baseline model
✔ Graph Neural Network (GraphSAGE) model
✔ Ensemble risk scoring
✔ Interactive AML monitoring dashboard
✔ Network graph visualization of suspicious accounts

---

# 🏗 System Architecture

```
Transaction Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Transaction Graph Construction
(NetworkX)
        │
        ▼
Graph Feature Extraction
        │
        ▼
Machine Learning Models
(Logistic Regression + GraphSAGE)
        │
        ▼
Risk Score Ensemble
        │
        ▼
AML Risk Dashboard
(Streamlit)
```

---
<img width="1600" height="863" alt="image" src="https://github.com/user-attachments/assets/eb4f03bc-7a2a-44df-8706-b055077b1143" />


# 📂 Project Structure

```
AML-Risk-Detection/
│
├── aml_pipeline.py
│   Main pipeline for dataset generation, preprocessing,
│   graph construction, model training, and risk scoring
│
├── dashboard.py
│   Streamlit dashboard for AML risk monitoring
│
├── aml_synthetic_transactions.csv
│   Generated synthetic transaction dataset
│
├── account_risk_scores.csv
│   Output risk scores for each account
│
├── outputs/
│   Generated results and evaluation outputs
│
└── README.md
```

---

# 📊 Dataset

The dataset contains **financial transaction records between bank accounts**.

### Example Features

| Feature          | Description                          |
| ---------------- | ------------------------------------ |
| sender_account   | Source bank account                  |
| receiver_account | Destination bank account             |
| amount           | Transaction amount                   |
| timestamp        | Transaction time                     |
| transaction_type | Type of transaction                  |
| is_suspicious    | Label indicating suspicious activity |

Synthetic transactions are generated to simulate **money laundering behaviors**.

---

# ⚙️ Methodology

## 1️⃣ Data Preprocessing

The raw transaction dataset undergoes several transformations:

* Timestamp parsing
* Transaction type encoding
* Log transformation of transaction amount
* Feature normalization

New features generated:

```
hour
dayofweek
txn_type_encoded
amount_log
amount_norm
```

---

# 2️⃣ Transaction Graph Construction

Transactions are represented as a **directed graph**:

```
Nodes → Bank Accounts
Edges → Transactions
```

Graph created using:

```
NetworkX
```

This captures **relationships between accounts**.

---

# 3️⃣ Graph Feature Extraction

Network analytics are used to identify suspicious patterns.

Extracted features include:

* In-degree
* Out-degree
* Degree centrality
* Betweenness centrality
* Clustering coefficient
* Transaction frequency
* Total incoming amount
* Total outgoing amount

These features help detect **abnormal transaction behavior**.

---

# 4️⃣ Machine Learning Models

## Logistic Regression (Baseline)

Uses tabular graph features to classify accounts.

Outputs:

```
LR Risk Score
```

---

## Graph Neural Network (GraphSAGE)

Uses the **transaction graph structure** to detect suspicious nodes.

Advantages:

* Captures **network relationships**
* Detects hidden laundering patterns

Outputs:

```
GNN Risk Score
```

---

# 5️⃣ Risk Score Ensemble

The final risk score is calculated by combining both models:

```
Final Risk Score =
(LR Score + GNN Score) / 2
```

Accounts with high scores are flagged as **high-risk accounts**.

---

# 📈 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.616    | 0.614     | 0.672  | 0.642    | 0.659   |
| GraphSAGE           | 0.448    | 0.468     | 0.578  | 0.518    | 0.417   |
| Ensemble            | 0.552    | 0.548     | 0.719  | 0.622    | 0.575   |

The **ensemble model improves recall**, which is critical in fraud detection.

---

# 🖥 Interactive Dashboard

The project includes a **Streamlit-based AML monitoring dashboard**.

### Features

* Risk score table for accounts
* High-risk account ranking
* Risk score distribution analysis
* Suspicious vs normal account visualization
* Transaction network graph
* Model performance evaluation

Run the dashboard:

```
streamlit run dashboard.py
```

---

# 🌐 Network Visualization

The system visualizes the **transaction network**:

* Nodes represent accounts
* Edges represent transactions
* Suspicious accounts highlighted in red
* Node size proportional to risk score

This helps analysts identify **fraud rings and suspicious transaction patterns**.

---

# 🚀 How to Run the Project

## 1️⃣ Install dependencies

```
pip install pandas numpy networkx scikit-learn plotly streamlit pyvis
```

---

## 2️⃣ Run the AML pipeline

```
python aml_pipeline.py
```

This generates:

```
aml_synthetic_transactions.csv
account_risk_scores.csv
```

---

## 3️⃣ Launch the dashboard

```
streamlit run dashboard.py
```

Open in browser:

```
http://localhost:8501
```

---

# 💡 Use Cases

This system can assist:

* Financial institutions
* Fraud detection teams
* Compliance analysts
* Financial intelligence units

Applications include:

* Money laundering detection
* Fraud monitoring
* Transaction network analysis

---

# 🔬 Technologies Used

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Core programming      |
| NetworkX     | Graph modeling        |
| Scikit-learn | Machine learning      |
| GraphSAGE    | Graph neural networks |
| Streamlit    | Dashboard UI          |
| Plotly       | Data visualization    |
| PyVis        | Network visualization |

---

# 📚 Future Improvements

* Real-world banking dataset integration
* Advanced Graph Neural Networks (GAT, GCN)
* Real-time transaction monitoring
* Temporal graph modeling
* Explainable AI for fraud detection

---

# 👩‍💻 Author

**Keerthana A.R**

AI & Data Science Student
Vellore Institute of Technology, Chennai

---

# 📜 License

This project is developed for **academic and research purposes**.

---


