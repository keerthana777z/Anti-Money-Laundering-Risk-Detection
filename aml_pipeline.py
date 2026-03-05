"""
==============================================================================
  Graph-Based Financial Transaction Risk Detection
  Using Network Analytics and Graph Neural Networks
==============================================================================
  Pipeline Sections:
    1.  Data Generation
    2.  Data Preprocessing
    3.  Graph Construction (NetworkX)
    4.  Network-Analytics Feature Extraction
    5.  Feature Matrix Creation
    6.  Baseline ML Model  (Logistic Regression)
    7.  Graph Neural Network  (GraphSAGE – NumPy implementation)
    8.  Risk Score Generation
    9.  Visualisation
    10. Evaluation Metrics
==============================================================================
"""

import pandas as pd
import numpy as np
import random
import warnings
import os
from datetime import datetime, timedelta

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1 – Generating Synthetic AML Dataset")
print("=" * 70)

NUM_ACCOUNTS     = 500
NUM_TRANSACTIONS = 5_000

accounts   = [f"A{i}" for i in range(NUM_ACCOUNTS)]
start_date = datetime(2023, 1, 1)

rows = []
for _ in range(NUM_TRANSACTIONS):
    sender   = random.choice(accounts)
    receiver = random.choice(accounts)
    while receiver == sender:
        receiver = random.choice(accounts)

    amount    = round(np.random.exponential(2_000), 2)
    timestamp = start_date + timedelta(minutes=random.randint(0, 500_000))
    txn_type  = random.choice(["ACH", "CreditCard", "DebitCard", "Wire", "CashDeposit"])

    is_suspicious = 1 if (amount > 10_000 or random.random() < 0.03) else 0

    rows.append([sender, receiver, amount, timestamp, txn_type, is_suspicious])

df_raw = pd.DataFrame(rows, columns=[
    "sender_account", "receiver_account", "amount",
    "timestamp", "transaction_type", "is_suspicious"
])

csv_path = os.path.join(OUTPUT_DIR, "aml_synthetic_transactions.csv")
df_raw.to_csv(csv_path, index=False)
print(f"  ✔  Dataset saved  →  {csv_path}")
print(f"  Rows : {len(df_raw):,}   |   Suspicious : {df_raw['is_suspicious'].sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2 – Data Preprocessing")
print("=" * 70)

df = pd.read_csv(csv_path)

# 2a. Parse timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"]      = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek

# 2b. Encode transaction_type
le_txn = LabelEncoder()
df["txn_type_enc"] = le_txn.fit_transform(df["transaction_type"])

# 2c. Normalise amount (log scale is common for financial data)
df["amount_log"] = np.log1p(df["amount"])
scaler_amount = StandardScaler()
df["amount_norm"] = scaler_amount.fit_transform(df[["amount_log"]])

print(f"  ✔  Timestamp parsed, transaction_type encoded, amount normalised")
print(f"  Columns: {list(df.columns)}")
print(df.head(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3 – Building Directed Transaction Graph (NetworkX)")
print("=" * 70)

G = nx.MultiDiGraph()

for _, row in df.iterrows():
    G.add_edge(
        row["sender_account"],
        row["receiver_account"],
        amount    = row["amount"],
        amount_norm = row["amount_norm"],
        timestamp = row["timestamp"],
        txn_type  = row["transaction_type"],
        suspicious = row["is_suspicious"]
    )

print(f"  ✔  Graph constructed")
print(f"  Nodes (accounts)    : {G.number_of_nodes():,}")
print(f"  Edges (transactions): {G.number_of_edges():,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – NETWORK-ANALYTICS FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4 – Network-Analytics Feature Extraction per Account")
print("=" * 70)

# Convert to DiGraph (summing multi-edges) for centrality calculations
G_simple = nx.DiGraph()
for u, v, data in G.edges(data=True):
    if G_simple.has_edge(u, v):
        G_simple[u][v]["weight"] += data["amount"]
        G_simple[u][v]["count"]  += 1
    else:
        G_simple.add_edge(u, v, weight=data["amount"], count=1)

# Centraliy metrics (can be slow for large graphs – sample betweenness)
print("  Computing degree centrality …")
deg_centrality = nx.degree_centrality(G_simple)

print("  Computing betweenness centrality (approximate, k=100) …")
betweenness = nx.betweenness_centrality(G_simple, k=100, normalized=True, seed=42)

print("  Computing clustering coefficients …")
G_undirected  = G_simple.to_undirected()
clustering    = nx.clustering(G_undirected)

# Per-account financial aggregates
node_stats = {}
for node in G.nodes():
    in_edges  = list(G.in_edges(node,  data=True))
    out_edges = list(G.out_edges(node, data=True))

    total_in_amt  = sum(d["amount"] for _, _, d in in_edges)
    total_out_amt = sum(d["amount"] for _, _, d in out_edges)

    node_stats[node] = {
        "account"            : node,
        "in_degree"          : G.in_degree(node),
        "out_degree"         : G.out_degree(node),
        "degree_centrality"  : deg_centrality.get(node, 0),
        "betweenness_centrality": betweenness.get(node, 0),
        "clustering_coeff"   : clustering.get(node, 0),
        "total_in_amount"    : total_in_amt,
        "total_out_amount"   : total_out_amt,
        "txn_frequency"      : G.in_degree(node) + G.out_degree(node),
    }

node_df = pd.DataFrame(list(node_stats.values())).set_index("account")
print(f"  ✔  Features extracted for {len(node_df):,} accounts")
print(node_df.head(3).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – FEATURE MATRIX & GROUND-TRUTH LABELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5 – Building Feature Matrix")
print("=" * 70)

# Label: account is suspicious if ANY of its transactions are flagged
susp_senders   = df[df["is_suspicious"] == 1]["sender_account"].unique()
susp_receivers = df[df["is_suspicious"] == 1]["receiver_account"].unique()
suspicious_accounts = set(susp_senders) | set(susp_receivers)

node_df["is_suspicious"] = node_df.index.isin(suspicious_accounts).astype(int)

feature_cols = [
    "in_degree", "out_degree", "degree_centrality",
    "betweenness_centrality", "clustering_coeff",
    "total_in_amount", "total_out_amount", "txn_frequency"
]

X = node_df[feature_cols].values
y = node_df["is_suspicious"].values

scaler_feat = StandardScaler()
X_scaled    = scaler_feat.fit_transform(X)

print(f"  ✔  Feature matrix  : {X_scaled.shape}")
print(f"  Label balance  →  Suspicious: {y.sum():,}  |  Normal: {(y==0).sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – BASELINE ML MODEL (LOGISTIC REGRESSION)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6 – Baseline Logistic Regression Model")
print("=" * 70)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_scaled, y, np.arange(len(y)),
    test_size=0.25, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train, y_train)

lr_proba = lr.predict_proba(X_test)[:, 1]
lr_pred  = lr.predict(X_test)

print(f"  ✔  Logistic Regression trained on {len(X_train):,} samples")
print(f"     Accuracy : {accuracy_score(y_test, lr_pred):.4f}")
print(f"     F1-score : {f1_score(y_test, lr_pred, zero_division=0):.4f}")
print(f"     ROC-AUC  : {roc_auc_score(y_test, lr_proba):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – GRAPH NEURAL NETWORK (GraphSAGE – NumPy Implementation)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 7 – GraphSAGE Node Classification (NumPy)")
print("=" * 70)
print("  Note: PyTorch Geometric unavailable (offline env).")
print("  Implementing GraphSAGE mean-aggregation in NumPy – architecturally")
print("  equivalent for academic demonstration.")

node_list = list(node_df.index)
node_idx  = {n: i for i, n in enumerate(node_list)}
N         = len(node_list)

# Build adjacency list (incoming neighbours for SAGE aggregation)
adj = [[] for _ in range(N)]
for u, v in G_simple.edges():
    if u in node_idx and v in node_idx:
        adj[node_idx[v]].append(node_idx[u])   # v receives from u

# ── GraphSAGE layer (mean aggregation) ──────────────────────────────────────
def sage_layer(H, adj, W_self, W_neigh, bias, activation=None):
    """Single GraphSAGE layer: h_v = activation(W_self·h_v + W_neigh·mean(h_u))"""
    N, d_in = H.shape
    d_out = W_self.shape[1]
    H_out = np.zeros((N, d_out))
    for i in range(N):
        h_self = H[i]
        neigh  = adj[i]
        h_neigh = H[neigh].mean(axis=0) if neigh else np.zeros(d_in)
        H_out[i] = h_self @ W_self + h_neigh @ W_neigh + bias
    if activation == "relu":
        H_out = np.maximum(0, H_out)
    elif activation == "sigmoid":
        H_out = 1 / (1 + np.exp(-np.clip(H_out, -50, 50)))
    return H_out

def xavier(d_in, d_out):
    limit = np.sqrt(6 / (d_in + d_out))
    return np.random.uniform(-limit, limit, (d_in, d_out))

# Architecture: 8 → 32 → 16 → 1
d_in, d_h1, d_h2, d_out_gnn = len(feature_cols), 32, 16, 1

np.random.seed(0)
W1_self  = xavier(d_in, d_h1);  W1_neigh = xavier(d_in, d_h1);  b1 = np.zeros(d_h1)
W2_self  = xavier(d_h1, d_h2);  W2_neigh = xavier(d_h1, d_h2);  b2 = np.zeros(d_h2)
W3_self  = xavier(d_h2, d_out_gnn); W3_neigh = xavier(d_h2, d_out_gnn); b3 = np.zeros(d_out_gnn)

# Training with mini-batch gradient descent (full-batch for simplicity)
H0 = X_scaled.copy()

def binary_cross_entropy(prob, label):
    eps = 1e-7
    prob = np.clip(prob, eps, 1 - eps)
    return -np.mean(label * np.log(prob) + (1 - label) * np.log(1 - prob))

def forward(H):
    H1 = sage_layer(H,  adj, W1_self, W1_neigh, b1, "relu")
    H2 = sage_layer(H1, adj, W2_self, W2_neigh, b2, "relu")
    H3 = sage_layer(H2, adj, W3_self, W3_neigh, b3, "sigmoid")
    return H1, H2, H3.squeeze()

EPOCHS = 80
LR_GNN = 0.01
EPSILON = 1e-4   # finite-difference step for gradient approximation

train_mask = np.zeros(N, dtype=bool)
test_mask  = np.zeros(N, dtype=bool)
train_mask[idx_train] = True
test_mask[idx_test]   = True

print(f"  Training GraphSAGE for {EPOCHS} epochs …")
losses = []
for epoch in range(EPOCHS):
    _, _, proba = forward(H0)
    loss = binary_cross_entropy(proba[train_mask], y[train_mask])
    losses.append(loss)

    # Simple perturbation-based weight update (educational approximation)
    # Full backprop through graph layers requires autograd; here we use
    # supervised signal at output node to nudge weights
    grad_out = (proba - y) / N  # shape (N,)

    # Back-signal into W3 weights using last hidden layer
    _, H2, _ = forward(H0)
    delta3 = grad_out[:, None]                       # (N,1)
    dW3_self  = H2.T @ delta3 * LR_GNN
    dW3_neigh = H2.T @ delta3 * LR_GNN * 0.5
    db3       = delta3.mean(axis=0) * LR_GNN

    W3_self  -= dW3_self
    W3_neigh -= dW3_neigh
    b3       -= db3

    if (epoch + 1) % 20 == 0:
        pred_bin = (proba[train_mask] > 0.5).astype(int)
        tr_acc = accuracy_score(y[train_mask], pred_bin)
        print(f"    Epoch {epoch+1:3d}/{EPOCHS}  |  Loss: {loss:.4f}  |  Train Acc: {tr_acc:.4f}")

_, _, gnn_proba_all = forward(H0)
gnn_proba_test = gnn_proba_all[test_mask]
gnn_pred_test  = (gnn_proba_test > 0.5).astype(int)

print(f"\n  ✔  GraphSAGE evaluation:")
print(f"     Accuracy : {accuracy_score(y_test, gnn_pred_test):.4f}")
print(f"     F1-score : {f1_score(y_test, gnn_pred_test, zero_division=0):.4f}")
print(f"     ROC-AUC  : {roc_auc_score(y_test, gnn_proba_test):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – RISK SCORE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 8 – Risk Score Generation")
print("=" * 70)

# Ensemble: average of LR and GNN scores (full dataset)
lr_proba_all = lr.predict_proba(X_scaled)[:, 1]

risk_df = node_df[["is_suspicious"]].copy()
risk_df["lr_risk_score"]   = lr_proba_all
risk_df["gnn_risk_score"]  = gnn_proba_all
risk_df["ensemble_score"]  = (lr_proba_all + gnn_proba_all) / 2
risk_df["risk_label"]      = (risk_df["ensemble_score"] > 0.5).astype(int)

risk_csv = os.path.join(OUTPUT_DIR, "account_risk_scores.csv")
risk_df.reset_index().to_csv(risk_csv, index=False)
print(f"  ✔  Risk scores saved  →  {risk_csv}")
print(risk_df.sort_values("ensemble_score", ascending=False).head(10).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 9 – Generating Visualisations")
print("=" * 70)

fig = plt.figure(figsize=(22, 26), facecolor="#0d1117")
fig.suptitle(
    "Graph-Based Financial Transaction Risk Detection\n"
    "AML Detection Pipeline — Academic Prototype",
    fontsize=18, fontweight="bold", color="white", y=0.98
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

ACCENT   = "#00d4aa"
DANGER   = "#ff4757"
SAFE     = "#2ed573"
WARN     = "#ffa502"
BG_PANEL = "#161b22"
TEXT_COL = "#c9d1d9"

def style_ax(ax, title):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_title(title, color=ACCENT, fontsize=11, fontweight="bold", pad=10)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)

# ── Plot 1: Transaction sub-graph ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "Transaction Network — Top-50 High-Value Accounts (Sample Sub-graph)")

top50 = risk_df.nlargest(50, "ensemble_score").index.tolist()
SG = G_simple.subgraph(top50).copy()

pos = nx.spring_layout(SG, seed=42, k=0.8)
node_colors = [DANGER if risk_df.loc[n, "risk_label"] == 1 else SAFE
               for n in SG.nodes()]
edge_weights = [SG[u][v]["weight"] / 10_000 for u, v in SG.edges()]

nx.draw_networkx_nodes(SG, pos, ax=ax1, node_color=node_colors,
                       node_size=200, alpha=0.9)
nx.draw_networkx_edges(SG, pos, ax=ax1, edge_color="#58a6ff",
                       alpha=0.4, arrows=True, arrowsize=12,
                       width=[min(w, 3) for w in edge_weights])
nx.draw_networkx_labels(SG, pos, ax=ax1, font_size=6, font_color="white")

from matplotlib.patches import Patch
legend = [Patch(color=DANGER, label="Suspicious"), Patch(color=SAFE, label="Normal")]
ax1.legend(handles=legend, loc="upper right", facecolor=BG_PANEL,
           edgecolor="#30363d", labelcolor=TEXT_COL)
ax1.axis("off")

# ── Plot 2: Risk Score Distribution ───────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
style_ax(ax2, "Risk Score Distribution (Ensemble)")

norm_scores = risk_df[risk_df["is_suspicious"] == 0]["ensemble_score"]
susp_scores = risk_df[risk_df["is_suspicious"] == 1]["ensemble_score"]

bins = np.linspace(0, 1, 30)
ax2.hist(norm_scores,  bins=bins, alpha=0.7, color=SAFE,   label="Normal",     edgecolor="#0d1117")
ax2.hist(susp_scores, bins=bins, alpha=0.7, color=DANGER, label="Suspicious", edgecolor="#0d1117")
ax2.axvline(0.5, color=WARN, linestyle="--", linewidth=1.5, label="Threshold 0.5")
ax2.set_xlabel("Risk Score"); ax2.set_ylabel("Count")
ax2.legend(facecolor=BG_PANEL, edgecolor="#30363d", labelcolor=TEXT_COL, fontsize=9)

# ── Plot 3: ROC Curves ────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
style_ax(ax3, "ROC Curves — LR vs GraphSAGE")

fpr_lr,  tpr_lr,  _ = roc_curve(y_test, lr_proba)
fpr_gnn, tpr_gnn, _ = roc_curve(y_test, gnn_proba_test)

auc_lr  = roc_auc_score(y_test, lr_proba)
auc_gnn = roc_auc_score(y_test, gnn_proba_test)

ax3.plot(fpr_lr,  tpr_lr,  color=ACCENT, lw=2, label=f"Logistic Reg  AUC={auc_lr:.3f}")
ax3.plot(fpr_gnn, tpr_gnn, color=WARN,   lw=2, label=f"GraphSAGE     AUC={auc_gnn:.3f}")
ax3.plot([0,1],[0,1], color="#30363d", linestyle="--", lw=1)
ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
ax3.legend(facecolor=BG_PANEL, edgecolor="#30363d", labelcolor=TEXT_COL, fontsize=9)

# ── Plot 4: Suspicious Cluster Heatmap ───────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
style_ax(ax4, "Suspicious Account Cluster — Top 30 Risk Heatmap")

top30_idx = risk_df.nlargest(30, "ensemble_score").index
top30 = node_df.loc[top30_idx, feature_cols[:6]]
feat_scaled_top30 = scaler_feat.transform(node_df.loc[top30_idx, feature_cols])
im = ax4.imshow(feat_scaled_top30.T, aspect="auto", cmap="RdYlGn_r",
                vmin=-2, vmax=2)
ax4.set_yticks(range(6))
ax4.set_yticklabels(feature_cols[:6], fontsize=8, color=TEXT_COL)
ax4.set_xlabel("Account (ranked by risk)", color=TEXT_COL)
plt.colorbar(im, ax=ax4, label="Normalised Value").ax.yaxis.label.set_color(TEXT_COL)
ax4.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

# ── Plot 5: Training Loss Curve ───────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
style_ax(ax5, "GraphSAGE Training Loss Curve")

ax5.plot(range(1, EPOCHS + 1), losses, color=ACCENT, lw=2)
ax5.fill_between(range(1, EPOCHS + 1), losses, alpha=0.15, color=ACCENT)
ax5.set_xlabel("Epoch"); ax5.set_ylabel("Binary Cross-Entropy Loss")

plt.savefig(os.path.join(OUTPUT_DIR, "aml_pipeline_dashboard.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✔  Dashboard saved  →  {OUTPUT_DIR}/aml_pipeline_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 10 – Full Evaluation Metrics")
print("=" * 70)

def metrics_report(name, y_true, y_pred, y_prob):
    print(f"\n  ── {name} ──")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_true, y_prob):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}")

metrics_report("Logistic Regression", y_test, lr_pred, lr_proba)
metrics_report("GraphSAGE (NumPy)",   y_test, gnn_pred_test, gnn_proba_test)

ensemble_pred_test = ((lr_proba + gnn_proba_test) / 2 > 0.5).astype(int)
ensemble_prob_test = (lr_proba + gnn_proba_test) / 2
metrics_report("Ensemble (LR + GNN)", y_test, ensemble_pred_test, ensemble_prob_test)

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE")
print(f"  Outputs saved in: {OUTPUT_DIR}")
print("=" * 70)
