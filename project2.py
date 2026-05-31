import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def execute_decodelabs_pipeline():
    print("Executing DecodeLabs Project 2: KNN Classifier\n")

    # ==========================================
    # 1. INPUT PHASE (Load & Initial Check)
    # ==========================================
    iris_data = load_iris()
    features = iris_data.data
    labels = iris_data.target
    target_names = iris_data.target_names

    print("--- 1. DATASET DIMENSIONS ---")
    print(f"Total Rows: {features.shape[0]}")
    print(f"Total Columns (Features): {features.shape[1]}")
    print(f"Classes: {list(target_names)}\n")

    # ==========================================
    # 2. PROCESS PHASE (Split, Scale & Train)
    # ==========================================

    # Step A: Shuffle & Train-Test Split (80/20)
    feat_train, feat_test, label_train, label_test = train_test_split(
        features, labels,
        test_size=0.20,
        random_state=42,
        shuffle=True,
        stratify=labels
    )

    print("--- 3. SPLIT SETS EXACT SIZE ---")
    print(f"Training Data: {len(feat_train)} records")
    print(f"Testing Data: {len(feat_test)} records\n")

    # Step B: The Gatekeeper Rule (StandardScaler)
    # Applied AFTER split to prevent data leakage
    data_scaler = StandardScaler()
    feat_train_scaled = data_scaler.fit_transform(feat_train)
    feat_test_scaled = data_scaler.transform(feat_test)

    print("--- 2. SCALING CONFIRMATION ---")
    print(f"Mean of Feature 0 (Train): {np.mean(feat_train_scaled[:, 0]):.2f} (Expected: 0.0)")
    print(f"Variance of Feature 0 (Train): {np.var(feat_train_scaled[:, 0]):.2f} (Expected: 1.0)\n")

    # Step C: The Algorithm (KNN with strictly K=5)
    # Scikit-Learn Workflow: Instantiate -> Fit -> Predict
    knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_classifier.fit(feat_train_scaled, label_train)

    print("--- 4. TRAINED KNN MODEL STATUS ---")
    print(f"Model Engine: {knn_classifier}")
    print("Status: Training Complete.\n")

    # ==========================================
    # 3. OUTPUT PHASE (Predict & Evaluate)
    # ==========================================
    predictions = knn_classifier.predict(feat_test_scaled)

    # Confusion Matrix
    conf_matrix = confusion_matrix(label_test, predictions)
    print("--- 5. FINAL CONFUSION MATRIX GRID ---")
    print("Rows: True Values | Columns: Predictions")
    print(conf_matrix)
    print("\n* Diagonals: Correct Predictions (TP/TN)")
    print("* Non-Diagonals: Errors (FP/FN)\n")

    # F1-Score & Metrics
    print("--- 6. TARGET CLASSES F1-SCORE ---")
    full_report = classification_report(label_test, predictions, target_names=target_names)
    print(full_report)

    # ==========================================
    # 4. HIGH-GRADE CORPORATE VISUALIZATION 🚀
    # ==========================================

    # Evaluate Elbow curve data for K=1 to 30
    k_list = range(1, 31)
    error_list = []
    for k_val in k_list:
        temp_model = KNeighborsClassifier(n_neighbors=k_val)
        temp_model.fit(feat_train_scaled, label_train)
        temp_preds = temp_model.predict(feat_test_scaled)
        error_list.append(1 - accuracy_score(label_test, temp_preds))

    # --- Applying Premium Aesthetics Themes ---
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'

    # High DPI Canvas for ultra-sharp corporate rendering
    plt.figure(figsize=(14, 6), dpi=120)

    # Plot 1: Premium Styled Elbow Curve (Matches Slide: Tuning the Engine)
    plt.subplot(1, 2, 1)
    plt.plot(list(k_list), error_list, marker='o', markersize=6, linewidth=2,
             color='#1e3d59', markerfacecolor='#ff6e40', markeredgecolor='white', markeredgewidth=1)

    # Enhanced Optimal Target Line
    plt.axvline(x=5, color='#ff6e40', linestyle='--', linewidth=2, label='The Elbow (Required K=5)')

    plt.title("Tuning the Engine: Choosing 'K'", fontsize=13, fontweight='bold', pad=15, color='#112233')
    plt.xlabel("Number of Neighbors (K Value)", fontsize=11, labelpad=8)
    plt.ylabel("Error Rate", fontsize=11, labelpad=8)
    plt.xlim(0, 31)
    plt.xticks(range(0, 31, 5))
    plt.grid(True, linestyle=':', alpha=0.6, color='#cfd8dc')
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)

    # Plot 2: Sophisticated Heatmap Layout (Matches Slide: The Diagnostic Tool)
    plt.subplot(1, 2, 2)
    # Soft professional sequential color mapping
    corporate_cmap = sns.light_palette("#1e3d59", as_cmap=True)

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=corporate_cmap, cbar=True,
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 12, "weight": "bold", "color": "#112233"},
                linewidths=2, linecolor='white', square=True)

    plt.title("The Diagnostic Tool: Confusion Matrix", fontsize=13, fontweight='bold', pad=15, color='#112233')
    plt.xlabel("Predicted Classes", fontsize=11, labelpad=10)
    plt.ylabel("Actual Classes", fontsize=11, labelpad=10)

    plt.tight_layout()
    plt.savefig("standard_knn_results.png")
    print("Plots saved as 'standard_knn_results.png' in premium high-grade theme.\n")
    plt.show()


# Run the script
if __name__ == "__main__":
    execute_decodelabs_pipeline()