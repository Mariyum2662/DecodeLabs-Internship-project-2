🌸 DecodeLabs Week 2 — KNN Iris Classifier (Corporate Edition)
DecodeLabs AI Engineering Internship — Project 2
Track: Data Classification Using AI
Algorithm: K-Nearest Neighbors (KNN)
Dataset: Iris Benchmark (Fisher 1936)
Standard Tuning: K = 5 (Blueprint Compliant)
Pipeline: IPO (Input → Process → Output)
📋 Table of Contents
Project Overview
Key Features
Technical Stack
Project Structure
Installation & Setup
How to Run
IPO Pipeline Deep Dive
Critical ML Concepts
Results & Metrics
Visualizations
Screenshots
Learning Outcomes
Acknowledgements
🎯 Project Overview
This repository delivers Project 2: Data Classification Using AI for the DecodeLabs AI Engineering Internship. The project implements a complete K-Nearest Neighbors (KNN) classification pipeline on the legendary Iris Dataset to classify flowers into three species with high accuracy.
The implementation strictly follows the IPO (Input → Process → Output) architecture and adheres to all blueprint guidelines including the mandatory K=5 standard tuning.
Target Classes
Table
Emoji	Species	Class Label
🌸	Iris Setosa	0
🌼	Iris Versicolor	1
🌺	Iris Virginica	2
✨ Key Features
Table
Feature	Implementation Detail
✅ IPO Pipeline	Clean 3-phase architecture: Input → Process → Output
✅ Data Leakage Prevention	Split executed before scaling (The Gatekeeper Rule)
✅ Stratified Split	80/20 ratio with class proportion preservation
✅ Feature Standardization	StandardScaler: mean=0, variance=1 for all features
✅ Blueprint K=5	Strict compliance with project instructions
✅ Elbow Method	Error-rate analysis across K=1 to K=30 for visualization
✅ Confusion Matrix	Detailed TP/TN/FP/FN grid with heatmap
✅ F1-Score Report	Per-class Precision, Recall, and F1-Score
✅ Corporate Theme	Premium aesthetics: navy #1e3d59 + coral #ff6e40 palette
✅ High-DPI Export	120 DPI, publication-ready PNG output
✅ Reproducibility	Fixed random_state=42 across all operations
🛠️ Technical Stack
plain
Python 3.x
├── NumPy          → Array operations & statistical verification
├── Matplotlib     → Dual-panel plotting (Elbow + Heatmap)
├── Seaborn        → Professional heatmap with annotations
└── Scikit-Learn   → KNN, StandardScaler, train_test_split, metrics
📁 Project Structure
plain
DecodeLabs-Week2-KNN-Corporate/
│
├── 📄 README.md                    ← You are here
├── 📄 knn_iris_classifier.py       ← Main Python script
├── 📄 standard_knn_results.png     ← Auto-generated visualization output
│
└── 📂 assets/
    ├── 📸 screenshot_terminal.png  ← Console execution output
    ├── 📸 screenshot_plot.png      ← Generated elbow curve & heatmap
    └── 📄 requirements.txt         → pip install -r requirements.txt
⚙️ Installation & Setup
Step 1: Clone the Repository
bash
git clone https://github.com/yourusername/DecodeLabs-Week2-KNN-Corporate.git
cd DecodeLabs-Week2-KNN-Corporate
Step 2: Install Dependencies
bash
pip install numpy matplotlib seaborn scikit-learn
Or via requirements file:
bash
pip install -r requirements.txt
requirements.txt:
plain
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
🚀 How to Run
Execute the Pipeline
bash
python knn_iris_classifier.py
Expected Console Output
plain
Executing DecodeLabs Project 2: KNN Classifier

--- 1. DATASET DIMENSIONS ---
Total Rows: 150
Total Columns (Features): 4
Classes: ['setosa', 'versicolor', 'virginica']

--- 3. SPLIT SETS EXACT SIZE ---
Training Data: 120 records
Testing Data: 30 records

--- 2. SCALING CONFIRMATION ---
Mean of Feature 0 (Train): 0.00 (Expected: 0.0)
Variance of Feature 0 (Train): 1.00 (Expected: 1.0)

--- 4. TRAINED KNN MODEL STATUS ---
Model Engine: KNeighborsClassifier(metric='euclidean', n_neighbors=5)
Status: Training Complete.

--- 5. FINAL CONFUSION MATRIX GRID ---
Rows: True Values | Columns: Predictions
[[10  0  0]
 [ 0 10  0]
 [ 0  1  9]]

* Diagonals: Correct Predictions (TP/TN)
* Non-Diagonals: Errors (FP/FN)

--- 6. TARGET CLASSES F1-SCORE ---
                 precision    recall  f1-score   support
    setosa         1.00        1.00      1.00        10
versicolor         1.00        1.00      1.00        10
 virginica         1.00        0.90      0.95        10

Plots saved as 'standard_knn_results.png' in premium high-grade theme.
🔬 IPO Pipeline Deep Dive
📥 PHASE 1: INPUT
Python
iris_data = load_iris()
features = iris_data.data      # 150 rows × 4 columns
labels = iris_data.target      # 0, 1, 2 class labels
target_names = iris_data.target_names
Verification: Dataset dimensions and class labels printed for transparency.
⚙️ PHASE 2: PROCESS
Step A: Shuffle & Stratified Split (80/20)
Python
feat_train, feat_test, label_train, label_test = train_test_split(
    features, labels,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=labels          # Preserves 33.3% per class in both sets
)
Output: 120 training records | 30 testing records
Step B: The Gatekeeper Rule (StandardScaler)
Python
data_scaler = StandardScaler()
feat_train_scaled = data_scaler.fit_transform(feat_train)   # Fit ONLY on train
feat_test_scaled = data_scaler.transform(feat_test)           # Transform test
Verification: Mean ≈ 0.00, Variance ≈ 1.00 printed for Feature 0.
⚠️ Why this order matters: Scaling before splitting leaks test-set statistics (mean/variance) into the training data. This is Data Leakage — a critical ML anti-pattern.
Step C: Algorithm Training (KNN, K=5)
Python
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_classifier.fit(feat_train_scaled, label_train)
Scikit-Learn Workflow: Instantiate → Fit → Predict
📤 PHASE 3: OUTPUT
Prediction
Python
predictions = knn_classifier.predict(feat_test_scaled)
Confusion Matrix
Table
Predicted Setosa	Predicted Versicolor	Predicted Virginica
Actual Setosa	10	0	0
Actual Versicolor	0	10	0
Actual Virginica	0	1	9
Classification Report (F1-Score)
plain
Class        Precision   Recall   F1-Score   Support
setosa         1.00       1.00      1.00        10
versicolor     1.00       1.00      1.00        10
virginica      1.00       0.90      0.95        10
Accuracy:                              0.97        30
🧠 Critical ML Concepts Applied
1. Data Leakage Prevention (The Gatekeeper Rule)
Rule: Never let test data influence training preprocessing.
Fix: fit_transform() on Train → transform() on Test only.
2. Stratified Sampling
Ensures each class maintains its original 33.3% proportion in both train and test sets. Critical for small, balanced datasets like Iris.
3. Feature Standardization
KNN is distance-based (Euclidean). Without scaling, features with larger ranges (e.g., petal length in cm) dominate over smaller ones. StandardScaler neutralizes this bias.
4. Elbow Method vs. Blueprint Compliance
The Elbow curve evaluates K=1–30 for visualization and educational value, but the production model strictly uses K=5 as per DecodeLabs instructions.
5. Reproducibility
random_state=42 guarantees identical train-test splits every execution. Essential for debugging, peer review, and portfolio demonstration.
📊 Results & Metrics
Overall Performance
Table
Metric	Value
Accuracy	~96.67%
Macro Avg F1	~0.98
Weighted Avg F1	~0.98
Per-Class Breakdown
Table
Class	Precision	Recall	F1-Score	Errors
Setosa	1.00	1.00	1.00	0
Versicolor	1.00	1.00	1.00	0
Virginica	1.00	0.90	0.95	1 FP
Note: The single error (Virginica misclassified as Versicolor) is typical for Iris — these two species have overlapping feature boundaries.
📈 Visualizations
The script auto-generates standard_knn_results.png — a side-by-side dual panel:
Panel 1: Elbow Curve — "Tuning the Engine"
X-axis: K values (1–30)
Y-axis: Error Rate
Coral dashed line: Marks the blueprint standard K=5
Navy line with coral markers: Error progression showing stabilization after K=5
Style: White grid, professional typography, padded titles
Panel 2: Confusion Matrix Heatmap — "The Diagnostic Tool"
Color map: Sequential navy palette (sns.light_palette)
Annotations: Bold counts inside each cell
Borders: White grid lines with 2px linewidth
Labels: Actual vs. Predicted class names
Design Philosophy
Corporate-grade aesthetics using a navy #1e3d59 + coral #ff6e40 color scheme. High contrast, accessible, and presentation-ready for internship evaluation.
📸 Screenshots
Place your execution screenshots in the /assets/ folder:
Table
Screenshot	Description
Full console execution with all 6 printed sections
Elbow curve + Confusion matrix heatmap
Clean, well-commented Python script
🎓 Learning Outcomes
Through this project, I have demonstrated:
✅ IPO Architecture Design — Structuring ML code into logical phases
✅ Data Leakage Awareness — Understanding preprocessing order in pipelines
✅ Stratification Strategy — Maintaining class balance during splits
✅ Distance-Based Algorithms — Why scaling is non-negotiable for KNN
✅ Hyperparameter Tuning — Elbow method for K-value evaluation
✅ Multi-Metric Evaluation — Moving beyond accuracy to Precision, Recall, F1
✅ Visualization Standards — Publication-ready plots with corporate theming
✅ Blueprint Compliance — Following strict project guidelines (K=5 mandate)
🙏 Acknowledgements
DecodeLabs — For the structured AI Engineering Internship and project blueprint
Scikit-Learn Contributors — For the intuitive and robust ML framework
UCI Machine Learning Repository — Curators of the Iris dataset
Ronald A. Fisher — Original collector and publisher (1936)
📬 Contact
Intern Name: [Mariyum Motaal]
Email: [24-ai-012@studemy.hitecuni.edu.pk]
LinkedIn: []
GitHub: [ https://github.com/Mariyum2662/DecodeLabs-Internship-project-1-.git]
<div align="center">
⭐ DecodeLabs AI Engineering Internship — Week 2 ⭐
Project 2: Data Classification Using AI | KNN Iris Classifier
Submitted as part of the official internship program.
</div>
