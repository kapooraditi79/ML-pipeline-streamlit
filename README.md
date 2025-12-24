# ML-pipeline-streamlit
🚀 No-Code ML Pipeline Builder
A robust, web-based AutoML tool that allows users to upload raw data, process it, train models, and visualize results—without writing a single line of code.

Designed with a focus on Machine Learning Engineering principles, this project bridges the gap between complex algorithmic logic and intuitive user experience.

📖 Overview
The objective was to build an end-to-end ML pipeline that is accessible to non-technical users but powerful enough to handle real-world, "dirty" datasets.

Unlike standard tutorial scripts, this application implements a dynamic preprocessing engine that automatically detects data types, handles missing values, and encodes categorical features, ensuring the pipeline never crashes on arbitrary user data.

🌟 Key Features
Universal Dataset Support: Upload any CSV/Excel. The system automatically identifies Target vs. Features and Text vs. Numbers.

Robust Preprocessing:

Automated Imputation (Mean for numerical, Mode for categorical).

Dynamic One-Hot Encoding and Standardization/Normalization.

Algorithm "Zoo":

Standard: Logistic Regression & Decision Trees (Scikit-Learn).

Custom Implementation: A MulticlassLogistic regression built from scratch using Numpy to demonstrate the underlying mathematics (Gradient Descent, Cross-Entropy Loss).

Interactive Visualizations:

Plotly Heatmaps for Confusion Matrices.

Tree Structure Visualization for Decision Tree interpretability.

Feature Importance analysis.

🛠 Technical Architecture
This project is not just a UI wrapper; it is engineered for stability and mathematical correctness.

1. The Preprocessing Pipeline (ColumnTransformer)
To handle raw user data safely, I implemented a split-path pipeline using Scikit-Learn’s ColumnTransformer. This ensures that:

Numerical columns are isolated, imputed, and scaled (Standard or MinMax).

Categorical columns are isolated, imputed, and One-Hot Encoded.

The paths merge back together into a generic NumPy array ready for training.

Python

# Architecture Snippet
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop'
)
2. Custom ML Implementation (The "From Scratch" Engine)
While Scikit-Learn is used for production reliability, I implemented a Custom Logistic Regression class to demonstrate an understanding of the core algorithms.

Optimization: Batch Gradient Descent.

Loss Function: Log-Loss (Binary Cross-Entropy) with L1/L2 Regularization support.

Multiclass Strategy: One-vs-Rest (OvR) wrapper around the binary classifier to handle multi-class datasets (e.g., Iris).

Initialization: Xavier/Glorot Initialization for weight stability.

📸 Visuals & Insights
The application prioritizes interpretability:

Decision Tree Plotting: Uses matplotlib to render the actual tree structure, allowing users to trace the decision logic of the model.

Interactive Confusion Matrix: A Plotly heatmap that allows users to hover over specific errors to understand False Positives/Negatives.

Exploratory Data Analysis (EDA): Automatic correlation heatmaps and target distribution checks before training begins.

⚙️ Installation & Usage
Prerequisites: Python 3.8+

Clone the Repository

Bash

git clone https://github.com/yourusername/no-code-ml-pipeline.git
cd no-code-ml-pipeline
Install Dependencies

Bash

pip install -r requirements.txt
Run the Application

Bash

streamlit run app.py
How to Use

Upload a .csv file (e.g., Titanic, Iris, or Customer Churn).

Select your Target Column in the sidebar.

Adjust Preprocessing (Scaling) and Model Hyperparameters.

Click Run Pipeline.

🧠 Design Philosophy (Why Streamlit?)
As an ML Engineer, my primary focus is on the model architecture and data integrity rather than frontend boilerplate.

I chose Streamlit because it allows for rapid prototyping of data applications. It enables the creation of a clean, functional UI while keeping the codebase 100% Python, facilitating easier integration with backend ML libraries like PyTorch, TensorFlow, or Scikit-Learn in the future.

This approach mimics real-world industry workflows where ML Engineers build "Proof of Concept" (PoC) apps to demonstrate model value to stakeholders before handing off to full-stack teams for scaling.

🚀 Future Roadmap
[ ] Model Persistence: Add functionality to download the trained model as a .pkl file.

[ ] Deep Learning Support: Add a simple Neural Network builder (PyTorch) for more complex datasets.

[ ] Auto-Model Selection: Implement a "Search" feature that runs multiple models and picks the best one automatically (AutoML).



