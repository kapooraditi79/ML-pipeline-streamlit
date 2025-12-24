import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CUSTOM ML IMPLEMENTATIONS (for logistic)

class BinaryLogistic:
    def __init__(self, regularization=None, epochs=1000, lr_=0.01, lambda_reg=0.01):
        self.m = None
        self.b = None
        self.losses = []
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.lr_ = lr_
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_losses(self, y_train, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        n = len(y_train)
        loss = -1/n * (y_train @ np.log(y_pred) + (1 - y_train) @ np.log(1 - y_pred))

        if self.regularization == 'l2':
            loss += self.lambda_reg * np.sum(self.m**2)
        elif self.regularization == 'l1':
            loss += self.lambda_reg * np.sum(np.abs(self.m))
        return loss

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        # Xavier Initialization
        self.m = np.random.randn(n_features) * np.sqrt(1 / n_features)
        self.b = 0

        for epoch in range(self.epochs):
            linear_model = np.dot(X_train, self.m) + self.b
            y_pred = self.sigmoid(linear_model)

            # Gradients
            loss_slope_m = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
            loss_slope_b = (1 / n_samples) * np.sum(y_pred - y_train)

            # Regularization Gradients
            if self.regularization == 'l2':
                loss_slope_m += 2 * self.lambda_reg * self.m
            elif self.regularization == 'l1':
                loss_slope_m += self.lambda_reg * np.sign(self.m)

            # Update
            self.m -= self.lr_ * loss_slope_m
            self.b -= self.lr_ * loss_slope_b
            
            if epoch % 100 == 0:
                self.losses.append(self.compute_losses(y_train, y_pred))

    def predict_proba(self, X):
        linear_model = np.dot(X, self.m) + self.b
        return self.sigmoid(linear_model)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

class MulticlassLogistic:
    def __init__(self, lr=0.01, epochs=1000, regularization=None, lambda_=0.01):
        self.lr = lr
        self.epochs = epochs
        self.regularization = regularization
        self.lambda_ = lambda_
        self.models = []
        self.classes = None

    def fit(self, X_train, y_train):
        # Ensure numpy array
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.classes = np.unique(y_train)
        
        # One-vs-Rest Training
        for class_label in self.classes:
            binary_y = (y_train == class_label).astype(int)
            model = BinaryLogistic(lr_=self.lr, epochs=self.epochs, 
                                 regularization=self.regularization, lambda_reg=self.lambda_)
            model.fit(X_train, binary_y)
            self.models.append(model)

    def predict(self, X):
        X = np.array(X)
        probabilities = np.array([model.predict_proba(X) for model in self.models]).T
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes[class_indices]

# 2. HELPER FUNCTIONS (Visualization & Utils)

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def plot_confusion_matrix_plotly(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    cm_text = [[str(y) for y in x] for x in cm]
    
    fig = ff.create_annotated_heatmap(
        z=cm, 
        x=[str(l) for l in labels], 
        y=[str(l) for l in labels], 
        annotation_text=cm_text,
        colorscale='Viridis'
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def plot_feature_importance(model, feature_names):
    # Extract importance based on model type
    importance = None
    if hasattr(model, 'coef_'): # Logistic Regression
        importance = model.coef_[0]
    elif hasattr(model, 'feature_importances_'): # Decision Tree
        importance = model.feature_importances_
    
    if importance is not None:
        # Create DF and sort
        df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        df_imp['Abs_Importance'] = df_imp['Importance'].abs()
        df_imp = df_imp.sort_values(by='Abs_Importance', ascending=True).tail(15) # Top 15
        
        fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                     title="Feature Importance (Top 15)", color='Importance', color_continuous_scale='RdBu')
        return fig
    return None



# STREAMLIT UI (The Pipeline)

st.set_page_config(page_title="No-Code ML Builder", layout="wide")

st.title("🧩 No-Code ML Pipeline Builder")
st.markdown("""
Build a machine learning pipeline in minutes. 
**Upload Data ➡ Preprocess ➡ Train ➡ Evaluate**
""")

# SIDEBAR CONFIGURATION
st.sidebar.header("Pipeline Configuration")

# Dataset Upload
uploaded_file = st.sidebar.file_uploader("1. Upload Dataset", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.subheader("📊 Data Preview")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head())
        with col2:
            st.info(f"**Rows:** {df.shape[0]}\n\n**Columns:** {df.shape[1]}")

        # 2. Select Target & Features
        st.sidebar.subheader("2. Select Data")
        all_cols = df.columns.tolist()
        target_col = st.sidebar.selectbox("Select Target Column (y)", all_cols, index=len(all_cols)-1)
        
        # Identify types automatically
        X_raw = df.drop(columns=[target_col])
        y_raw = df[target_col]
        
        numeric_cols = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()

        # EDA Visuals
        with st.expander("🔎 Exploratory Data Analysis (EDA)", expanded=True):
            tab1, tab2 = st.tabs(["Target Distribution", "Correlation Heatmap"])
            
            with tab1:
                # Check for imbalance
                fig_target = px.histogram(df, x=target_col, color=target_col, title="Class Distribution")
                st.plotly_chart(fig_target, use_container_width=True)
                
            with tab2:
                if len(numeric_cols) > 1:
                    corr = X_raw[numeric_cols].corr()
                    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix")
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.write("Not enough numerical columns for correlation.")

        # 3. Preprocessing Settings
        st.sidebar.subheader("3. Preprocessing")
        scaler_type = st.sidebar.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler", "None"])
        
        # 4. Train-Test Split
        st.sidebar.subheader("4. Split Data")
        split_pct = st.sidebar.slider("Test Set Size (%)", 10, 50, 20) / 100.0
        
        # 5. Model Selection
        st.sidebar.subheader("5. Model Selection")
        model_name = st.sidebar.selectbox("Choose Algorithm", 
                                          ["Logistic Regression (Sklearn)", 
                                           "Logistic Regression (Custom - Multiclass Classification)", 
                                           "Decision Tree Classifier"])
        
        params = {}
        if "Decision Tree" in model_name:
            params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
            params['criterion']= st.sidebar.radio('Criterion', ['gini','entropy'])
            params['min_sample_split']= st.sidebar.slider('Min Samples to split',2, 100, 2, help="Minimum samples required to split a node. Higher= prevents overfitting. [but not too high]")
        elif "Custom" in model_name:
            params['lr'] = st.sidebar.number_input("Learning Rate", 0.0001, 1.0, 0.01, format="%.4f")
            params['epochs'] = st.sidebar.slider("Epochs", 100, 5000, 1000)

        # RUN PIPELINE BUTTON 
        if st.sidebar.button("🚀 Run Pipeline", type="primary"):
            st.divider()
            with st.spinner("Preprocessing data and training model..."):
                
                # Data Splitting
                # Encode Target if it's categorical strings
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_raw)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_raw, y_encoded, test_size=split_pct, stratify=y_encoded, random_state=42
                )
                
                st.success(f"**Data Split:** Train ({len(X_train)} samples) | Test ({len(X_test)} samples)")

                # Preprocessing Pipeline
                # Numeric: Impute (mean) -> Scale
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler() if scaler_type == "StandardScaler" else (MinMaxScaler() if scaler_type == "MinMaxScaler" else None))
                ])

                # Categorical: Impute (mode) -> OneHot
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False is crucial for custom code
                ])

                # Combine
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_cols),
                        ('cat', categorical_transformer, categorical_cols)
                    ],
                    remainder='drop' # Drop columns we don't know how to handle
                )

                # Apply transformations
                # Note: We must transform NOW to pass numpy arrays to your custom code
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)
                
                # Get feature names for visualization
                try:
                    num_names = numeric_cols
                    cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
                    feature_names = np.concatenate([num_names, cat_names])
                except:
                    feature_names = [f"Feature_{i}" for i in range(X_train_processed.shape[1])]

                # Model Training
                model = None
                
                if model_name == "Logistic Regression (Sklearn)":
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train_processed, y_train)
                    y_pred = model.predict(X_test_processed)
                    
                elif model_name == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(max_depth=params['max_depth'], 
                                                   criterion=params["criterion"],
                                                   min_samples_split=params['min_sample_split'],
                                                   random_state=42)
                    model.fit(X_train_processed, y_train)
                    y_pred = model.predict(X_test_processed)
                    
                elif model_name == "Logistic Regression (Custom - Your Code)":
                    # Use my custom Multiclass wrapper which handles Binary too via OVR
                    model = MulticlassLogistic(lr=params['lr'], epochs=params['epochs'])
                    try:
                        model.fit(X_train_processed, y_train)
                        y_pred = model.predict(X_test_processed)
                    except Exception as e:
                        st.error(f"Custom Model Error: {e}")
                        y_pred = None

                # Evaluation
                if y_pred is not None:
                    acc = accuracy_score(y_test, y_pred)
                    
                    st.subheader("🎯 Model Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Accuracy", f"{acc*100:.2f}%")
                    col2.metric("Train Size", len(X_train))
                    col3.metric("Test Size", len(X_test))

                    # 1. Confusion Matrix
                    st.write("### Confusion Matrix")
                    fig_cm = plot_confusion_matrix_plotly(y_test, y_pred, le.classes_)
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # 2. Classification Report
                    st.write("### Classification Report")
                    report_dict = classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.background_gradient(cmap='Blues'))
                    
                    # 3. Feature Importance (Only for Sklearn models for now)
                    if "Custom" not in model_name:
                        st.write("### Feature Importance")
                        fig_imp = plot_feature_importance(model, feature_names)
                        if fig_imp:
                            st.plotly_chart(fig_imp, use_container_width=True)

                    if model_name=='Decision Tree Classifier':
                        st.write("### Decision Tree Visualization")
                        st.write("The tree below shows how the model makes decisions. Max Depth: {params['max_depth']}")

                        fig_tree, ax = plt.subplots(figsize=(20, 10))
                        tree.plot_tree(model, 
                                      filled=True, 
                                      feature_names=feature_names, 
                                      class_names=[str(c) for c in le.classes_],
                                      rounded=True, 
                                      fontsize=10,
                                      ax=ax)
                        st.pyplot(fig_tree)
                    else:
                        st.info("Feature importance visualization is currently enabled for Sklearn models only.")

else:
    st.info(" Please upload a dataset from the sidebar to begin.")