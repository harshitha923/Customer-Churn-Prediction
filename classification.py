import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

def build_classification_models(data):
    """
    Build and evaluate classification models for churn prediction
    """
    st.write("## Churn Classification")
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Check if 'Churn' column exists
    if 'Churn' not in df.columns:
        st.error("Churn column not found in the dataset. Cannot perform classification.")
        return
    
    # Convert 'Churn' to binary (1 if 'Yes', 0 otherwise)
    if df['Churn'].dtype == 'object':
        df['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    else:
        df['Churn_Binary'] = df['Churn']
    
    # Sidebar for model configuration
    st.sidebar.subheader("Model Configuration")
    
    # Step 1: Feature Selection
    st.write("### Step 1: Feature Selection")
    
    # Separate categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ['Churn']]
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in ['Churn_Binary']]
    
    # Display columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Categorical Features")
        selected_cat_cols = st.multiselect(
            "Select categorical features",
            categorical_cols,
            default=categorical_cols
        )
    
    with col2:
        st.write("#### Numerical Features")
        selected_num_cols = st.multiselect(
            "Select numerical features",
            numerical_cols,
            default=numerical_cols
        )
    
    # Step 2: Data Preprocessing
    st.write("### Step 2: Data Preprocessing")
    
    # Missing values handling
    missing_strategy = st.selectbox(
        "Missing Values Strategy",
        ["Mean/Mode imputation", "Median/Most Frequent imputation", "Drop rows with missing values"]
    )
    
    # Scaling option
    scaling_option = st.selectbox(
        "Scaling Option for Numerical Features",
        ["StandardScaler", "MinMaxScaler", "No scaling"]
    )
    
    # Train-test split
    test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    
    # Step 3: Model Selection
    st.write("### Step 3: Model Selection")
    
    # Model selection
    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Support Vector Machine": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB()
    }
    
    selected_model = st.selectbox("Select model", list(model_options.keys()))
    
    # Advanced options expander
    with st.expander("Advanced Model Configuration"):
        if selected_model == "Logistic Regression":
            C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.01)
            solver = st.selectbox("Solver", ["liblinear", "newton-cg", "lbfgs", "sag", "saga"])
            model = LogisticRegression(C=C, solver=solver, max_iter=1000)
        
        elif selected_model == "Random Forest":
            n_estimators = st.slider("Number of trees", 50, 500, 100, 10)
            max_depth = st.slider("Maximum depth", 1, 50, 10, 1)
            min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        
        elif selected_model == "Gradient Boosting":
            n_estimators = st.slider("Number of boosting stages", 50, 500, 100, 10)
            learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
            max_depth = st.slider("Maximum depth", 1, 10, 3, 1)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        
        elif selected_model == "Support Vector Machine":
            C = st.slider("Regularization parameter (C)", 0.1, 10.0, 1.0, 0.1)
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = SVC(C=C, kernel=kernel, probability=True)
        
        elif selected_model == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of neighbors", 1, 50, 5, 1)
            weights = st.selectbox("Weight function", ["uniform", "distance"])
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        
        elif selected_model == "Decision Tree":
            max_depth = st.slider("Maximum depth", 1, 50, 10, 1)
            min_samples_split = st.slider("Minimum samples to split", 2, 20, 2, 1)
            criterion = st.selectbox("Criterion", ["gini", "entropy"])
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)
        
        elif selected_model == "Naive Bayes":
            model = GaussianNB()
    
    # Step 4: Training and Evaluation
    if st.button("Train and Evaluate Model"):
        st.write("### Step 4: Training and Evaluation")
        
        # Prepare the feature matrix and target variable
        X = df[selected_cat_cols + selected_num_cols]
        y = df['Churn_Binary']
        
        # Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(
            selected_num_cols,
            selected_cat_cols,
            missing_strategy,
            scaling_option
        )
        
        # Create the full pipeline
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        with st.spinner("Training model..."):
            # Train the model
            full_pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = full_pipeline.predict(X_test)
            y_pred_prob = full_pipeline.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Calculate ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            
            # Store model in session state
            st.session_state.model = full_pipeline
            
            # Display metrics
            st.write("#### Model Performance Metrics")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
            
            with metrics_col2:
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
            
            st.metric("AUC-ROC", f"{roc_auc:.4f}")
            
            # Display confusion matrix
            st.write("#### Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, 
                                index=['Actual Negative', 'Actual Positive'], 
                                columns=['Predicted Negative', 'Predicted Positive'])
            
            fig = px.imshow(cm_df, 
                           text_auto=True, 
                           color_continuous_scale='Blues',
                           title="Confusion Matrix")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display classification report
            st.write("#### Classification Report")
            
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Display ROC curve
            st.write("#### ROC Curve")
            
            fig = px.area(
                x=fpr, y=tpr,
                title=f'ROC Curve (AUC={roc_auc:.4f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate'),
                width=700, height=500
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance for applicable models
            if selected_model in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
                st.write("#### Feature Importance")
                
                # Extract feature names after transformation
                feature_names = get_feature_names(full_pipeline, X)
                
                # Extract feature importances
                importances = full_pipeline.named_steps['classifier'].feature_importances_
                
                # Create DataFrame for importance
                importance_df = pd.DataFrame({
                    'Feature': feature_names[:len(importances)],  # Ensure matching length
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                            title='Feature Importance',
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Viridis')
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Step 5: Model Deployment and Prediction
        st.write("### Step 5: Interactive Prediction")
        
        # Create interactive prediction form
        with st.form("prediction_form"):
            st.write("Enter customer information to predict churn probability:")
            
            # Create input fields for each feature
            prediction_inputs = {}
            
            # Group features into categories for better organization
            demographic_cols = [col for col in selected_cat_cols + selected_num_cols 
                              if any(term in col.lower() for term in ['gender', 'senior', 'partner', 'dependent'])]
            
            service_cols = [col for col in selected_cat_cols + selected_num_cols 
                          if any(term in col.lower() for term in ['service', 'internet', 'online', 'device', 'tech', 'streaming', 'multiple'])]
            
            contract_cols = [col for col in selected_cat_cols + selected_num_cols 
                           if any(term in col.lower() for term in ['contract', 'billing', 'payment'])]
            
            other_cols = [col for col in selected_cat_cols + selected_num_cols 
                        if col not in demographic_cols + service_cols + contract_cols]
            
            # Demographics section
            if demographic_cols:
                st.write("#### Demographics")
                demo_cols = st.columns(min(3, len(demographic_cols)))
                
                for i, col in enumerate(demographic_cols):
                    with demo_cols[i % len(demo_cols)]:
                        if col in numerical_cols:
                            # For numerical features
                            min_val = float(df[col].min())
                            max_val = float(df[col].max())
                            mean_val = float(df[col].mean())
                            
                            prediction_inputs[col] = st.number_input(
                                f"{col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                step=(max_val - min_val) / 100
                            )
                        else:
                            # For categorical features
                            unique_values = df[col].unique().tolist()
                            prediction_inputs[col] = st.selectbox(
                                f"{col}",
                                unique_values,
                                index=0
                            )
            
            # Services section
            if service_cols:
                st.write("#### Services")
                service_cols_list = st.columns(min(3, len(service_cols)))
                
                for i, col in enumerate(service_cols):
                    with service_cols_list[i % len(service_cols_list)]:
                        if col in numerical_cols:
                            # For numerical features
                            min_val = float(df[col].min())
                            max_val = float(df[col].max())
                            mean_val = float(df[col].mean())
                            
                            prediction_inputs[col] = st.number_input(
                                f"{col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                step=(max_val - min_val) / 100
                            )
                        else:
                            # For categorical features
                            unique_values = df[col].unique().tolist()
                            prediction_inputs[col] = st.selectbox(
                                f"{col}",
                                unique_values,
                                index=0
                            )
            
            # Contract section
            if contract_cols:
                st.write("#### Contract Information")
                contract_cols_list = st.columns(min(3, len(contract_cols)))
                
                for i, col in enumerate(contract_cols):
                    with contract_cols_list[i % len(contract_cols_list)]:
                        if col in numerical_cols:
                            # For numerical features
                            min_val = float(df[col].min())
                            max_val = float(df[col].max())
                            mean_val = float(df[col].mean())
                            
                            prediction_inputs[col] = st.number_input(
                                f"{col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                step=(max_val - min_val) / 100
                            )
                        else:
                            # For categorical features
                            unique_values = df[col].unique().tolist()
                            prediction_inputs[col] = st.selectbox(
                                f"{col}",
                                unique_values,
                                index=0
                            )
            
            # Other section
            if other_cols:
                st.write("#### Other Information")
                other_cols_list = st.columns(min(3, len(other_cols)))
                
                for i, col in enumerate(other_cols):
                    with other_cols_list[i % len(other_cols_list)]:
                        if col in numerical_cols:
                            # For numerical features
                            min_val = float(df[col].min())
                            max_val = float(df[col].max())
                            mean_val = float(df[col].mean())
                            
                            prediction_inputs[col] = st.number_input(
                                f"{col}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                step=(max_val - min_val) / 100
                            )
                        else:
                            # For categorical features
                            unique_values = df[col].unique().tolist()
                            prediction_inputs[col] = st.selectbox(
                                f"{col}",
                                unique_values,
                                index=0
                            )
            
            # Submit button
            submitted = st.form_submit_button("Predict Churn Probability")
        
        # Make prediction if form is submitted
        if submitted:
            # Create a DataFrame with the input data
            input_df = pd.DataFrame([prediction_inputs])
            
            # Make prediction
            churn_probability = full_pipeline.predict_proba(input_df)[0, 1]
            churn_prediction = "Yes" if churn_probability >= 0.5 else "No"
            
            # Display prediction
            st.write("#### Prediction Result")
            
            # Create columns for visual layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Display probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=churn_probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Churn Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display prediction result
                st.markdown(f"### Prediction: {churn_prediction}")
                st.markdown(f"Probability: {churn_probability:.2%}")
                
                # Risk level
                if churn_probability < 0.3:
                    risk_level = "Low Risk"
                    color = "green"
                elif churn_probability < 0.7:
                    risk_level = "Medium Risk"
                    color = "orange"
                else:
                    risk_level = "High Risk"
                    color = "red"
                
                st.markdown(f"Risk Level: <span style='color:{color};font-weight:bold'>{risk_level}</span>", unsafe_allow_html=True)
                
                # Recommended actions based on risk level
                st.write("#### Recommended Actions")
                
                if risk_level == "Low Risk":
                    st.markdown("""
                    - Continue regular service
                    - Occasionally check for satisfaction
                    - Offer complementary upgrades when available
                    """)
                elif risk_level == "Medium Risk":
                    st.markdown("""
                    - Reach out to assess satisfaction
                    - Offer loyalty discounts
                    - Provide personalized service recommendations
                    - Review billing issues if applicable
                    """)
                else:  # High Risk
                    st.markdown("""
                    - Immediate retention campaign
                    - Offer significant contract incentives
                    - Schedule personal follow-up
                    - Address service issues promptly
                    - Consider special retention package
                    """)
            
            # Get most similar customers for context
            st.write("#### Similar Customer Profiles")
            
            # Create a function to calculate similarity
            def calculate_similarity(row):
                similarity = 0
                
                for col in selected_num_cols:
                    # Calculate normalized difference for numerical columns
                    max_val = df[col].max()
                    min_val = df[col].min()
                    range_val = max_val - min_val
                    
                    if range_val > 0:
                        normalized_diff = abs(row[col] - prediction_inputs[col]) / range_val
                        similarity += normalized_diff
                
                for col in selected_cat_cols:
                    # Add 1 if categories don't match
                    if row[col] != prediction_inputs[col]:
                        similarity += 1
                
                return similarity
            
            # Calculate similarity for each row
            df['similarity'] = df.apply(calculate_similarity, axis=1)
            
            # Get top 5 most similar customers
            similar_customers = df.sort_values('similarity').head(5)
            
            # Display similar customers
            st.dataframe(similar_customers[['Churn'] + selected_cat_cols + selected_num_cols])

def create_preprocessing_pipeline(numerical_cols, categorical_cols, missing_strategy, scaling_option):
    """
    Create a preprocessing pipeline based on user selections
    """
    # Define numeric transformer
    if missing_strategy == "Mean/Mode imputation":
        num_imputer = SimpleImputer(strategy='mean')
    elif missing_strategy == "Median/Most Frequent imputation":
        num_imputer = SimpleImputer(strategy='median')
    else:  # Drop rows with missing values - handled before pipeline
        num_imputer = SimpleImputer(strategy='mean')  # Default, won't be used if rows are dropped
    
    if scaling_option == "StandardScaler":
        numeric_transformer = Pipeline(steps=[
            ('imputer', num_imputer),
            ('scaler', StandardScaler())
        ])
    elif scaling_option == "MinMaxScaler":
        numeric_transformer = Pipeline(steps=[
            ('imputer', num_imputer),
            ('scaler', MinMaxScaler())
        ])
    else:  # No scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', num_imputer)
        ])
    
    # Define categorical transformer
    if missing_strategy == "Mean/Mode imputation" or missing_strategy == "Median/Most Frequent imputation":
        cat_imputer = SimpleImputer(strategy='most_frequent')
    else:  # Drop rows with missing values - handled before pipeline
        cat_imputer = SimpleImputer(strategy='most_frequent')  # Default, won't be used if rows are dropped
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', cat_imputer),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def get_feature_names(pipeline, X):
    """
    Get feature names from pipeline after one-hot encoding
    """
    # Get the column transformer
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get the one-hot encoder
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        
        # Get the numerical and categorical column names
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Get the one-hot encoded feature names
        cat_features = []
        for i, col in enumerate(cat_cols):
            values = ohe.categories_[i]
            for val in values:
                cat_features.append(f"{col}_{val}")
        
        # Combine with numerical column names
        feature_names = num_cols + cat_features
        
        return feature_names
    except:
        # If any error occurs, return column names as is
        return X.columns.tolist()
