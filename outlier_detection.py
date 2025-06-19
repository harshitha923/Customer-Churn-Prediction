import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def detect_outliers(data):
    """
    Detect outliers in telecom customer data using various algorithms
    """
    st.write("## Outlier Detection")
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Sidebar for outlier detection configuration
    st.sidebar.subheader("Outlier Detection Configuration")
    
    # Step 1: Feature Selection
    st.write("### Step 1: Feature Selection")
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target column from features if present
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    
    if 'Churn_Binary' in numerical_cols:
        numerical_cols.remove('Churn_Binary')
    
    # Let user select features for outlier detection
    st.write("Select features to use for outlier detection:")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Numerical Features")
        selected_num_cols = st.multiselect(
            "Select numerical features",
            numerical_cols,
            default=numerical_cols[:min(5, len(numerical_cols))]  # Default to first 5 or fewer
        )
    
    with col2:
        st.write("#### Categorical Features")
        selected_cat_cols = st.multiselect(
            "Select categorical features",
            categorical_cols,
            default=[]  # No defaults for categorical
        )
    
    # Step 2: Data Preprocessing
    st.write("### Step 2: Data Preprocessing")
    
    # Scaling option
    scaling_option = st.selectbox(
        "Select scaling method for numerical features",
        ["StandardScaler", "No scaling"]
    )
    
    # Handle categorical features
    if selected_cat_cols:
        encoding_option = st.selectbox(
            "Select encoding method for categorical features",
            ["One-Hot Encoding", "Skip categorical features"]
        )
    else:
        encoding_option = "Skip categorical features"
    
    # Step 3: Dimensionality Reduction (Optional)
    st.write("### Step 3: Dimensionality Reduction (Optional)")
    
    use_pca = st.checkbox("Apply PCA before outlier detection", value=False)
    
    if use_pca:
        # Only visible if PCA is selected
        n_components = st.slider(
            "Select number of principal components",
            min_value=2, 
            max_value=min(len(selected_num_cols) + (len(selected_cat_cols) if encoding_option == "One-Hot Encoding" else 0), 10),
            value=2
        )
    
    # Step 4: Outlier Detection Algorithm
    st.write("### Step 4: Outlier Detection Algorithm")
    
    algorithm = st.selectbox(
        "Select outlier detection algorithm",
        ["Isolation Forest", "Local Outlier Factor", "Z-Score", "IQR (Interquartile Range)"]
    )
    
    # Algorithm specific parameters
    if algorithm == "Isolation Forest":
        contamination = st.slider(
            "Contamination (expected proportion of outliers)",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01
        )
        
        n_estimators = st.slider(
            "Number of estimators",
            min_value=50,
            max_value=500,
            value=100,
            step=10
        )
    
    elif algorithm == "Local Outlier Factor":
        n_neighbors = st.slider(
            "Number of neighbors",
            min_value=5,
            max_value=50,
            value=20,
            step=1
        )
        
        contamination = st.slider(
            "Contamination (expected proportion of outliers)",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01
        )
    
    elif algorithm == "Z-Score":
        threshold = st.slider(
            "Z-Score threshold",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1
        )
    
    elif algorithm == "IQR (Interquartile Range)":
        iqr_multiplier = st.slider(
            "IQR multiplier",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1
        )
    
    # Execute outlier detection when button is clicked
    if st.button("Detect Outliers"):
        
        # Prepare data for outlier detection
        X = df.copy()
        
        # Process numerical features
        if selected_num_cols:
            num_X = X[selected_num_cols].copy()
            
            # Handle missing values in numerical features
            for col in selected_num_cols:
                if num_X[col].isnull().sum() > 0:
                    num_X[col] = num_X[col].fillna(num_X[col].median())
            
            # Apply scaling if selected
            if scaling_option == "StandardScaler":
                scaler = StandardScaler()
                num_X_scaled = scaler.fit_transform(num_X)
                num_X_scaled_df = pd.DataFrame(num_X_scaled, columns=selected_num_cols)
            else:  # No scaling
                num_X_scaled_df = num_X
        else:
            num_X_scaled_df = pd.DataFrame()
        
        # Process categorical features
        if selected_cat_cols and encoding_option == "One-Hot Encoding":
            cat_X = X[selected_cat_cols].copy()
            
            # Handle missing values in categorical features
            for col in selected_cat_cols:
                if cat_X[col].isnull().sum() > 0:
                    cat_X[col] = cat_X[col].fillna(cat_X[col].mode()[0])
            
            # Apply one-hot encoding
            cat_X_encoded = pd.get_dummies(cat_X, drop_first=False)
        else:
            cat_X_encoded = pd.DataFrame()
        
        # Combine numerical and categorical features
        if not num_X_scaled_df.empty and not cat_X_encoded.empty:
            X_prepared = pd.concat([num_X_scaled_df.reset_index(drop=True), 
                                    cat_X_encoded.reset_index(drop=True)], axis=1)
        elif not num_X_scaled_df.empty:
            X_prepared = num_X_scaled_df
        elif not cat_X_encoded.empty:
            X_prepared = cat_X_encoded
        else:
            st.error("No features selected for outlier detection.")
            return
        
        # Apply PCA if selected
        if use_pca:
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_prepared)
            
            # Show explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            st.write("#### PCA Explained Variance")
            
            # Create DataFrame for variance plot
            variance_df = pd.DataFrame({
                'Component': [f"PC{i+1}" for i in range(len(explained_variance))],
                'Explained Variance': explained_variance,
                'Cumulative Variance': cumulative_variance
            })
            
            # Plot explained variance
            fig = px.bar(variance_df, x='Component', y='Explained Variance',
                        title="Explained Variance by Principal Component",
                        labels={'Explained Variance': 'Proportion of Variance Explained'},
                        color='Explained Variance',
                        color_continuous_scale=px.colors.sequential.Viridis)
            
            # Add cumulative variance line
            fig.add_trace(
                go.Scatter(
                    x=variance_df['Component'],
                    y=variance_df['Cumulative Variance'],
                    mode='lines+markers',
                    name='Cumulative Variance',
                    yaxis='y2',
                    line=dict(color='red', width=2)
                )
            )
            
            # Update layout for second y-axis
            fig.update_layout(
                yaxis2=dict(
                    title='Cumulative Variance',
                    overlaying='y',
                    side='right',
                    range=[0, 1],
                    showgrid=False
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Use PCA results for outlier detection
            X_for_detection = X_pca
            feature_names = [f"PC{i+1}" for i in range(n_components)]
        else:
            X_for_detection = X_prepared.values
            feature_names = X_prepared.columns.tolist()
        
        # Perform outlier detection
        with st.spinner("Detecting outliers..."):
            if algorithm == "Isolation Forest":
                # Fit Isolation Forest
                clf = IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=42
                )
                
                # Predict outliers (1 for inliers, -1 for outliers)
                outliers = clf.fit_predict(X_for_detection)
                
                # Convert to boolean (True for outliers)
                is_outlier = outliers == -1
                
                # Get outlier scores (negative of decision function)
                outlier_scores = -clf.decision_function(X_for_detection)
                
                # Display model information
                st.write(f"#### Isolation Forest Results")
                st.write(f"Detected {sum(is_outlier)} outliers out of {len(is_outlier)} samples ({sum(is_outlier)/len(is_outlier)*100:.2f}%)")
                
            elif algorithm == "Local Outlier Factor":
                # Fit Local Outlier Factor
                clf = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contamination
                )
                
                # Predict outliers (1 for inliers, -1 for outliers)
                outliers = clf.fit_predict(X_for_detection)
                
                # Convert to boolean (True for outliers)
                is_outlier = outliers == -1
                
                # Get outlier scores (negative of local outlier factor)
                outlier_scores = clf.negative_outlier_factor_
                
                # Display model information
                st.write(f"#### Local Outlier Factor Results")
                st.write(f"Detected {sum(is_outlier)} outliers out of {len(is_outlier)} samples ({sum(is_outlier)/len(is_outlier)*100:.2f}%)")
                
            elif algorithm == "Z-Score":
                # Calculate Z-scores for each feature
                z_scores = np.zeros(X_for_detection.shape)
                
                for i in range(X_for_detection.shape[1]):
                    mean = np.mean(X_for_detection[:, i])
                    std = np.std(X_for_detection[:, i])
                    
                    if std > 0:  # Avoid division by zero
                        z_scores[:, i] = (X_for_detection[:, i] - mean) / std
                    else:
                        z_scores[:, i] = 0
                
                # Take the maximum absolute Z-score across all features
                max_z_scores = np.max(np.abs(z_scores), axis=1)
                
                # Mark as outlier if max Z-score exceeds threshold
                is_outlier = max_z_scores > threshold
                
                # Use max Z-score as outlier score
                outlier_scores = max_z_scores
                
                # Display model information
                st.write(f"#### Z-Score Results (threshold = {threshold})")
                st.write(f"Detected {sum(is_outlier)} outliers out of {len(is_outlier)} samples ({sum(is_outlier)/len(is_outlier)*100:.2f}%)")
                
            elif algorithm == "IQR (Interquartile Range)":
                # Calculate IQR outliers for each feature
                feature_outliers = np.zeros(X_for_detection.shape, dtype=bool)
                
                for i in range(X_for_detection.shape[1]):
                    q1 = np.percentile(X_for_detection[:, i], 25)
                    q3 = np.percentile(X_for_detection[:, i], 75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - iqr_multiplier * iqr
                    upper_bound = q3 + iqr_multiplier * iqr
                    
                    feature_outliers[:, i] = (X_for_detection[:, i] < lower_bound) | (X_for_detection[:, i] > upper_bound)
                
                # Mark as outlier if it's an outlier in any feature
                is_outlier = np.any(feature_outliers, axis=1)
                
                # Count outlier features for each sample
                outlier_scores = np.sum(feature_outliers, axis=1) / X_for_detection.shape[1]
                
                # Display model information
                st.write(f"#### IQR Results (multiplier = {iqr_multiplier})")
                st.write(f"Detected {sum(is_outlier)} outliers out of {len(is_outlier)} samples ({sum(is_outlier)/len(is_outlier)*100:.2f}%)")
            
            # Add outlier information to original data
            X['is_outlier'] = is_outlier
            X['outlier_score'] = outlier_scores
            
            # Display outlier distribution
            st.write("#### Outlier Score Distribution")
            
            # Create histogram of outlier scores
            fig = px.histogram(
                X, 
                x='outlier_score',
                color='is_outlier',
                marginal="box",
                title="Distribution of Outlier Scores",
                labels={'outlier_score': 'Outlier Score'},
                color_discrete_map={True: 'red', False: 'blue'},
                category_orders={"is_outlier": [False, True]}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualize outliers
            st.write("#### Outlier Visualization")
            
            if use_pca:
                # PCA was already applied, use first two components
                if n_components >= 2:
                    # Create scatter plot
                    fig = px.scatter(
                        x=X_for_detection[:, 0],
                        y=X_for_detection[:, 1],
                        color=X['is_outlier'],
                        title=f"Outlier Visualization (PCA)",
                        labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
                        color_discrete_map={True: 'red', False: 'blue'},
                        category_orders={"is_outlier": [False, True]},
                        opacity=0.7,
                        size=X['outlier_score'] * 10 + 5
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create 3D visualization if 3 or more components
                    if n_components >= 3:
                        fig_3d = px.scatter_3d(
                            x=X_for_detection[:, 0],
                            y=X_for_detection[:, 1],
                            z=X_for_detection[:, 2],
                            color=X['is_outlier'],
                            title=f"3D Outlier Visualization (PCA)",
                            labels={
                                'x': 'Principal Component 1', 
                                'y': 'Principal Component 2', 
                                'z': 'Principal Component 3'
                            },
                            color_discrete_map={True: 'red', False: 'blue'},
                            category_orders={"is_outlier": [False, True]},
                            opacity=0.7,
                            size=X['outlier_score'] * 10 + 5
                        )
                        
                        # Update layout for better visualization
                        fig_3d.update_layout(
                            margin=dict(l=0, r=0, b=0, t=40),
                            scene=dict(
                                xaxis_title='PC1',
                                yaxis_title='PC2',
                                zaxis_title='PC3'
                            )
                        )
                        
                        st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("Not enough PCA components for visualization.")
            else:
                # Use the first two numerical features for visualization
                if len(selected_num_cols) >= 2:
                    x_col, y_col = selected_num_cols[0], selected_num_cols[1]
                    
                    fig = px.scatter(
                        X,
                        x=x_col,
                        y=y_col,
                        color='is_outlier',
                        title=f"Outlier Visualization ({x_col} vs {y_col})",
                        labels={x_col: x_col, y_col: y_col},
                        color_discrete_map={True: 'red', False: 'blue'},
                        category_orders={"is_outlier": [False, True]},
                        opacity=0.7,
                        size=X['outlier_score']  * 10 + 5
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create 3D visualization if 3 or more features
                    if len(selected_num_cols) >= 3:
                        z_col = selected_num_cols[2]
                        
                        fig_3d = px.scatter_3d(
                            X,
                            x=x_col,
                            y=y_col,
                            z=z_col,
                            color='is_outlier',
                            title=f"3D Outlier Visualization",
                            labels={x_col: x_col, y_col: y_col, z_col: z_col},
                            color_discrete_map={True: 'red', False: 'blue'},
                            category_orders={"is_outlier": [False, True]},
                            opacity=0.7,
                            size=X['outlier_score'] * 10 + 5
                        )
                        
                        st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.warning("Not enough numerical features for visualization.")
            
            # Feature-wise outlier analysis
            if len(selected_num_cols) > 0:
                st.write("#### Feature-wise Outlier Analysis")
                
                # Create box plots for each numerical feature
                with st.expander("Show Box Plots of Numerical Features"):
                    for feature in selected_num_cols:
                        fig = px.box(
                            X,
                            x='is_outlier',
                            y=feature,
                            color='is_outlier',
                            title=f"Distribution of {feature} by Outlier Status",
                            color_discrete_map={True: 'red', False: 'blue'},
                            category_orders={"is_outlier": [False, True]}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # Analyze outliers
            st.write("#### Outlier Analysis")
            
            # Calculate statistics for outliers vs non-outliers
            outlier_stats = X.groupby('is_outlier')[selected_num_cols].agg(['mean', 'median', 'std']).reset_index()
            
            # Display statistics in an expander
            with st.expander("Show Statistical Summary of Outliers vs Non-Outliers"):
                st.dataframe(outlier_stats)
            
            # Check if Churn column exists
            if 'Churn' in df.columns:
                # Calculate churn rate for outliers vs non-outliers
                if df['Churn'].dtype == 'object':
                    # Convert to binary (1 if 'Yes', 0 otherwise)
                    X['Churn_Binary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
                else:
                    X['Churn_Binary'] = df['Churn']
                
                # Calculate churn rate
                churn_by_outlier = X.groupby('is_outlier')['Churn_Binary'].mean().reset_index()
                churn_by_outlier['Churn Rate (%)'] = churn_by_outlier['Churn_Binary'] * 100
                
                # Display churn rate by outlier status
                st.write("#### Churn Rate by Outlier Status")
                
                fig = px.bar(
                    churn_by_outlier,
                    x='is_outlier',
                    y='Churn Rate (%)',
                    color='is_outlier',
                    title="Churn Rate by Outlier Status",
                    labels={'is_outlier': 'Is Outlier', 'Churn Rate (%)': 'Churn Rate (%)'},
                    color_discrete_map={True: 'red', False: 'blue'},
                    category_orders={"is_outlier": [False, True]},
                    text_auto='.1f'
                )
                
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                fig.update_layout(yaxis_range=[0, 100])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate relative risk of churn for outliers
                churn_rate_outliers = churn_by_outlier[churn_by_outlier['is_outlier']]['Churn Rate (%)'].values[0]
                churn_rate_non_outliers = churn_by_outlier[~churn_by_outlier['is_outlier']]['Churn Rate (%)'].values[0]
                
                if churn_rate_non_outliers > 0:
                    relative_risk = churn_rate_outliers / churn_rate_non_outliers
                    
                    if relative_risk > 1.5:
                        st.warning(f"Outliers are {relative_risk:.2f}x more likely to churn than non-outliers!")
                    elif relative_risk < 0.67:
                        st.success(f"Outliers are {1/relative_risk:.2f}x less likely to churn than non-outliers.")
                    else:
                        st.info(f"Outliers have a similar churn rate to non-outliers (relative risk: {relative_risk:.2f}x).")
            
            # Display top outliers
            st.write("#### Top Outliers")
            
            # Sort by outlier score
            top_outliers = X[X['is_outlier']].sort_values('outlier_score', ascending=False)
            
            # Check if any outliers were detected
            if len(top_outliers) > 0:
                # Display top 10 outliers or fewer if less than 10
                st.dataframe(top_outliers.head(min(10, len(top_outliers))).drop(['is_outlier', 'outlier_score'], axis=1))
                
                # Analyze characteristics of outliers
                st.write("#### Outlier Characteristics")
                
                # Create a summary of categorical features for outliers vs non-outliers
                if selected_cat_cols:
                    for cat_col in selected_cat_cols:
                        # Create cross tabulation
                        cross_tab = pd.crosstab(
                            X['is_outlier'], 
                            X[cat_col], 
                            normalize='index'
                        ).reset_index()
                        
                        # Convert to long format for plotting
                        cross_tab_long = pd.melt(
                            cross_tab, 
                            id_vars=['is_outlier'],
                            var_name=cat_col,
                            value_name='Proportion'
                        )
                        
                        # Create grouped bar chart
                        fig = px.bar(
                            cross_tab_long,
                            x=cat_col,
                            y='Proportion',
                            color='is_outlier',
                            barmode='group',
                            title=f"Distribution of {cat_col} by Outlier Status",
                            labels={'Proportion': 'Proportion', cat_col: cat_col},
                            color_discrete_map={True: 'red', False: 'blue'},
                            category_orders={"is_outlier": [False, True]}
                        )
                        
                        # Update y-axis to percentage
                        fig.update_layout(yaxis_tickformat='.0%')
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Business insights
                st.write("#### Business Insights")
                
                # Calculate average monthly and total charges for outliers vs non-outliers
                if 'MonthlyCharges' in X.columns or 'monthlycharges' in X.columns.str.lower():
                    monthly_col = X.columns[X.columns.str.lower() == 'monthlycharges'][0]
                    
                    monthly_charges_by_outlier = X.groupby('is_outlier')[monthly_col].mean().reset_index()
                    monthly_charges_by_outlier['Average Monthly Charges'] = monthly_charges_by_outlier[monthly_col]
                    
                    fig = px.bar(
                        monthly_charges_by_outlier,
                        x='is_outlier',
                        y='Average Monthly Charges',
                        color='is_outlier',
                        title="Average Monthly Charges by Outlier Status",
                        labels={'is_outlier': 'Is Outlier', 'Average Monthly Charges': 'Average Monthly Charges'},
                        color_discrete_map={True: 'red', False: 'blue'},
                        category_orders={"is_outlier": [False, True]},
                        text_auto='.2f'
                    )
                    
                    fig.update_traces(texttemplate='$%{text}', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'TotalCharges' in X.columns or 'totalcharges' in X.columns.str.lower():
                    total_col = X.columns[X.columns.str.lower() == 'totalcharges'][0]
                    
                    # Convert to numeric if it's not already
                    if X[total_col].dtype == 'object':
                        X[total_col] = pd.to_numeric(X[total_col], errors='coerce')
                    
                    total_charges_by_outlier = X.groupby('is_outlier')[total_col].mean().reset_index()
                    total_charges_by_outlier['Average Total Charges'] = total_charges_by_outlier[total_col]
                    
                    fig = px.bar(
                        total_charges_by_outlier,
                        x='is_outlier',
                        y='Average Total Charges',
                        color='is_outlier',
                        title="Average Total Charges by Outlier Status",
                        labels={'is_outlier': 'Is Outlier', 'Average Total Charges': 'Average Total Charges'},
                        color_discrete_map={True: 'red', False: 'blue'},
                        category_orders={"is_outlier": [False, True]},
                        text_auto='.2f'
                    )
                    
                    fig.update_traces(texttemplate='$%{text}', textposition='outside')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.write("#### Recommendations")
                
                st.markdown("""
                Based on the outlier analysis, consider the following actions:
                
                1. **Investigate Unusual Patterns**: Review the top outliers to understand unusual customer behavior patterns.
                
                2. **Customer Segmentation**: Consider creating a separate segment for outlier customers with specialized retention strategies.
                
                3. **Fraud Detection**: Some outliers may represent fraudulent activity or data entry errors that should be addressed.
                
                4. **Service Customization**: For legitimate outlier customers, develop customized service packages that better meet their unique needs.
                
                5. **Data Quality Improvement**: Use outlier detection regularly to identify and correct data quality issues.
                """)
            else:
                st.info("No outliers were detected with the current settings. Try adjusting the algorithm parameters.")
            
            # Add download option
            X_download = X.copy()
            X_download['outlier_score'] = X_download['outlier_score'].round(4)
            
            csv = X_download.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Download Data with Outlier Information",
                data=csv,
                file_name="telecom_outliers.csv",
                mime="text/csv"
            )
