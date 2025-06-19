import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def perform_clustering(data):
    """
    Perform customer segmentation using clustering algorithms
    """
    st.write("## Customer Segmentation")
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Sidebar for clustering configuration
    st.sidebar.subheader("Clustering Configuration")
    
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
    
    # Let user select features for clustering
    st.write("Select features to use for clustering:")
    
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
        ["StandardScaler", "MinMaxScaler", "No scaling"]
    )
    
    # Handle categorical features
    if selected_cat_cols:
        encoding_option = st.selectbox(
            "Select encoding method for categorical features",
            ["One-Hot Encoding", "Skip categorical features"]
        )
    else:
        encoding_option = "Skip categorical features"
    
    # Step 3: Dimensionality Reduction
    st.write("### Step 3: Dimensionality Reduction (Optional)")
    
    use_pca = st.checkbox("Apply PCA before clustering", value=False)
    
    if use_pca:
        # Only visible if PCA is selected
        n_components = st.slider(
            "Select number of principal components",
            min_value=2, 
            max_value=min(len(selected_num_cols) + (len(selected_cat_cols) if encoding_option == "One-Hot Encoding" else 0), 10),
            value=2
        )
    
    # Step 4: Clustering Algorithm
    st.write("### Step 4: Clustering Algorithm")
    
    algorithm = st.selectbox(
        "Select clustering algorithm",
        ["K-Means", "Hierarchical Clustering", "DBSCAN"]
    )
    
    # Algorithm specific parameters
    if algorithm == "K-Means":
        n_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3)
        
        # Optional: elbow method
        show_elbow = st.checkbox("Show Elbow Method plot to help select k", value=True)
    
    elif algorithm == "Hierarchical Clustering":
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        linkage = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
    
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon (neighborhood size)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        min_samples = st.slider("Minimum samples in neighborhood", min_value=2, max_value=20, value=5)
    
    # Execute clustering when button is clicked
    if st.button("Perform Clustering"):
        
        # Prepare data for clustering
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
            elif scaling_option == "MinMaxScaler":
                scaler = MinMaxScaler()
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
            st.error("No features selected for clustering.")
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
            
            # Use PCA results for clustering
            X_for_clustering = X_pca
            feature_names = [f"PC{i+1}" for i in range(n_components)]
        else:
            X_for_clustering = X_prepared.values
            feature_names = X_prepared.columns.tolist()
        
        # Perform clustering
        if algorithm == "K-Means":
            
            # Show elbow method if requested
            if show_elbow:
                st.write("#### Elbow Method for K Selection")
                
                inertia = []
                silhouette = []
                k_range = range(2, 11)
                
                for k in k_range:
                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_temp.fit(X_for_clustering)
                    inertia.append(kmeans_temp.inertia_)
                    
                    # Calculate silhouette score
                    if len(np.unique(kmeans_temp.labels_)) > 1:  # Need at least 2 clusters
                        silhouette.append(silhouette_score(X_for_clustering, kmeans_temp.labels_))
                    else:
                        silhouette.append(0)
                
                # Create elbow method plot
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add inertia line
                fig.add_trace(
                    go.Scatter(
                        x=list(k_range),
                        y=inertia,
                        mode='lines+markers',
                        name='Inertia',
                        line=dict(color='blue', width=2)
                    )
                )
                
                # Add silhouette score line on secondary y-axis
                fig.add_trace(
                    go.Scatter(
                        x=list(k_range),
                        y=silhouette,
                        mode='lines+markers',
                        name='Silhouette Score',
                        line=dict(color='red', width=2)
                    ),
                    secondary_y=True
                )
                
                # Update layout
                fig.update_layout(
                    title="Elbow Method and Silhouette Score for K Selection",
                    xaxis_title="Number of Clusters (k)",
                    yaxis_title="Inertia (Sum of Squared Distances)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **How to interpret:** 
                - Look for an 'elbow' point where inertia (blue) starts decreasing more slowly
                - Higher silhouette scores (red) indicate better clustering
                - Choose a k value that balances both metrics
                """)
            
            # Fit KMeans with selected parameters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_for_clustering)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_for_clustering, clusters)
            
            # Store cluster centers
            if use_pca:
                centers = kmeans.cluster_centers_
            else:
                centers = kmeans.cluster_centers_
            
            # Add cluster labels to original data
            X['Cluster'] = clusters
            
            # Display model information
            st.write(f"#### K-Means Clustering Results (k={n_clusters})")
            st.write(f"Silhouette Score: {silhouette_avg:.4f}")
            
        elif algorithm == "Hierarchical Clustering":
            # Fit hierarchical clustering
            hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            clusters = hc.fit_predict(X_for_clustering)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_for_clustering, clusters)
            
            # Add cluster labels to original data
            X['Cluster'] = clusters
            
            # Display model information
            st.write(f"#### Hierarchical Clustering Results (clusters={n_clusters}, linkage={linkage})")
            st.write(f"Silhouette Score: {silhouette_avg:.4f}")
            
        elif algorithm == "DBSCAN":
            # Fit DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_for_clustering)
            
            # Calculate number of clusters (excluding noise points with label -1)
            n_clusters_dbscan = len(set(clusters)) - (1 if -1 in clusters else 0)
            
            # Calculate silhouette score if more than one cluster (excluding noise)
            if n_clusters_dbscan > 1 and not all(label == -1 for label in clusters):
                # Get indices of points that are not noise
                non_noise_indices = clusters != -1
                
                if sum(non_noise_indices) > 1 and len(set(clusters[non_noise_indices])) > 1:
                    silhouette_avg = silhouette_score(
                        X_for_clustering[non_noise_indices], 
                        clusters[non_noise_indices]
                    )
                    st.write(f"Silhouette Score (excluding noise points): {silhouette_avg:.4f}")
                else:
                    st.warning("Not enough non-noise points in different clusters to calculate silhouette score.")
            else:
                st.warning("Cannot calculate silhouette score with only one cluster or all noise points.")
            
            # Add cluster labels to original data
            X['Cluster'] = clusters
            
            # Display model information
            st.write(f"#### DBSCAN Clustering Results (eps={eps}, min_samples={min_samples})")
            st.write(f"Number of clusters found: {n_clusters_dbscan}")
            st.write(f"Number of noise points: {list(clusters).count(-1)}")
        
        # Display cluster distribution
        st.write("#### Cluster Distribution")
        
        # Count records in each cluster
        cluster_counts = X['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        
        # Sort by cluster number, handling negative numbers for DBSCAN noise points
        cluster_counts = cluster_counts.sort_values('Cluster')
        
        # Create cluster labels, with "Noise" for -1 cluster in DBSCAN
        cluster_counts['Cluster Label'] = cluster_counts['Cluster'].apply(
            lambda x: f"Cluster {int(x)}" if x >= 0 else "Noise Points"
        )
        
        # Plot cluster distribution
        fig = px.bar(
            cluster_counts, 
            x='Cluster Label', 
            y='Count', 
            color='Cluster Label',
            title="Distribution of Clusters",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Visualize clusters
        st.write("#### Cluster Visualization")
        
        if use_pca:
            # PCA was already applied, use first two components
            X_plot = X_for_clustering
            x_col, y_col = 0, 1  # First two PCA components
            
            # Create PCA visualization
            fig = px.scatter(
                x=X_plot[:, x_col],
                y=X_plot[:, y_col],
                color=X['Cluster'].astype(str),
                title=f"Cluster Visualization (PCA)",
                labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Add cluster centers for KMeans
            if algorithm == "K-Means":
                for i, center in enumerate(centers):
                    fig.add_trace(
                        go.Scatter(
                            x=[center[x_col]],
                            y=[center[y_col]],
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                size=15,
                                color='black',
                                line=dict(width=2)
                            ),
                            name=f"Center {i}"
                        )
                    )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create 3D visualization if 3 or more components
            if n_components >= 3:
                fig_3d = px.scatter_3d(
                    x=X_plot[:, 0],
                    y=X_plot[:, 1],
                    z=X_plot[:, 2],
                    color=X['Cluster'].astype(str),
                    title=f"3D Cluster Visualization (PCA)",
                    labels={
                        'x': 'Principal Component 1', 
                        'y': 'Principal Component 2', 
                        'z': 'Principal Component 3'
                    },
                    color_discrete_sequence=px.colors.qualitative.Bold
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
            # Use the first two numerical features for visualization
            if len(selected_num_cols) >= 2:
                x_col, y_col = selected_num_cols[0], selected_num_cols[1]
                
                fig = px.scatter(
                    X,
                    x=x_col,
                    y=y_col,
                    color=X['Cluster'].astype(str),
                    title=f"Cluster Visualization ({x_col} vs {y_col})",
                    labels={'color': 'Cluster'},
                    color_discrete_sequence=px.colors.qualitative.Bold
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
                        color=X['Cluster'].astype(str),
                        title=f"3D Cluster Visualization",
                        labels={'color': 'Cluster'},
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Not enough numerical features for 2D visualization.")
        
        # Analyze cluster characteristics
        st.write("#### Cluster Characteristics")
        
        # Include Churn column in analysis if available
        include_cols = selected_num_cols + selected_cat_cols
        if 'Churn' in df.columns and 'Churn' not in include_cols:
            include_cols.append('Churn')
        
        # Create cluster profile for numerical features
        if selected_num_cols:
            st.write("##### Numerical Features by Cluster")
            
            # Calculate statistics by cluster
            cluster_profiles = X.groupby('Cluster')[selected_num_cols].agg(['mean', 'median', 'std']).reset_index()
            
            # For each numerical feature, create a bar chart comparing means across clusters
            for feature in selected_num_cols:
                # Extract mean values for the feature by cluster
                feature_by_cluster = X.groupby('Cluster')[feature].mean().reset_index()
                feature_by_cluster['Cluster'] = feature_by_cluster['Cluster'].apply(
                    lambda x: f"Cluster {int(x)}" if x >= 0 else "Noise Points"
                )
                
                # Create bar chart
                fig = px.bar(
                    feature_by_cluster,
                    x='Cluster',
                    y=feature,
                    title=f"Average {feature} by Cluster",
                    color='Cluster',
                    color_discrete_sequence=px.colors.qualitative.Bold
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Create boxplots for numerical features by cluster
            with st.expander("Show Detailed Distribution Boxplots"):
                # For each numerical feature, create a box plot by cluster
                for feature in selected_num_cols:
                    # Create box plot
                    fig = px.box(
                        X,
                        x='Cluster',
                        y=feature,
                        title=f"Distribution of {feature} by Cluster",
                        color='Cluster',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Create cluster profile for categorical features
        if selected_cat_cols:
            st.write("##### Categorical Features by Cluster")
            
            # For each categorical feature, create a stacked bar chart showing distribution within clusters
            for feature in selected_cat_cols:
                # Create cross-tabulation
                cross_tab = pd.crosstab(
                    X['Cluster'], 
                    X[feature], 
                    normalize='index'
                ).reset_index()
                
                # Convert to long format for plotting
                cross_tab_long = pd.melt(
                    cross_tab, 
                    id_vars=['Cluster'],
                    var_name=feature,
                    value_name='Proportion'
                )
                
                # Map cluster labels
                cross_tab_long['Cluster Label'] = cross_tab_long['Cluster'].apply(
                    lambda x: f"Cluster {int(x)}" if x >= 0 else "Noise Points"
                )
                
                # Create stacked bar chart
                fig = px.bar(
                    cross_tab_long,
                    x='Cluster Label',
                    y='Proportion',
                    color=feature,
                    title=f"Distribution of {feature} by Cluster",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Analyze churn rate by cluster if available
        if 'Churn' in df.columns:
            st.write("##### Churn Analysis by Cluster")
            
            # Check if Churn is binary or categorical
            if df['Churn'].dtype == 'object':
                # Count churn in each cluster
                churn_by_cluster = pd.crosstab(
                    X['Cluster'], 
                    df['Churn'], 
                    normalize='index'
                ).reset_index()
                
                # Handle case where Churn is 'Yes'/'No'
                if 'Yes' in churn_by_cluster.columns:
                    churn_by_cluster['Churn Rate'] = churn_by_cluster['Yes'] * 100
                else:
                    # Try to find the positive class
                    positive_class = df['Churn'].value_counts().index[0]  # Assume first class is positive
                    churn_by_cluster['Churn Rate'] = churn_by_cluster[positive_class] * 100
            else:
                # Assume binary numeric churn indicator
                churn_by_cluster = X.groupby('Cluster')['Churn'].mean().reset_index()
                churn_by_cluster['Churn Rate'] = churn_by_cluster['Churn'] * 100
            
            # Add cluster labels
            churn_by_cluster['Cluster Label'] = churn_by_cluster['Cluster'].apply(
                lambda x: f"Cluster {int(x)}" if x >= 0 else "Noise Points"
            )
            
            # Create bar chart of churn rate by cluster
            fig = px.bar(
                churn_by_cluster,
                x='Cluster Label',
                y='Churn Rate',
                title="Churn Rate by Cluster",
                color='Churn Rate',
                color_continuous_scale='Reds',
                text_auto='.1f'
            )
            
            fig.update_traces(texttemplate='%{text}%', textposition='outside')
            fig.update_layout(yaxis_range=[0, 100])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Identify high-risk clusters
            high_risk_threshold = 50  # Define high risk as >50% churn rate
            high_risk_clusters = churn_by_cluster[churn_by_cluster['Churn Rate'] > high_risk_threshold]
            
            if not high_risk_clusters.empty:
                st.warning(f"High-risk clusters identified: {', '.join(high_risk_clusters['Cluster Label'].astype(str))}")
            
            # Identify low-risk clusters
            low_risk_threshold = 20  # Define low risk as <20% churn rate
            low_risk_clusters = churn_by_cluster[churn_by_cluster['Churn Rate'] < low_risk_threshold]
            
            if not low_risk_clusters.empty:
                st.success(f"Low-risk clusters identified: {', '.join(low_risk_clusters['Cluster Label'].astype(str))}")
        
        # Cluster summary and interpretation
        st.write("#### Cluster Interpretation")
        
        # Create a summary table of all clusters
        cluster_summary = pd.DataFrame({'Cluster': sorted(X['Cluster'].unique())})
        cluster_summary['Size'] = cluster_summary['Cluster'].apply(lambda x: sum(X['Cluster'] == x))
        cluster_summary['Percentage'] = 100 * cluster_summary['Size'] / len(X)
        
        # Add churn rate if available
        if 'Churn' in df.columns:
            def get_churn_rate(cluster):
                cluster_data = X[X['Cluster'] == cluster]
                if df['Churn'].dtype == 'object':
                    # Categorical churn column
                    return 100 * sum(df.loc[cluster_data.index, 'Churn'] == 'Yes') / len(cluster_data)
                else:
                    # Numeric churn column
                    return 100 * df.loc[cluster_data.index, 'Churn'].mean()
            
            cluster_summary['Churn Rate (%)'] = cluster_summary['Cluster'].apply(get_churn_rate)
        
        # Format the cluster label
        cluster_summary['Cluster Label'] = cluster_summary['Cluster'].apply(
            lambda x: f"Cluster {int(x)}" if x >= 0 else "Noise Points"
        )
        
        # Add key characteristics for each cluster
        def get_key_characteristics(cluster):
            # Get cluster data
            cluster_data = X[X['Cluster'] == cluster]
            
            characteristics = []
            
            # Check numerical features
            for col in selected_num_cols:
                # Get overall mean and standard deviation
                overall_mean = df[col].mean()
                overall_std = df[col].std()
                
                # Get cluster mean
                cluster_mean = df.loc[cluster_data.index, col].mean()
                
                # Check if significantly different from overall
                if abs(cluster_mean - overall_mean) > overall_std:
                    if cluster_mean > overall_mean:
                        characteristics.append(f"High {col}")
                    else:
                        characteristics.append(f"Low {col}")
            
            # Check categorical features
            for col in selected_cat_cols:
                # Get overall distribution
                overall_dist = df[col].value_counts(normalize=True)
                
                # Get cluster distribution
                cluster_dist = df.loc[cluster_data.index, col].value_counts(normalize=True)
                
                # Check for each category
                for category in cluster_dist.index:
                    if category in overall_dist.index:
                        # Check if proportion in cluster is significantly higher
                        if cluster_dist[category] > 1.5 * overall_dist[category]:
                            characteristics.append(f"High {col}={category}")
            
            # Return key characteristics
            return ", ".join(characteristics[:3])  # Limit to top 3
        
        cluster_summary['Key Characteristics'] = cluster_summary['Cluster'].apply(get_key_characteristics)
        
        # Display the summary table
        st.dataframe(cluster_summary[['Cluster Label', 'Size', 'Percentage', 'Churn Rate (%)' if 'Churn Rate (%)' in cluster_summary.columns else None, 'Key Characteristics']].fillna(''))
        
        # Cluster interpretation text
        st.write("##### Suggested Interpretations")
        
        # Sort clusters by size
        sorted_clusters = cluster_summary.sort_values('Size', ascending=False)
        
        for _, row in sorted_clusters.iterrows():
            cluster_label = row['Cluster Label']
            
            st.write(f"**{cluster_label}** ({row['Size']} customers, {row['Percentage']:.1f}%):")
            
            interpretation = f"This segment is characterized by {row['Key Characteristics']}. "
            
            if 'Churn Rate (%)' in row:
                churn_rate = row['Churn Rate (%)']
                
                if churn_rate > 50:
                    interpretation += f"With a high churn rate of {churn_rate:.1f}%, this is a **high-risk segment** that requires immediate attention."
                elif churn_rate > 30:
                    interpretation += f"Having a moderate churn rate of {churn_rate:.1f}%, this segment needs targeted retention strategies."
                else:
                    interpretation += f"With a low churn rate of {churn_rate:.1f}%, this segment represents loyal customers."
            
            st.write(interpretation)
        
        # Add cluster download option
        st.write("#### Download Clustered Data")
        
        # Add original DataFrame columns to the clustered data
        output_df = df.copy()
        output_df['Cluster'] = X['Cluster']
        
        # Convert to CSV for download
        csv = output_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Clustered Data as CSV",
            data=csv,
            file_name="clustered_telecom_data.csv",
            mime="text/csv"
        )
