import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_visualization_dashboard(data):
    """
    Create an interactive visualization dashboard for telecom customer data
    """
    st.write("## Data Visualization Dashboard")
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Check if 'Churn' column exists for churn-specific visualizations
    has_churn = 'Churn' in df.columns
    
    # Sidebar visualization options
    st.sidebar.subheader("Visualization Options")
    viz_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["Customer Demographics", "Service Usage Analysis", "Churn Analysis", 
         "Correlation Analysis", "Feature Distribution", "Custom Visualization"]
    )
    
    # Customer Demographics Visualizations
    if viz_type == "Customer Demographics":
        st.write("### Customer Demographics Analysis")
        
        demo_type = st.selectbox(
            "Select Demographic Analysis",
            ["Gender Distribution", "Senior Citizen Status", "Partner Status", 
             "Dependents", "Contract Type", "Tenure Distribution"]
        )
        
        if demo_type == "Gender Distribution":
            if 'gender' in df.columns.str.lower():
                gender_col = df.columns[df.columns.str.lower() == 'gender'][0]
                
                # Create pie chart for gender distribution
                fig = px.pie(df, names=gender_col, title="Customer Gender Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)
                
                # If Churn column exists, show gender vs churn
                if has_churn:
                    st.write("#### Gender vs Churn Rate")
                    gender_churn = df.groupby([gender_col, 'Churn']).size().unstack().fillna(0)
                    gender_churn_pct = gender_churn.div(gender_churn.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(gender_churn_pct, y='Yes', 
                                labels={'index': 'Gender', 'y': 'Churn Rate (%)'},
                                title="Churn Rate by Gender",
                                color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Gender column not found in the dataset.")
        
        elif demo_type == "Senior Citizen Status":
            if 'seniorcitizen' in df.columns.str.lower():
                senior_col = df.columns[df.columns.str.lower() == 'seniorcitizen'][0]
                
                # Create pie chart for senior citizen distribution
                labels = {0: 'Non-Senior', 1: 'Senior'}
                df['SeniorCitizenLabel'] = df[senior_col].map(labels)
                
                fig = px.pie(df, names='SeniorCitizenLabel', title="Senior Citizen Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
                
                # If Churn column exists, show senior citizen vs churn
                if has_churn:
                    st.write("#### Senior Citizen Status vs Churn Rate")
                    senior_churn = df.groupby([senior_col, 'Churn']).size().unstack().fillna(0)
                    senior_churn_pct = senior_churn.div(senior_churn.sum(axis=1), axis=0) * 100
                    
                    # Create a mapping dataframe
                    mapping_df = pd.DataFrame({
                        'SeniorCitizen': [0, 1],
                        'SeniorCitizenLabel': ['Non-Senior', 'Senior'],
                        'ChurnRate': senior_churn_pct['Yes'].values
                    })
                    
                    fig = px.bar(mapping_df, x='SeniorCitizenLabel', y='ChurnRate',
                                labels={'x': 'Senior Citizen Status', 'y': 'Churn Rate (%)'},
                                title="Churn Rate by Senior Citizen Status",
                                color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("SeniorCitizen column not found in the dataset.")
        
        elif demo_type == "Partner Status":
            if 'partner' in df.columns.str.lower():
                partner_col = df.columns[df.columns.str.lower() == 'partner'][0]
                
                # Create pie chart for partner status
                fig = px.pie(df, names=partner_col, title="Partner Status Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig, use_container_width=True)
                
                # If Churn column exists, show partner status vs churn
                if has_churn:
                    st.write("#### Partner Status vs Churn Rate")
                    partner_churn = df.groupby([partner_col, 'Churn']).size().unstack().fillna(0)
                    partner_churn_pct = partner_churn.div(partner_churn.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(partner_churn_pct, y='Yes',
                                labels={'index': 'Partner Status', 'y': 'Churn Rate (%)'},
                                title="Churn Rate by Partner Status",
                                color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Partner column not found in the dataset.")
        
        elif demo_type == "Dependents":
            if 'dependents' in df.columns.str.lower():
                dep_col = df.columns[df.columns.str.lower() == 'dependents'][0]
                
                # Create pie chart for dependents
                fig = px.pie(df, names=dep_col, title="Dependents Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel1)
                st.plotly_chart(fig, use_container_width=True)
                
                # If Churn column exists, show dependents vs churn
                if has_churn:
                    st.write("#### Dependents Status vs Churn Rate")
                    dep_churn = df.groupby([dep_col, 'Churn']).size().unstack().fillna(0)
                    dep_churn_pct = dep_churn.div(dep_churn.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(dep_churn_pct, y='Yes',
                                labels={'index': 'Has Dependents', 'y': 'Churn Rate (%)'},
                                title="Churn Rate by Dependents Status",
                                color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Dependents column not found in the dataset.")
        
        elif demo_type == "Contract Type":
            if 'contract' in df.columns.str.lower():
                contract_col = df.columns[df.columns.str.lower() == 'contract'][0]
                
                # Create bar chart for contract types
                contract_counts = df[contract_col].value_counts().reset_index()
                contract_counts.columns = ['Contract', 'Count']
                
                fig = px.bar(contract_counts, x='Contract', y='Count',
                            title="Distribution of Contract Types",
                            color='Contract',
                            color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig, use_container_width=True)
                
                # If Churn column exists, show contract type vs churn
                if has_churn:
                    st.write("#### Contract Type vs Churn Rate")
                    contract_churn = df.groupby([contract_col, 'Churn']).size().unstack().fillna(0)
                    contract_churn_pct = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(contract_churn_pct, y='Yes',
                                labels={'index': 'Contract Type', 'y': 'Churn Rate (%)'},
                                title="Churn Rate by Contract Type",
                                color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Contract column not found in the dataset.")
        
        elif demo_type == "Tenure Distribution":
            if 'tenure' in df.columns.str.lower():
                tenure_col = df.columns[df.columns.str.lower() == 'tenure'][0]
                
                # Create histogram for tenure
                fig = px.histogram(df, x=tenure_col, 
                                   title="Customer Tenure Distribution",
                                   labels={tenure_col: 'Tenure (months)'},
                                   color_discrete_sequence=['#4ECDC4'])
                st.plotly_chart(fig, use_container_width=True)
                
                # If Churn column exists, show tenure vs churn
                if has_churn:
                    st.write("#### Tenure vs Churn Rate")
                    # Create tenure bins
                    df['TenureBin'] = pd.cut(df[tenure_col], 
                                            bins=[0, 12, 24, 36, 48, 60, 72],
                                            labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72'])
                    
                    tenure_bin_churn = df.groupby(['TenureBin', 'Churn']).size().unstack().fillna(0)
                    tenure_bin_churn_pct = tenure_bin_churn.div(tenure_bin_churn.sum(axis=1), axis=0) * 100
                    
                    fig = px.line(x=tenure_bin_churn_pct.index, y=tenure_bin_churn_pct['Yes'], 
                                markers=True,
                                labels={'x': 'Tenure (months)', 'y': 'Churn Rate (%)'},
                                title="Churn Rate by Tenure",
                                color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Tenure column not found in the dataset.")
    
    # Service Usage Analysis
    elif viz_type == "Service Usage Analysis":
        st.write("### Service Usage Analysis")
        
        service_type = st.selectbox(
            "Select Service Analysis",
            ["Internet Service", "Phone Service", "Multiple Lines", 
             "Online Security", "Online Backup", "Device Protection", 
             "Tech Support", "Streaming TV", "Streaming Movies"]
        )
        
        # Get the column name that matches the service type (case insensitive)
        service_cols = [col for col in df.columns if col.lower().replace(" ", "") == service_type.lower().replace(" ", "")]
        
        if service_cols:
            service_col = service_cols[0]
            
            # Create pie chart for service usage
            fig = px.pie(df, names=service_col, title=f"{service_type} Usage Distribution",
                        color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)
            
            # If Churn column exists, show service usage vs churn
            if has_churn:
                st.write(f"#### {service_type} vs Churn Rate")
                service_churn = df.groupby([service_col, 'Churn']).size().unstack().fillna(0)
                service_churn_pct = service_churn.div(service_churn.sum(axis=1), axis=0) * 100
                
                fig = px.bar(service_churn_pct, y='Yes',
                            labels={'index': service_type, 'y': 'Churn Rate (%)'},
                            title=f"Churn Rate by {service_type}",
                            color_discrete_sequence=['#FF6B6B'])
                fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"{service_type} column not found in the dataset.")
    
    # Churn Analysis
    elif viz_type == "Churn Analysis":
        if has_churn:
            st.write("### Churn Analysis")
            
            churn_type = st.selectbox(
                "Select Churn Analysis Type",
                ["Overall Churn Rate", "Churn by Monthly Charges", "Churn by Total Charges",
                 "Churn by Payment Method", "Churn by Paperless Billing", "Churn Factors"]
            )
            
            if churn_type == "Overall Churn Rate":
                # Create pie chart for overall churn rate
                fig = px.pie(df, names='Churn', title="Overall Churn Rate",
                            color_discrete_sequence=['#4ECDC4', '#FF6B6B'])
                st.plotly_chart(fig, use_container_width=True)
            
            elif churn_type == "Churn by Monthly Charges":
                if 'monthlycharges' in df.columns.str.lower():
                    monthly_col = df.columns[df.columns.str.lower() == 'monthlycharges'][0]
                    
                    # Create histogram by churn
                    fig = px.histogram(df, x=monthly_col, color='Churn',
                                      marginal="box", 
                                      title="Distribution of Monthly Charges by Churn Status",
                                      labels={monthly_col: 'Monthly Charges'},
                                      color_discrete_sequence=['#4ECDC4', '#FF6B6B'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create monthly charges bins
                    df['MonthlyChargesBin'] = pd.qcut(df[monthly_col], 5)
                    
                    monthly_bin_churn = df.groupby(['MonthlyChargesBin', 'Churn']).size().unstack().fillna(0)
                    monthly_bin_churn_pct = monthly_bin_churn.div(monthly_bin_churn.sum(axis=1), axis=0) * 100
                    
                    # Convert index to string for better plotting
                    monthly_bin_churn_pct = monthly_bin_churn_pct.reset_index()
                    monthly_bin_churn_pct['MonthlyChargesBin'] = monthly_bin_churn_pct['MonthlyChargesBin'].astype(str)
                    
                    fig = px.line(monthly_bin_churn_pct, x='MonthlyChargesBin', y='Yes',
                                 markers=True,
                                 labels={'MonthlyChargesBin': 'Monthly Charges Range', 'Yes': 'Churn Rate (%)'},
                                 title="Churn Rate by Monthly Charges",
                                 color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("MonthlyCharges column not found in the dataset.")
            
            elif churn_type == "Churn by Total Charges":
                if 'totalcharges' in df.columns.str.lower():
                    total_col = df.columns[df.columns.str.lower() == 'totalcharges'][0]
                    
                    # Create histogram by churn
                    fig = px.histogram(df, x=total_col, color='Churn',
                                      marginal="box", 
                                      title="Distribution of Total Charges by Churn Status",
                                      labels={total_col: 'Total Charges'},
                                      color_discrete_sequence=['#4ECDC4', '#FF6B6B'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create total charges bins
                    df['TotalChargesBin'] = pd.qcut(
                        pd.to_numeric(df[total_col], errors='coerce'), 
                        5, 
                        duplicates='drop'
                    )
                    
                    total_bin_churn = df.groupby(['TotalChargesBin', 'Churn']).size().unstack().fillna(0)
                    total_bin_churn_pct = total_bin_churn.div(total_bin_churn.sum(axis=1), axis=0) * 100
                    
                    # Convert index to string for better plotting
                    total_bin_churn_pct = total_bin_churn_pct.reset_index()
                    total_bin_churn_pct['TotalChargesBin'] = total_bin_churn_pct['TotalChargesBin'].astype(str)
                    
                    fig = px.line(total_bin_churn_pct, x='TotalChargesBin', y='Yes',
                                 markers=True,
                                 labels={'TotalChargesBin': 'Total Charges Range', 'Yes': 'Churn Rate (%)'},
                                 title="Churn Rate by Total Charges",
                                 color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("TotalCharges column not found in the dataset.")
            
            elif churn_type == "Churn by Payment Method":
                if 'paymentmethod' in df.columns.str.lower():
                    payment_col = df.columns[df.columns.str.lower() == 'paymentmethod'][0]
                    
                    # Calculate churn rate by payment method
                    payment_churn = df.groupby([payment_col, 'Churn']).size().unstack().fillna(0)
                    payment_churn_pct = payment_churn.div(payment_churn.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(payment_churn_pct, y='Yes',
                                labels={'index': 'Payment Method', 'y': 'Churn Rate (%)'},
                                title="Churn Rate by Payment Method",
                                color='Yes',
                                color_continuous_scale=px.colors.sequential.Reds)
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show payment method distribution
                    payment_counts = df[payment_col].value_counts().reset_index()
                    payment_counts.columns = ['Payment Method', 'Count']
                    
                    fig = px.pie(payment_counts, values='Count', names='Payment Method',
                                title="Distribution of Payment Methods",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("PaymentMethod column not found in the dataset.")
            
            elif churn_type == "Churn by Paperless Billing":
                if 'paperlessbilling' in df.columns.str.lower():
                    billing_col = df.columns[df.columns.str.lower() == 'paperlessbilling'][0]
                    
                    # Calculate churn rate by paperless billing
                    billing_churn = df.groupby([billing_col, 'Churn']).size().unstack().fillna(0)
                    billing_churn_pct = billing_churn.div(billing_churn.sum(axis=1), axis=0) * 100
                    
                    fig = px.bar(billing_churn_pct, y='Yes',
                                labels={'index': 'Paperless Billing', 'y': 'Churn Rate (%)'},
                                title="Churn Rate by Paperless Billing",
                                color_discrete_sequence=['#FF6B6B'])
                    fig.update_layout(yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show paperless billing distribution
                    fig = px.pie(df, names=billing_col, title="Paperless Billing Distribution",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("PaperlessBilling column not found in the dataset.")
            
            elif churn_type == "Churn Factors":
                # Identify categorical columns for feature importance visualization
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                categorical_cols = [col for col in categorical_cols if col != 'Churn']
                
                # Calculate and display churn rate for each feature
                st.write("#### Churn Rate by Different Factors")
                
                # Create a figure with subplots
                fig = make_subplots(rows=len(categorical_cols)//2 + len(categorical_cols)%2, 
                                    cols=2, 
                                    subplot_titles=[f"Churn Rate by {col}" for col in categorical_cols])
                
                # Add bar charts for each categorical feature
                for i, col in enumerate(categorical_cols):
                    row = i//2 + 1
                    col_pos = i%2 + 1
                    
                    # Calculate churn rate
                    feature_churn = df.groupby([col, 'Churn']).size().unstack().fillna(0)
                    feature_churn_pct = feature_churn.div(feature_churn.sum(axis=1), axis=0) * 100
                    
                    # Add bar chart to subplot
                    fig.add_trace(
                        go.Bar(x=feature_churn_pct.index, y=feature_churn_pct['Yes'], name=col),
                        row=row, col=col_pos
                    )
                
                # Update layout
                fig.update_layout(height=300*len(categorical_cols)//2, showlegend=False)
                fig.update_yaxes(range=[0, 100], title_text="Churn Rate (%)")
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Churn column not found in the dataset. Cannot perform churn analysis.")
    
    # Correlation Analysis
    elif viz_type == "Correlation Analysis":
        st.write("### Correlation Analysis")
        
        # Identify numerical columns for correlation analysis
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numerical_cols) > 1:
            # Compute correlation matrix
            correlation = df[numerical_cols].corr()
            
            # Create heatmap
            fig = px.imshow(correlation, 
                           text_auto=True, 
                           color_continuous_scale='RdBu_r',
                           title="Correlation Matrix of Numerical Features")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show specific correlations
            st.write("#### Feature Correlation with Selected Variable")
            
            target_var = st.selectbox("Select Target Variable", numerical_cols)
            
            # Sort correlations
            sorted_corr = correlation[target_var].sort_values(ascending=False)
            
            # Create bar chart of correlations
            fig = px.bar(x=sorted_corr.index, y=sorted_corr.values,
                        labels={'x': 'Features', 'y': f'Correlation with {target_var}'},
                        title=f"Correlation of Features with {target_var}",
                        color=sorted_corr.values,
                        color_continuous_scale='RdBu_r')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numerical columns for correlation analysis.")
    
    # Feature Distribution
    elif viz_type == "Feature Distribution":
        st.write("### Feature Distribution Analysis")
        
        # Get all columns
        all_cols = df.columns.tolist()
        
        # Select feature
        feature = st.selectbox("Select Feature", all_cols)
        
        # Check data type of the feature
        if df[feature].dtype in ['int64', 'float64']:
            # Numerical feature
            st.write(f"#### Distribution of {feature} (Numerical)")
            
            # Create histogram
            fig = px.histogram(df, x=feature,
                              marginal="box", 
                              title=f"Distribution of {feature}",
                              color_discrete_sequence=['#4ECDC4'])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show descriptive statistics
            st.write("#### Descriptive Statistics")
            stats = df[feature].describe()
            stats_df = pd.DataFrame(stats).transpose()
            st.dataframe(stats_df)
            
            # If Churn column exists, show distribution by churn
            if has_churn:
                st.write(f"#### Distribution of {feature} by Churn Status")
                
                fig = px.histogram(df, x=feature, color='Churn',
                                  marginal="box", 
                                  title=f"Distribution of {feature} by Churn Status",
                                  color_discrete_sequence=['#4ECDC4', '#FF6B6B'])
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Categorical feature
            st.write(f"#### Distribution of {feature} (Categorical)")
            
            # Create bar chart
            value_counts = df[feature].value_counts().reset_index()
            value_counts.columns = [feature, 'Count']
            
            fig = px.bar(value_counts, x=feature, y='Count',
                        title=f"Distribution of {feature}",
                        color=feature,
                        color_discrete_sequence=px.colors.qualitative.Bold)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show value counts
            st.write("#### Value Counts")
            st.dataframe(value_counts)
            
            # If Churn column exists, show distribution by churn
            if has_churn:
                st.write(f"#### Distribution of {feature} by Churn Status")
                
                # Calculate counts by churn
                churn_counts = df.groupby([feature, 'Churn']).size().unstack().fillna(0)
                
                # Calculate percentages
                churn_pct = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
                
                # Create grouped bar chart
                fig = px.bar(churn_counts, 
                            barmode='group',
                            title=f"Distribution of {feature} by Churn Status",
                            labels={'value': 'Count', 'variable': 'Churn Status'},
                            color_discrete_sequence=['#4ECDC4', '#FF6B6B'])
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create stacked percentage bar chart
                fig = px.bar(churn_pct, 
                            barmode='stack',
                            title=f"Percentage Distribution of {feature} by Churn Status",
                            labels={'value': 'Percentage (%)', 'variable': 'Churn Status'},
                            color_discrete_sequence=['#4ECDC4', '#FF6B6B'])
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Custom Visualization
    elif viz_type == "Custom Visualization":
        st.write("### Custom Visualization")
        
        # Get all columns
        all_cols = df.columns.tolist()
        
        # Get numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Select plot type
        plot_type = st.selectbox(
            "Select Plot Type",
            ["Scatter Plot", "Bar Chart", "Line Chart", "Box Plot", "Violin Plot", "Heatmap"]
        )
        
        if plot_type == "Scatter Plot":
            # Select x and y axes
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-Axis", numerical_cols, index=0)
            
            with col2:
                y_axis = st.selectbox("Select Y-Axis", numerical_cols, index=min(1, len(numerical_cols)-1))
            
            # Optional: select color
            color_var = st.selectbox("Color by (optional)", ["None"] + all_cols)
            color_var = None if color_var == "None" else color_var
            
            # Create scatter plot
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_var,
                           title=f"{x_axis} vs {y_axis}",
                           labels={x_axis: x_axis, y_axis: y_axis},
                           color_discrete_sequence=px.colors.qualitative.Bold if color_var else ['#4ECDC4'])
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Bar Chart":
            # Select x and y axes
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-Axis (Category)", categorical_cols, index=0)
            
            with col2:
                agg_options = ["Count", "Sum", "Mean", "Median", "Min", "Max"]
                agg_func = st.selectbox("Select Aggregation", agg_options, index=0)
            
            if agg_func == "Count":
                # Count of records
                agg_data = df[x_axis].value_counts().reset_index()
                agg_data.columns = [x_axis, 'Count']
                
                fig = px.bar(agg_data, x=x_axis, y='Count',
                            title=f"Count of Records by {x_axis}",
                            color=x_axis,
                            color_discrete_sequence=px.colors.qualitative.Bold)
            else:
                # Select y-axis for aggregation
                y_axis = st.selectbox("Select Y-Axis (Value to aggregate)", numerical_cols, index=0)
                
                # Map aggregation function
                agg_map = {
                    "Sum": "sum",
                    "Mean": "mean",
                    "Median": "median",
                    "Min": "min",
                    "Max": "max"
                }
                
                # Perform aggregation
                agg_data = df.groupby(x_axis)[y_axis].agg(agg_map[agg_func]).reset_index()
                
                fig = px.bar(agg_data, x=x_axis, y=y_axis,
                            title=f"{agg_func} of {y_axis} by {x_axis}",
                            color=x_axis,
                            color_discrete_sequence=px.colors.qualitative.Bold)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Line Chart":
            # Select x and y axes
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-Axis", all_cols, index=0)
            
            with col2:
                y_axis = st.selectbox("Select Y-Axis", numerical_cols, index=0)
            
            # Optional: group by
            group_by = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
            group_by = None if group_by == "None" else group_by
            
            # If x_axis is numerical, bin it
            if df[x_axis].dtype in ['int64', 'float64']:
                # Create bins
                num_bins = st.slider("Number of bins", min_value=5, max_value=50, value=10)
                df[f'{x_axis}_bin'] = pd.cut(df[x_axis], num_bins)
                x_axis = f'{x_axis}_bin'
            
            # Calculate aggregation
            if group_by:
                agg_data = df.groupby([x_axis, group_by])[y_axis].mean().reset_index()
                
                fig = px.line(agg_data, x=x_axis, y=y_axis, color=group_by,
                             title=f"{y_axis} by {x_axis} grouped by {group_by}",
                             labels={x_axis: x_axis, y_axis: y_axis},
                             markers=True,
                             color_discrete_sequence=px.colors.qualitative.Bold)
            else:
                agg_data = df.groupby(x_axis)[y_axis].mean().reset_index()
                
                fig = px.line(agg_data, x=x_axis, y=y_axis,
                             title=f"{y_axis} by {x_axis}",
                             labels={x_axis: x_axis, y_axis: y_axis},
                             markers=True,
                             color_discrete_sequence=['#4ECDC4'])
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            # Select x and y axes
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-Axis (Category)", categorical_cols, index=0)
            
            with col2:
                y_axis = st.selectbox("Select Y-Axis (Numerical)", numerical_cols, index=0)
            
            # Optional: color
            color_var = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
            color_var = None if color_var == "None" else color_var
            
            # Create box plot
            fig = px.box(df, x=x_axis, y=y_axis, color=color_var,
                        title=f"Box Plot of {y_axis} by {x_axis}",
                        labels={x_axis: x_axis, y_axis: y_axis},
                        color_discrete_sequence=px.colors.qualitative.Bold if color_var else ['#4ECDC4'])
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Violin Plot":
            # Select x and y axes
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-Axis (Category)", categorical_cols, index=0)
            
            with col2:
                y_axis = st.selectbox("Select Y-Axis (Numerical)", numerical_cols, index=0)
            
            # Create violin plot
            fig = px.violin(df, x=x_axis, y=y_axis, box=True,
                           title=f"Violin Plot of {y_axis} by {x_axis}",
                           labels={x_axis: x_axis, y_axis: y_axis},
                           color=x_axis,
                           color_discrete_sequence=px.colors.qualitative.Bold)
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Heatmap":
            # Select categorical columns for cross-tabulation
            col1, col2 = st.columns(2)
            
            with col1:
                row_var = st.selectbox("Select Row Variable", categorical_cols, index=0)
            
            with col2:
                col_var = st.selectbox("Select Column Variable", categorical_cols, index=min(1, len(categorical_cols)-1))
            
            # Select aggregation
            agg_options = ["Count", "Sum", "Mean", "Median"]
            agg_func = st.selectbox("Select Aggregation", agg_options, index=0)
            
            if agg_func == "Count":
                # Create cross-tabulation of counts
                cross_tab = pd.crosstab(df[row_var], df[col_var])
                
                title = f"Count of Records by {row_var} and {col_var}"
            else:
                # Select value to aggregate
                value_var = st.selectbox("Select Value to Aggregate", numerical_cols, index=0)
                
                # Map aggregation function
                agg_map = {
                    "Sum": np.sum,
                    "Mean": np.mean,
                    "Median": np.median
                }
                
                # Create cross-tabulation with aggregation
                cross_tab = pd.crosstab(
                    df[row_var], 
                    df[col_var], 
                    values=df[value_var], 
                    aggfunc=agg_map[agg_func]
                )
                
                title = f"{agg_func} of {value_var} by {row_var} and {col_var}"
            
            # Create heatmap
            fig = px.imshow(cross_tab,
                           title=title,
                           labels=dict(x=col_var, y=row_var, color=agg_func),
                           text_auto=True,
                           aspect="auto",
                           color_continuous_scale='Viridis')
            
            st.plotly_chart(fig, use_container_width=True)
            
    # Additional info and help
    with st.sidebar.expander("Visualization Help"):
        st.write("""
        ### How to use this dashboard:
        
        1. **Select a visualization type** from the dropdown in the sidebar.
        2. **Customize the plots** using the options provided.
        3. **Interact with the plots** by hovering, zooming, and panning.
        4. **Analyze patterns** in the data, especially relating to customer churn.
        
        ### Tips:
        - Look for patterns in churn rates across different customer segments.
        - Examine how services and contract types impact churn.
        - Check correlations between numerical variables.
        - Identify customer segments with high retention for targeted marketing.
        """)
