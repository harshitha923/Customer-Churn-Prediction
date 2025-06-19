import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

def mine_association_rules(data):
    """
    Perform association rule mining on telecom customer data
    """
    st.write("## Association Rule Mining")
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Sidebar for association rule mining configuration
    st.sidebar.subheader("Association Rule Mining Configuration")
    
    # Step 1: Feature Selection and Binarization
    st.write("### Step 1: Feature Selection and Binarization")
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Include binary numeric columns
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if df[col].nunique() <= 2:
            categorical_cols.append(col)
    
    # Remove target column from features if it shouldn't be part of the rules
    exclude_target = st.checkbox("Exclude Churn from association rules", value=False)
    if exclude_target and 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    
    # Let user select features for association rule mining
    st.write("Select features to use for association rule mining:")
    
    selected_cols = st.multiselect(
        "Select features",
        categorical_cols,
        default=categorical_cols[:min(6, len(categorical_cols))]  # Default to first 6 or fewer
    )
    
    if not selected_cols:
        st.warning("Please select at least one feature for association rule mining.")
        return
    
    # Step 2: Data Preprocessing
    st.write("### Step 2: Data Preprocessing")
    
    # Determine how to binarize/encode categorical data
    binarization_method = st.radio(
        "Select method to prepare data for association mining",
        ["One-Hot Encoding", "Custom Value Selection"]
    )
    
    if binarization_method == "One-Hot Encoding":
        # Use all values of the selected columns
        st.info("One-hot encoding will create binary items for all values in the selected columns.")
    else:
        # For each column, allow selecting specific values of interest
        st.write("#### Select values of interest for each feature:")
        
        selected_values = {}
        
        for col in selected_cols:
            unique_values = df[col].unique().tolist()
            
            selected_values[col] = st.multiselect(
                f"Select values for {col}",
                unique_values,
                default=unique_values[:min(2, len(unique_values))]  # Default to first 2 or fewer
            )
    
    # Step 3: Association Rule Mining Parameters
    st.write("### Step 3: Association Rule Mining Parameters")
    
    # Set minimum support threshold
    min_support = st.slider(
        "Minimum Support",
        min_value=0.01,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Minimum support threshold for itemsets (higher values result in fewer rules)"
    )
    
    # Set minimum confidence threshold
    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence threshold for rules (higher values result in stronger rules)"
    )
    
    # Select metric to filter/sort rules
    metric = st.selectbox(
        "Metric for sorting rules",
        ["confidence", "lift", "support", "conviction"],
        index=1  # Default to lift
    )
    
    # Number of top rules to display
    max_rules = st.slider(
        "Maximum number of rules to display",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )
    
    # Execute association rule mining when button is clicked
    if st.button("Discover Association Rules"):
        
        # Prepare data for association rule mining
        with st.spinner("Preparing data and mining association rules..."):
            # Create binary DataFrame for association rule mining
            if binarization_method == "One-Hot Encoding":
                # Use get_dummies for one-hot encoding
                encoded_df = pd.get_dummies(df[selected_cols])
            else:
                # Create custom binary features based on selected values
                encoded_df = pd.DataFrame()
                
                for col in selected_cols:
                    for val in selected_values[col]:
                        # Create binary column for each selected value
                        encoded_df[f"{col}_{val}"] = (df[col] == val).astype(int)
            
            # Check if encoded DataFrame is empty
            if encoded_df.empty:
                st.error("No valid binary features were created. Please adjust your feature selection.")
                return
            
            # Run apriori algorithm to find frequent itemsets
            st.write("#### Frequent Itemsets")
            
            try:
                # Find frequent itemsets
                frequent_itemsets = apriori(
                    encoded_df,
                    min_support=min_support,
                    use_colnames=True
                )
                
                # Check if any frequent itemsets were found
                if frequent_itemsets.empty:
                    st.warning("No frequent itemsets found with the current support threshold. Try lowering the minimum support.")
                    return
                
                # Display number of frequent itemsets
                st.write(f"Found {len(frequent_itemsets)} frequent itemsets with minimum support of {min_support}.")
                
                # Generate association rules
                rules = association_rules(
                    frequent_itemsets,
                    metric=metric,
                    min_threshold=min_confidence if metric == "confidence" else 0.0
                )
                
                # Check if any rules were found
                if rules.empty:
                    st.warning("No association rules found with the current thresholds. Try lowering the minimum confidence or support.")
                    return
                
                # Filter rules by confidence if metric is not confidence
                if metric != "confidence":
                    rules = rules[rules['confidence'] >= min_confidence]
                
                # Check again if any rules remain after filtering
                if rules.empty:
                    st.warning("No association rules found with the current thresholds. Try lowering the minimum confidence or support.")
                    return
                
                # Sort rules by selected metric (descending)
                rules = rules.sort_values(metric, ascending=False)
                
                # Limit to top rules
                top_rules = rules.head(max_rules)
                
                # Format the rules for better display
                formatted_rules = format_rules(top_rules)
                
                # Display the rules
                st.write(f"#### Top {len(formatted_rules)} Association Rules")
                st.dataframe(formatted_rules)
                
                # Visualize the rules
                st.write("#### Rule Visualization")
                
                # Scatter plot of rules (support vs confidence)
                scatter_fig = px.scatter(
                    top_rules,
                    x='support',
                    y='confidence',
                    size='lift',
                    color='lift',
                    hover_name=top_rules.index,
                    title="Association Rules (Support vs Confidence)",
                    labels={
                        'support': 'Support',
                        'confidence': 'Confidence',
                        'lift': 'Lift'
                    },
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # Create network visualization of rules
                st.write("#### Network Visualization of Rules")
                create_network_visualization(top_rules)
                
                # Create heatmap for item support
                st.write("#### Support Heatmap for Top Items")
                create_item_support_heatmap(frequent_itemsets, encoded_df.columns, min_support)
                
                # Business Insights
                st.write("#### Business Insights")
                
                # Analyze rules containing Churn
                if 'Churn' in df.columns and not exclude_target:
                    analyze_churn_rules(top_rules, metric)
                
                # Find interesting patterns
                find_interesting_patterns(top_rules, metric)
                
                # Recommendations
                st.write("#### Recommendations")
                
                st.markdown("""
                Based on the discovered association rules, consider the following actions:
                
                1. **Bundle Services**: Create service bundles for items that frequently appear together, offering discounts for the package.
                
                2. **Cross-Selling Opportunities**: Use the rules to identify opportunities for cross-selling services to existing customers.
                
                3. **Churn Prevention**: Identify combinations of services or features that are associated with lower churn rates and promote these to at-risk customers.
                
                4. **Customer Profiling**: Use the association patterns to better understand different customer segments and their preferences.
                
                5. **Marketing Campaigns**: Design targeted marketing campaigns based on the discovered associations.
                """)
                
                # Generate rule-based recommendations for churn prevention
                if 'Churn' in df.columns and not exclude_target:
                    st.write("#### Churn Prevention Recommendations")
                    generate_churn_recommendations(top_rules)
                
                # Add download option
                csv = formatted_rules.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Association Rules as CSV",
                    data=csv,
                    file_name="telecom_association_rules.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"An error occurred during association rule mining: {str(e)}")
                st.info("This might be due to insufficient data or inappropriate parameter settings. Try adjusting the parameters or selecting different features.")

def format_rules(rules_df):
    """
    Format the association rules for better display
    """
    # Create a copy to avoid modifying the original DataFrame
    formatted_df = rules_df.copy()
    
    # Convert frozensets to strings for better display
    formatted_df['antecedents_str'] = formatted_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    formatted_df['consequents_str'] = formatted_df['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Create a readable rule representation
    formatted_df['rule'] = formatted_df.apply(lambda row: f"{row['antecedents_str']} → {row['consequents_str']}", axis=1)
    
    # Round metrics for cleaner display
    for col in ['support', 'confidence', 'lift', 'conviction']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].round(3)
    
    # Select and reorder columns for display
    display_cols = ['rule', 'support', 'confidence', 'lift']
    
    if 'conviction' in formatted_df.columns:
        display_cols.append('conviction')
    
    return formatted_df[display_cols]

def create_network_visualization(rules_df):
    """
    Create a network visualization of association rules
    """
    # Create a network graph
    G = nx.DiGraph()
    
    # Add edges from antecedents to consequents
    for idx, row in rules_df.iterrows():
        # Get antecedents and consequents
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                # Add nodes and edge
                G.add_node(antecedent)
                G.add_node(consequent)
                
                # Use lift as edge weight and label
                G.add_edge(
                    antecedent, 
                    consequent, 
                    weight=row['lift'],
                    label=f"lift: {row['lift']:.2f}"
                )
    
    # Check if the graph has any edges
    if len(G.edges) == 0:
        st.info("No connections found for network visualization.")
        return
    
    # Create positions for nodes
    try:
        pos = nx.spring_layout(G, seed=42)
    except:
        pos = {node: (np.random.random(), np.random.random()) for node in G.nodes}
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Add edge label as hover text
        edge_text.append(f"{edge[0]} → {edge[1]}<br>Lift: {edge[2]['weight']:.2f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=15,
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left'
            ),
            line_width=2
        )
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Association Rule Network',
                    #titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))
    
    st.plotly_chart(fig, use_container_width=True)

def create_item_support_heatmap(frequent_itemsets, all_items, min_support):
    """
    Create a heatmap showing support for individual items
    """
    # Extract all individual items from frequent itemsets
    all_frequent_items = set()
    for itemset in frequent_itemsets['itemsets']:
        all_frequent_items.update(itemset)
    
    # Calculate support for each individual item
    item_support = {}
    for item in all_frequent_items:
        # Find the support for the individual item
        for idx, row in frequent_itemsets.iterrows():
            if row['itemsets'] == frozenset([item]):
                item_support[item] = row['support']
                break
    
    # Sort items by support
    sorted_items = sorted(item_support.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 20 items for cleaner visualization
    top_items = sorted_items[:20]
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame({
        'Item': [item for item, _ in top_items],
        'Support': [support for _, support in top_items]
    })
    
    # Create heatmap
    fig = px.bar(
        heatmap_df,
        x='Support',
        y='Item',
        orientation='h',
        title='Support for Top Items',
        color='Support',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    
    st.plotly_chart(fig, use_container_width=True)

def analyze_churn_rules(rules_df, metric):
    """
    Analyze rules containing churn indicators
    """
    # Look for rules with churn in consequents
    churn_rules = []
    no_churn_rules = []
    
    for idx, row in rules_df.iterrows():
        for consequent in row['consequents']:
            if 'Churn_Yes' in consequent or consequent == 'Churn_Yes' or 'Churn_1' in consequent or consequent == 'Churn_1':
                churn_rules.append((idx, row))
            elif 'Churn_No' in consequent or consequent == 'Churn_No' or 'Churn_0' in consequent or consequent == 'Churn_0':
                no_churn_rules.append((idx, row))
    
    # Display churn-related rules
    if churn_rules:
        st.write("##### Factors Associated with Churn")
        
        # Extract the rules
        churn_rules_df = pd.DataFrame([row for _, row in churn_rules])
        
        # Format and display
        formatted_churn_rules = format_rules(churn_rules_df)
        st.dataframe(formatted_churn_rules)
        
        # Extract key factors
        key_factors = []
        for _, row in churn_rules:
            key_factors.extend(list(row['antecedents']))
        
        # Count occurrences of each factor
        from collections import Counter
        factor_counts = Counter(key_factors)
        
        # Display top factors
        if factor_counts:
            top_factors = pd.DataFrame({
                'Factor': list(factor_counts.keys()),
                'Occurrence Count': list(factor_counts.values())
            }).sort_values('Occurrence Count', ascending=False)
            
            st.write("Top Factors Associated with Churn:")
            
            fig = px.bar(
                top_factors.head(10),
                x='Occurrence Count',
                y='Factor',
                orientation='h',
                title='Top Factors Associated with Churn',
                color='Occurrence Count',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
    
    if no_churn_rules:
        st.write("##### Factors Associated with Retention")
        
        # Extract the rules
        no_churn_rules_df = pd.DataFrame([row for _, row in no_churn_rules])
        
        # Format and display
        formatted_no_churn_rules = format_rules(no_churn_rules_df)
        st.dataframe(formatted_no_churn_rules)
        
        # Extract key factors
        key_factors = []
        for _, row in no_churn_rules:
            key_factors.extend(list(row['antecedents']))
        
        # Count occurrences of each factor
        from collections import Counter
        factor_counts = Counter(key_factors)
        
        # Display top factors
        if factor_counts:
            top_factors = pd.DataFrame({
                'Factor': list(factor_counts.keys()),
                'Occurrence Count': list(factor_counts.values())
            }).sort_values('Occurrence Count', ascending=False)
            
            st.write("Top Factors Associated with Retention:")
            
            fig = px.bar(
                top_factors.head(10),
                x='Occurrence Count',
                y='Factor',
                orientation='h',
                title='Top Factors Associated with Retention',
                color='Occurrence Count',
                color_continuous_scale='Greens'
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
    
    # If no churn rules found
    if not churn_rules and not no_churn_rules:
        st.info("No specific rules involving churn were found. Try adjusting the parameters or include 'Churn' in the selected features.")

def find_interesting_patterns(rules_df, metric):
    """
    Find and highlight interesting patterns in the rules
    """
    # Look for high lift/confidence rules
    if metric == 'lift':
        threshold = 3.0  # High lift threshold
        high_metric_rules = rules_df[rules_df['lift'] > threshold]
    else:
        threshold = 0.8  # High confidence threshold
        high_metric_rules = rules_df[rules_df['confidence'] > threshold]
    
    if not high_metric_rules.empty:
        st.write(f"##### High {metric.capitalize()} Rules")
        
        # Format and display
        formatted_high_rules = format_rules(high_metric_rules)
        st.dataframe(formatted_high_rules)
        
        # Analyze frequent antecedents
        antecedents = []
        for idx, row in high_metric_rules.iterrows():
            antecedents.extend(list(row['antecedents']))
        
        # Count occurrences
        from collections import Counter
        antecedent_counts = Counter(antecedents)
        
        # Display top antecedents
        if antecedent_counts:
            top_antecedents = pd.DataFrame({
                'Antecedent': list(antecedent_counts.keys()),
                'Occurrence Count': list(antecedent_counts.values())
            }).sort_values('Occurrence Count', ascending=False)
            
            st.write(f"Most Common Antecedents in High {metric.capitalize()} Rules:")
            
            fig = px.bar(
                top_antecedents.head(10),
                x='Occurrence Count',
                y='Antecedent',
                orientation='h',
                title=f'Most Common Antecedents in High {metric.capitalize()} Rules',
                color='Occurrence Count',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No rules with exceptionally high {metric} were found. Try adjusting the parameters.")

def generate_churn_recommendations(rules_df):
    """
    Generate recommendations for churn prevention based on association rules
    """
    # Find rules where churn is in the consequent
    churn_rules = []
    no_churn_rules = []
    
    for idx, row in rules_df.iterrows():
        for consequent in row['consequents']:
            if 'Churn_Yes' in consequent or consequent == 'Churn_Yes' or 'Churn_1' in consequent or consequent == 'Churn_1':
                churn_rules.append((idx, row))
            elif 'Churn_No' in consequent or consequent == 'Churn_No' or 'Churn_0' in consequent or consequent == 'Churn_0':
                no_churn_rules.append((idx, row))
    
    # Generate recommendations
    recommendations = []
    
    # Based on high churn rules
    for idx, row in churn_rules:
        antecedents = list(row['antecedents'])
        
        if len(antecedents) >= 1:
            antecedent_str = ', '.join(antecedents)
            
            # Create recommendation
            recommendation = f"Customers with {antecedent_str} have a higher risk of churn. "
            recommendation += f"The confidence is {row['confidence']:.2f} and lift is {row['lift']:.2f}."
            
            recommendations.append({
                'type': 'risk',
                'recommendation': recommendation,
                'confidence': row['confidence'],
                'lift': row['lift']
            })
    
    # Based on low churn (retention) rules
    for idx, row in no_churn_rules:
        antecedents = list(row['antecedents'])
        
        if len(antecedents) >= 1:
            antecedent_str = ', '.join(antecedents)
            
            # Create recommendation
            recommendation = f"Customers with {antecedent_str} have a higher chance of retention. "
            recommendation += f"The confidence is {row['confidence']:.2f} and lift is {row['lift']:.2f}."
            
            recommendations.append({
                'type': 'retention',
                'recommendation': recommendation,
                'confidence': row['confidence'],
                'lift': row['lift']
            })
    
    # Display recommendations
    if recommendations:
        # Sort by lift
        recommendations_df = pd.DataFrame(recommendations).sort_values('lift', ascending=False)
        
        # Display risk factors
        risk_recommendations = recommendations_df[recommendations_df['type'] == 'risk']
        if not risk_recommendations.empty:
            st.write("##### Risk Factors for Churn")
            
            for _, row in risk_recommendations.iterrows():
                st.markdown(f"- {row['recommendation']}")
        
        # Display retention factors
        retention_recommendations = recommendations_df[recommendations_df['type'] == 'retention']
        if not retention_recommendations.empty:
            st.write("##### Retention Strategies")
            
            for _, row in retention_recommendations.iterrows():
                st.markdown(f"- {row['recommendation']}")
        
        # Overall recommendations
        st.write("##### Overall Recommendations")
        
        st.markdown("""
        Based on the association rules analysis, we recommend:
        
        1. **Target High-Risk Combinations**: Proactively reach out to customers with feature combinations associated with high churn.
        
        2. **Promote Retention-Associated Services**: Encourage adoption of services and features associated with customer retention.
        
        3. **Bundle Complementary Services**: Create bundles of services that frequently appear together in retention-associated rules.
        
        4. **Address Pain Points**: Investigate and improve services that frequently appear in churn-associated rules.
        
        5. **Personalized Retention Offers**: Design personalized retention offers based on customer features and their association with churn.
        """)
    else:
        st.info("No specific churn-related recommendations could be generated. Try including 'Churn' in your feature selection and adjusting the parameters.")
