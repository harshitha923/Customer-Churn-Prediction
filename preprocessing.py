import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io
def preprocess_data(data):
    """
    Preprocess telecom customer data
    """
    st.write("## Data Preprocessing")
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Display original data info
    st.write("### Original Data Information")
    buffer = None
    
    with st.expander("View Data Information"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write("#### Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0] if not missing_values.empty else "No missing values")
        
        st.write("#### Descriptive Statistics")
        st.write(df.describe())
    
    # Data cleaning options
    st.write("### Data Cleaning Options")
    
    # Handle missing values
    missing_strategy = st.selectbox(
        "How to handle missing values?",
        ["Drop rows with any missing values", "Fill numeric with mean", "Fill numeric with median",
         "Fill categorical with mode", "Fill with zeros/empty string"]
    )
    
    # Handle duplicates
    handle_duplicates = st.checkbox("Remove duplicate rows", value=True)
    
    # Feature selection
    with st.expander("Feature Selection"):
        all_columns = list(df.columns)
        selected_columns = st.multiselect(
            "Select columns to keep (leave empty to keep all)",
            all_columns,
            default=all_columns
        )
    
    # Scaling options for numeric features
    scaling_option = st.selectbox(
        "Scale numeric features?",
        ["No scaling", "StandardScaler (mean=0, std=1)", "MinMaxScaler (range 0-1)"]
    )
    
    # Encoding options for categorical features
    encoding_option = st.selectbox(
        "Encode categorical features?",
        ["No encoding", "One-Hot Encoding", "Label Encoding"]
    )
    
    # Apply preprocessing when button is clicked
    if st.button("Apply Preprocessing"):
        
        # Remove duplicates if selected
        if handle_duplicates:
            initial_rows = df.shape[0]
            df = df.drop_duplicates()
            st.write(f"Removed {initial_rows - df.shape[0]} duplicate rows")
        
        # Handle missing values based on selected strategy
        if missing_strategy == "Drop rows with any missing values":
            initial_rows = df.shape[0]
            df = df.dropna()
            st.write(f"Dropped {initial_rows - df.shape[0]} rows with missing values")
        else:
            # Identify numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Apply the selected strategy
            if missing_strategy == "Fill numeric with mean":
                for col in numeric_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mean())
                        st.write(f"Filled missing values in {col} with mean")
                
                for col in categorical_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mode()[0])
                        st.write(f"Filled missing values in {col} with mode")
                
            elif missing_strategy == "Fill numeric with median":
                for col in numeric_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].median())
                        st.write(f"Filled missing values in {col} with median")
                
                for col in categorical_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mode()[0])
                        st.write(f"Filled missing values in {col} with mode")
                
            elif missing_strategy == "Fill categorical with mode":
                for col in numeric_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].median())
                        st.write(f"Filled missing values in {col} with median")
                
                for col in categorical_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].mode()[0])
                        st.write(f"Filled missing values in {col} with mode")
                
            elif missing_strategy == "Fill with zeros/empty string":
                for col in numeric_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(0)
                        st.write(f"Filled missing values in {col} with 0")
                
                for col in categorical_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna("")
                        st.write(f"Filled missing values in {col} with empty string")
        
        # Feature selection
        if selected_columns and len(selected_columns) < len(all_columns):
            df = df[selected_columns]
            st.write(f"Selected {len(selected_columns)} columns: {', '.join(selected_columns)}")
        
        # Scale numeric features
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if scaling_option != "No scaling" and numeric_cols:
            if scaling_option == "StandardScaler (mean=0, std=1)":
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.write(f"Applied StandardScaler to {len(numeric_cols)} numeric columns")
            
            elif scaling_option == "MinMaxScaler (range 0-1)":
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.write(f"Applied MinMaxScaler to {len(numeric_cols)} numeric columns")
        
        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if encoding_option != "No encoding" and categorical_cols:
            if encoding_option == "One-Hot Encoding":
                df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
                st.write(f"Applied One-Hot Encoding to {len(categorical_cols)} categorical columns")
                st.write(f"Data shape changed from {df.shape} to {df_encoded.shape}")
                df = df_encoded
            
            elif encoding_option == "Label Encoding":
                for col in categorical_cols:
                    df[col] = pd.factorize(df[col])[0]
                st.write(f"Applied Label Encoding to {len(categorical_cols)} categorical columns")
        
        # Display processed data info
        st.write("### Processed Data Information")
        
        with st.expander("View Processed Data Information"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.write("#### Descriptive Statistics After Preprocessing")
            st.write(df.describe())
        
        st.write("### Processed Data Preview")
        st.dataframe(df.head())
        
        st.success("Data preprocessing completed successfully!")
        
        # Return the processed dataframe
        return df
    
    return data  # Return original data if button not clicked

def identify_data_types(df):
    """
    Identify categorical and numerical columns in the dataset
    """
    # Identify data types
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Check for binary numeric columns (often these are actually categorical)
    for col in numerical_cols.copy():
        if df[col].nunique() == 2:
            categorical_cols.append(col)
            numerical_cols.remove(col)
    
    return categorical_cols, numerical_cols

def create_preprocessing_pipeline(df, numerical_cols, categorical_cols):
    """
    Create a preprocessing pipeline for the data
    """
    # Numerical preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor
