import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import hashlib
import sqlite3
from datetime import datetime

# Import custom modules
import preprocessing
import visualization
import classification
import clustering
import outlier_detection
import association_mining

# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database setup
def init_db():
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    
    # Create users table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            email TEXT,
            full_name TEXT,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create admin user if it doesn't exist
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        hashed_password = hashlib.sha256('admin123'.encode()).hexdigest()
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ('admin', hashed_password, 'admin')
        )
    
    conn.commit()
    conn.close()

init_db()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    
    c.execute(
        "SELECT username, password, role FROM users WHERE username = ?",
        (username,)
    )
    user = c.fetchone()
    conn.close()
    
    if user and user[1] == hash_password(password):
        return {'username': user[0], 'role': user[2]}
    return None

def create_user(username, password, email, full_name, role='user'):
    conn = sqlite3.connect('user_auth.db')
    c = conn.cursor()
    
    try:
        c.execute(
            "INSERT INTO users (username, password, email, full_name, role) VALUES (?, ?, ?, ?, ?)",
            (username, hash_password(password), email, full_name, role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()

# Define functions for the main app
def load_sample_data():
    """Load the sample telecom customer data"""
    return pd.read_csv('sample_data.csv')

def upload_data():
    """Allow users to upload their own data"""
    uploaded_file = st.file_uploader("Upload your telecom customer dataset (CSV)", type=['csv'])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    return None

def search_customer(df, search_term, search_column):
    """Search for customers based on criteria"""
    if search_column in df.columns:
        if df[search_column].dtype == 'object':  # String columns
            return df[df[search_column].str.contains(search_term, case=False, na=False)]
        else:  # Numeric columns
            try:
                search_value = float(search_term)
                return df[df[search_column] == search_value]
            except ValueError:
                return pd.DataFrame()  # Return empty df if search term not convertible to float
    return pd.DataFrame()  # Return empty df if column not found

def get_download_link(df, filename="data.csv"):
    """Generate a download link for DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download data as CSV</a>'
    return href

# Login page
def login_page():
    st.title("Telecom Customer Churn Prediction System")
    st.subheader("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        user = authenticate_user(username, password)
        if user:
            st.session_state['authenticated'] = True
            st.session_state['user'] = user
            st.success(f"Welcome {user['username']}!")
            st.rerun()
        else:
            st.error("Invalid username or password")
    
    if st.button("Sign Up"):
        st.session_state['show_signup'] = True
        st.rerun()

# Signup page
def signup_page():
    st.title("Telecom Customer Churn Prediction System")
    st.subheader("Sign Up")
    
    with st.form("signup_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        email = st.text_input("Email")
        full_name = st.text_input("Full Name")
        
        submitted = st.form_submit_button("Sign Up")
        
        if submitted:
            if password != confirm_password:
                st.error("Passwords do not match!")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters long!")
            else:
                success = create_user(username, password, email, full_name)
                if success:
                    st.success("Account created successfully! Please login.")
                    st.session_state['show_signup'] = False
                    st.rerun()
                else:
                    st.error("Username already exists!")
    
    if st.button("Back to Login"):
        st.session_state['show_signup'] = False
        st.rerun()

# Main application
def main_app():
    st.title("Telecom Customer Churn Prediction System")
    
    # Sidebar with user info and logout
    st.sidebar.title("Navigation")
    st.sidebar.write(f"Logged in as: {st.session_state.user['username']} ({st.session_state.user['role']})")
    
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Show different pages based on user role
    if st.session_state.user['role'] == 'admin':
        pages = [
            "Home", 
            "Data Preprocessing", 
            "Visualization Dashboard", 
            "Churn Classification", 
            "Customer Clustering", 
            "Outlier Detection", 
            "Association Rule Mining",
            "User Management"
        ]
    else:
        pages = [
            "Home", 
            "Data Preprocessing", 
            "Visualization Dashboard", 
            "Churn Classification",
            "Customer Clustering", 
            "Outlier Detection", 
            "Association Rule Mining",
        ]
    
    page = st.sidebar.radio("Select a Page", pages)
    
    # Initialize session state for data
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    # Data loading options
    if page == "Home":
        st.write("## Welcome to the Telecom Customer Churn Prediction System")
        st.write(f"""
        Welcome back, {st.session_state.user['username']}!
        
        This application helps telecom companies analyze customer data and predict churn.
        
        Features available to you:
        - Data preprocessing and cleaning
        - Interactive visualization dashboard
        - Customer search functionality
        - Churn prediction with machine learning
        - Customer segmentation with clustering
        - Outlier detection to identify unusual patterns
        - Association rule mining for pattern discovery
        """)
        
        if st.session_state.user['role'] == 'admin':
            st.write("""
            Additional admin features:
            - User management
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Use Sample Data"):
                st.session_state.data = load_sample_data()
                st.session_state.processed_data = st.session_state.data.copy()
                st.success("Sample data loaded successfully!")
        
        with col2:
            uploaded_data = upload_data()
            if uploaded_data is not None:
                st.session_state.data = uploaded_data
                st.session_state.processed_data = uploaded_data.copy()
                st.success("Data uploaded successfully!")
        
        # Show data preview if available
        if st.session_state.data is not None:
            st.write("## Data Preview")
            st.dataframe(st.session_state.data.head())
            
            # Data summary
            st.write("## Data Summary")
            st.write(f"- Number of customers: {st.session_state.data.shape[0]}")
            st.write(f"- Number of features: {st.session_state.data.shape[1]}")
            
            if 'Churn' in st.session_state.data.columns:
                churn_rate = st.session_state.data['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
                st.write(f"- Overall churn rate: {churn_rate:.2f}%")
            
            # Customer search functionality
            st.write("## Customer Search")
            search_col, search_term_col = st.columns(2)
            
            with search_col:
                search_column = st.selectbox("Select column to search", st.session_state.data.columns)
            
            with search_term_col:
                search_term = st.text_input("Enter search term")
            
            if search_term:
                search_results = search_customer(st.session_state.data, search_term, search_column)
                if not search_results.empty:
                    st.write(f"Found {search_results.shape[0]} matching customers:")
                    st.dataframe(search_results)
                    st.markdown(get_download_link(search_results, "search_results.csv"), unsafe_allow_html=True)
                else:
                    st.info("No matching customers found.")
    
    # Data Preprocessing
    elif page == "Data Preprocessing":
        if st.session_state.data is None:
            st.warning("Please load data first from the Home page.")
        else:
            st.session_state.processed_data = preprocessing.preprocess_data(st.session_state.data)
    
    # Visualization Dashboard
    elif page == "Visualization Dashboard":
        if st.session_state.processed_data is None:
            st.warning("Please load and preprocess data first.")
        else:
            visualization.show_visualization_dashboard(st.session_state.processed_data)
    
    # Churn Classification
    elif page == "Churn Classification":
        if st.session_state.processed_data is None:
            st.warning("Please load and preprocess data first.")
        else:
            classification.build_classification_models(st.session_state.processed_data)
    
    # Customer Clustering
    elif page == "Customer Clustering":
        if st.session_state.processed_data is None:
            st.warning("Please load and preprocess data first.")
        else:
            clustering.perform_clustering(st.session_state.processed_data)
    
    # Outlier Detection
    elif page == "Outlier Detection":
        if st.session_state.processed_data is None:
            st.warning("Please load and preprocess data first.")
        else:
            outlier_detection.detect_outliers(st.session_state.processed_data)
    
    # Association Rule Mining
    elif page == "Association Rule Mining":
        if st.session_state.processed_data is None:
            st.warning("Please load and preprocess data first.")
        else:
            association_mining.mine_association_rules(st.session_state.processed_data)
    
    # User Management (Admin only)
    elif page == "User Management":
        if st.session_state.user['role'] != 'admin':
            st.warning("You don't have permission to access this page.")
        else:
            st.header("User Management")
            
            # List all users
            conn = sqlite3.connect('user_auth.db')
            users_df = pd.read_sql("SELECT username, email, role, created_at FROM users", conn)
            st.dataframe(users_df)
            
            # Add new user
            st.subheader("Add New User")
            with st.form("add_user_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                new_email = st.text_input("Email")
                new_full_name = st.text_input("Full Name")
                new_role = st.selectbox("Role", ["user", "admin"])
                
                submitted = st.form_submit_button("Add User")
                if submitted:
                    success = create_user(new_username, new_password, new_email, new_full_name, new_role)
                    if success:
                        st.success("User created successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Username already exists!")
            
            # Delete user
            st.subheader("Delete User")
            delete_username = st.selectbox("Select user to delete", users_df['username'].tolist())
            if st.button("Delete User"):
                if delete_username == st.session_state.user['username']:
                    st.error("You cannot delete your own account!")
                else:
                    c = conn.cursor()
                    c.execute("DELETE FROM users WHERE username = ?", (delete_username,))
                    conn.commit()
                    st.success(f"User {delete_username} deleted successfully!")
                    st.experimental_rerun()
            
            conn.close()

# Main function
def main():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['show_signup'] = False
    
    if not st.session_state['authenticated']:
        if st.session_state.get('show_signup', False):
            signup_page()
        else:
            login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()