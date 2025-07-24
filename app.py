import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add src to path for imports
# This ensures that the application can find the custom modules in the 'src' directory.
sys.path.append(str(Path(__file__).parent / "src"))

from src.model_training import HousePriceModelTrainer
from src.data_preprocessing import HousePricePreprocessor
from src.utils import (
    load_sample_data, create_correlation_heatmap, create_feature_importance_plot,
    create_price_distribution_plot, create_prediction_plot, create_residuals_plot,
    format_currency, calculate_model_metrics, create_model_comparison_plot,
    display_metrics_table, create_neighborhood_price_plot, get_prediction_explanation
)

# Page configuration for the Streamlit app
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the application
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        color: #2e8b57;
        text-align: center;
        padding: 1rem;
        background-color: #f0f8ff;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load training data with caching to improve performance."""
    train_path = "data/train.csv"
    if os.path.exists(train_path):
        return pd.read_csv(train_path)
    else:
        st.warning("Training data not found. Using sample data for demonstration.")
        return load_sample_data()

@st.cache_resource
def initialize_model():
    """Initialize model trainer with caching to avoid re-initialization on every run."""
    return HousePriceModelTrainer()

def main():
    """Main function to control the app's flow."""
    st.markdown('<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Data Exploration", "Model Training", "Price Prediction", "Model Analysis"]
    )
    
    # Initialize model trainer
    model_trainer = initialize_model()
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "Model Training":
        show_model_training(model_trainer)
    elif page == "Price Prediction":
        show_price_prediction(model_trainer)
    elif page == "Model Analysis":
        show_model_analysis(model_trainer)

def show_home_page():
    """Display the home page with an overview of the application."""
    st.markdown("""
    ## Welcome to the House Price Predictor! 🏡
    
    This application uses machine learning to predict house prices based on various features.
    
    ### How to Use:
    1. **📊 Data Exploration**: Explore the dataset and understand feature relationships.
    2. **🤖 Model Training**: Train different machine learning models and compare their performance.
    3. **💰 Price Prediction**: Input house features to get a price prediction.
    4. **📈 Model Analysis**: Analyze the trained model's performance and feature importance.
    
    Navigate using the sidebar to get started.
    """)
    
    try:
        df = load_data()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Houses", len(df))
        with col2:
            avg_price = df['SalePrice'].mean()
            st.metric("Average Price", format_currency(avg_price))
        with col3:
            median_price = df['SalePrice'].median()
            st.metric("Median Price", format_currency(median_price))
        with col4:
            st.metric("Features", len(df.columns) - 1)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_data_exploration():
    """Display the data exploration page with various visualizations."""
    st.markdown('<h2 class="sub-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    
    try:
        df = load_data()
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        st.subheader("Price Distribution")
        fig = create_price_distribution_plot(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top Features Correlated with Sale Price")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numerical_cols].corr()['SalePrice'].abs().sort_values(ascending=False)[1:16]
        fig_corr = px.bar(x=correlations.values, y=correlations.index, orientation='h')
        st.plotly_chart(fig_corr, use_container_width=True)

    except Exception as e:
        st.error(f"Error in data exploration: {str(e)}")

def show_model_training(model_trainer):
    """Display the model training page."""
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    try:
        df = load_data()
        if 'SalePrice' not in df.columns:
            st.error("SalePrice column not found!")
            return
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                X, y = model_trainer.prepare_data(df)
                results, best_model_name = model_trainer.train_all_models(X, y)
                st.success(f"✅ Training completed! Best model: {best_model_name}")
                
                st.subheader("Model Performance Comparison")
                fig = create_model_comparison_plot(results)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Detailed Metrics")
                metrics_df = display_metrics_table(results)
                st.dataframe(metrics_df)
                
                model_trainer.save_model()
                st.success("✅ Best model has been saved automatically!")
                st.session_state['model_trained'] = True

    except Exception as e:
        st.error(f"Error during training: {str(e)}")

def show_price_prediction(model_trainer):
    """Display the price prediction page with an input form."""
    st.markdown('<h2 class="sub-header">💰 House Price Prediction</h2>', unsafe_allow_html=True)
    
    # Ensure model is loaded before proceeding
    if model_trainer.best_model is None:
        try:
            model_trainer.load_model()
            st.success("✅ Model loaded successfully!")
        except FileNotFoundError:
            st.warning("⚠️ No trained model found. Please go to the 'Model Training' page to train a model.")
            return
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return
            
    with st.form("prediction_form"):
        st.write("Enter the features of the house to get a price prediction.")
        col1, col2, col3 = st.columns(3)
        
        # User input fields
        with col1:
            overall_qual = st.slider("Overall Quality", 1, 10, 7)
            year_built = st.number_input("Year Built", 1800, 2025, 2000)
            gr_liv_area = st.number_input("Living Area (sq ft)", 500, 8000, 1500)
        with col2:
            garage_cars = st.slider("Garage Car Capacity", 0, 5, 2)
            full_bath = st.slider("Full Bathrooms", 1, 5, 2)
            total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 1000)
        with col3:
            neighborhood = st.selectbox("Neighborhood", ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'])
            exter_qual = st.selectbox("Exterior Quality", ['Ex', 'Gd', 'TA', 'Fa'])
            kitchen_qual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa'])

        submitted = st.form_submit_button("🔮 Predict Price", type="primary")

        if submitted:
            try:
                # Create a dictionary of all features required by the model
                # Use default values for features not in the form
                df_train = load_data()
                features = {col: df_train[col].median() if df_train[col].dtype in ['int64', 'float64'] else df_train[col].mode()[0] for col in df_train.columns if col != 'SalePrice'}
                
                # Update with user inputs
                features.update({
                    'OverallQual': overall_qual, 'YearBuilt': year_built, 'GrLivArea': gr_liv_area,
                    'GarageCars': garage_cars, 'FullBath': full_bath, 'TotalBsmtSF': total_bsmt_sf,
                    'Neighborhood': neighborhood, 'ExterQual': exter_qual, 'KitchenQual': kitchen_qual
                })

                # Convert to DataFrame for prediction
                input_df = pd.DataFrame([features])
                
                # --- FIX: Use the correct 'predict' method ---
                predictions = model_trainer.predict(input_df)
                prediction = predictions[0] # Get the single value from the array
                
                st.markdown(f'<div class="prediction-result">Predicted Price: {format_currency(prediction)}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making prediction: {e}")

def show_model_analysis(model_trainer):
    """Display the model analysis page."""
    st.markdown('<h2 class="sub-header">📈 Model Analysis</h2>', unsafe_allow_html=True)
    
    if model_trainer.best_model is None:
        try:
            model_trainer.load_model()
        except FileNotFoundError:
            st.warning("⚠️ No trained model available. Please train a model first.")
            return

    try:
        st.subheader("Feature Importance Analysis")
        # --- FIX: Get feature importance DataFrame before passing to plot function ---
        feature_importance_df = model_trainer.get_feature_importance()
        if feature_importance_df is not None:
            fig = create_feature_importance_plot(feature_importance_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance is not available for the selected model.")

    except Exception as e:
        st.error(f"Error in model analysis: {e}")

if __name__ == "__main__":
    main()