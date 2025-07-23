import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def load_sample_data():
    """Create sample data for demonstration when actual data is not available"""
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'MSSubClass': np.random.choice([20, 30, 40, 50, 60, 70, 80, 90], n_samples),
        'MSZoning': np.random.choice(['RL', 'RM', 'C (all)', 'FV', 'RH'], n_samples),
        'LotFrontage': np.random.normal(70, 20, n_samples),
        'LotArea': np.random.normal(10000, 5000, n_samples),
        'Street': np.random.choice(['Grvl', 'Pave'], n_samples),
        'LotShape': np.random.choice(['Reg', 'IR1', 'IR2', 'IR3'], n_samples),
        'LandContour': np.random.choice(['Lvl', 'Bnk', 'HLS', 'Low'], n_samples),
        'Utilities': np.random.choice(['AllPub', 'NoSewr'], n_samples),
        'LotConfig': np.random.choice(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], n_samples),
        'LandSlope': np.random.choice(['Gtl', 'Mod', 'Sev'], n_samples),
        'Neighborhood': np.random.choice(['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst'], n_samples),
        'Condition1': np.random.choice(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe'], n_samples),
        'BldgType': np.random.choice(['1Fam', 'TwnhsE', '2fmCon', 'Duplex', 'Twnhs'], n_samples),
        'HouseStyle': np.random.choice(['2Story', '1Story', '1.5Fin', 'SLvl', 'SFoyer'], n_samples),
        'OverallQual': np.random.randint(1, 11, n_samples),
        'OverallCond': np.random.randint(1, 11, n_samples),
        'YearBuilt': np.random.randint(1900, 2010, n_samples),
        'YearRemodAdd': np.random.randint(1950, 2010, n_samples),
        'RoofStyle': np.random.choice(['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat'], n_samples),
        'RoofMatl': np.random.choice(['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran'], n_samples),
        'Exterior1st': np.random.choice(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace'], n_samples),
        'MasVnrType': np.random.choice(['None', 'BrkFace', 'Stone', 'BrkCmn'], n_samples),
        'MasVnrArea': np.random.normal(100, 150, n_samples),
        'ExterQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], n_samples),
        'ExterCond': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples),
        'Foundation': np.random.choice(['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab'], n_samples),
        'BsmtQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'None'], n_samples),
        'BsmtCond': np.random.choice(['TA', 'Gd', 'Fa', 'Po', 'None'], n_samples),
        'BsmtExposure': np.random.choice(['No', 'Gd', 'Mn', 'Av', 'None'], n_samples),
        'BsmtFinType1': np.random.choice(['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'None'], n_samples),
        'BsmtFinSF1': np.random.normal(500, 300, n_samples),
        'BsmtUnfSF': np.random.normal(500, 400, n_samples),
        'TotalBsmtSF': np.random.normal(1000, 500, n_samples),
        'Heating': np.random.choice(['GasA', 'GasW', 'Grav', 'Wall', 'OthW'], n_samples),
        'HeatingQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples),
        'CentralAir': np.random.choice(['N', 'Y'], n_samples),
        'Electrical': np.random.choice(['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix'], n_samples),
        '1stFlrSF': np.random.normal(1000, 300, n_samples),
        '2ndFlrSF': np.random.normal(500, 400, n_samples),
        'LowQualFinSF': np.random.normal(0, 50, n_samples),
        'GrLivArea': np.random.normal(1500, 500, n_samples),
        'BsmtFullBath': np.random.randint(0, 3, n_samples),
        'BsmtHalfBath': np.random.randint(0, 2, n_samples),
        'FullBath': np.random.randint(1, 4, n_samples),
        'HalfBath': np.random.randint(0, 3, n_samples),
        'BedroomAbvGr': np.random.randint(1, 6, n_samples),
        'KitchenAbvGr': np.random.randint(1, 3, n_samples),
        'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], n_samples),
        'TotRmsAbvGrd': np.random.randint(3, 12, n_samples),
        'Functional': np.random.choice(['Typ', 'Min1', 'Min2', 'Mod', 'Maj1'], n_samples),
        'Fireplaces': np.random.randint(0, 4, n_samples),
        'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], n_samples),
        'GarageType': np.random.choice(['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'None'], n_samples),
        'GarageYrBlt': np.random.randint(1900, 2010, n_samples),
        'GarageFinish': np.random.choice(['RFn', 'Unf', 'Fin', 'None'], n_samples),
        'GarageCars': np.random.randint(0, 4, n_samples),
        'GarageArea': np.random.normal(500, 200, n_samples),
        'GarageQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], n_samples),
        'GarageCond': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'], n_samples),
        'PavedDrive': np.random.choice(['Y', 'P', 'N'], n_samples),
        'WoodDeckSF': np.random.normal(100, 150, n_samples),
        'OpenPorchSF': np.random.normal(50, 100, n_samples),
        'EnclosedPorch': np.random.normal(20, 80, n_samples),
        '3SsnPorch': np.random.normal(5, 30, n_samples),
        'ScreenPorch': np.random.normal(15, 60, n_samples),
        'PoolArea': np.random.normal(5, 50, n_samples),
        'MiscVal': np.random.normal(50, 200, n_samples),
        'MoSold': np.random.randint(1, 13, n_samples),
        'YrSold': np.random.randint(2006, 2011, n_samples),
        'SaleType': np.random.choice(['WD', 'New', 'COD', 'ConLD', 'ConLI'], n_samples),
        'SaleCondition': np.random.choice(['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family'], n_samples),
    })
    
    # Ensure non-negative values for area columns
    area_columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 
                   'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                   'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
                   '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    
    for col in area_columns:
        sample_data[col] = np.maximum(0, sample_data[col])
    
    # Create realistic SalePrice based on key features
    sample_data['SalePrice'] = (
        50000 + 
        sample_data['OverallQual'] * 15000 +
        sample_data['GrLivArea'] * 50 +
        sample_data['GarageCars'] * 5000 +
        (sample_data['YearBuilt'] - 1900) * 100 +
        np.random.normal(0, 20000, n_samples)
    )
    sample_data['SalePrice'] = np.maximum(30000, sample_data['SalePrice'])
    
    return sample_data

def create_correlation_heatmap(df, figsize=(12, 10)):
    """Create correlation heatmap for numerical features"""
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    return plt

def create_feature_importance_plot(feature_importance_df):
    """Create feature importance plot using Plotly"""
    fig = px.bar(
        feature_importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Top Feature Importances',
        labels={'importance': 'Importance Score', 'feature': 'Features'}
    )
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_price_distribution_plot(df):
    """Create price distribution plot"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Sale Price Distribution', 'Log Sale Price Distribution')
    )
    
    # Original price distribution
    fig.add_trace(
        go.Histogram(x=df['SalePrice'], nbinsx=50, name='Sale Price'),
        row=1, col=1
    )
    
    # Log price distribution
    fig.add_trace(
        go.Histogram(x=np.log1p(df['SalePrice']), nbinsx=50, name='Log Sale Price'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Price Distribution Analysis",
        showlegend=False,
        height=400
    )
    
    return fig

def create_prediction_plot(y_true, y_pred):
    """Create actual vs predicted values plot"""
    # Calculate R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    fig = go.Figure()
    
    # Scatter plot of predictions vs actual
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            color='blue',
            opacity=0.6
        )
    ))
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'Actual vs Predicted Prices (R² = {r2:.3f})',
        xaxis_title='Actual Prices',
        yaxis_title='Predicted Prices',
        height=500
    )
    
    return fig

def create_residuals_plot(y_true, y_pred):
    """Create residuals plot"""
    residuals = y_pred - y_true
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(
            color='green',
            opacity=0.6
        )
    ))
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title='Residuals Plot',
        xaxis_title='Predicted Prices',
        yaxis_title='Residuals (Predicted - Actual)',
        height=400
    )
    
    return fig

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def calculate_model_metrics(y_true, y_pred):
    """Calculate and return model performance metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    
    return metrics

def create_model_comparison_plot(results_dict):
    """Create model comparison plot"""
    models = list(results_dict.keys())
    test_rmse = [results_dict[model]['test_rmse'] for model in models]
    test_r2 = [results_dict[model]['test_r2'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Test RMSE', 'Test R²'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # RMSE plot
    fig.add_trace(
        go.Bar(x=models, y=test_rmse, name='RMSE', marker_color='lightblue'),
        row=1, col=1
    )
    
    # R² plot
    fig.add_trace(
        go.Bar(x=models, y=test_r2, name='R²', marker_color='lightgreen'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def display_metrics_table(metrics_dict):
    """Display metrics in a formatted table"""
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df = metrics_df.round(4)
    return metrics_df

def create_neighborhood_price_plot(df):
    """Create neighborhood vs price plot"""
    if 'Neighborhood' in df.columns and 'SalePrice' in df.columns:
        neighborhood_prices = df.groupby('Neighborhood')['SalePrice'].agg(['mean', 'count']).reset_index()
        neighborhood_prices = neighborhood_prices[neighborhood_prices['count'] >= 5]  # Filter neighborhoods with at least 5 houses
        neighborhood_prices = neighborhood_prices.sort_values('mean', ascending=True)
        
        fig = px.bar(
            neighborhood_prices,
            x='mean',
            y='Neighborhood',
            orientation='h',
            title='Average House Prices by Neighborhood',
            labels={'mean': 'Average Sale Price', 'Neighborhood': 'Neighborhood'}
        )
        
        fig.update_layout(height=600)
        return fig
    else:
        return None

def get_prediction_explanation(prediction, feature_values, feature_importance):
    """Generate explanation for prediction"""
    explanation = f"Predicted house price: {format_currency(prediction)}\n\n"
    explanation += "Key factors influencing this prediction:\n"
    
    # Get top 5 most important features
    top_features = feature_importance.head(5)
    
    for _, row in top_features.iterrows():
        feature_name = row['feature']
        importance = row['importance']
        
        if feature_name in feature_values:
            value = feature_values[feature_name]
            explanation += f"• {feature_name}: {value} (Importance: {importance:.3f})\n"
    
    return explanation