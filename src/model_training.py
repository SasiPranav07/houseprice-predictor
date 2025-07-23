import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from .data_preprocessing import HousePricePreprocessor
import warnings
warnings.filterwarnings('ignore')

class HousePriceModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.preprocessor = HousePricePreprocessor()
        self.model_scores = {}
        
    def load_data(self, train_path):
        """Load training data"""
        return pd.read_csv(train_path)
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Separate features and target
        if 'SalePrice' in df.columns:
            X = df.drop('SalePrice', axis=1)
            y = np.log1p(df['SalePrice'])  # Log transform target for better performance
        else:
            raise ValueError("SalePrice column not found in training data")
        
        # Preprocess features
        X_processed = self.preprocessor.fit_transform(X)
        
        return X_processed, y
    
    def initialize_models(self):
        """Initialize different regression models"""
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.001, max_iter=10000)
        }
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """Evaluate a single model"""
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_rmse': cv_rmse,
            'cv_std': np.sqrt(cv_scores.std())
        }
    
    def train_all_models(self, X, y, test_size=0.2):
        """Train and evaluate all models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate each model
        results = {}
        trained_models = {}
        
        print("Training models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                scores = self.evaluate_model(model, X_train, X_test, y_train, y_test)
                results[name] = scores
                trained_models[name] = model
                print(f"{name} - Test RMSE: {scores['test_rmse']:.4f}, Test R²: {scores['test_r2']:.4f}")
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Find best model based on test RMSE
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_rmse'])
        self.best_model = trained_models[best_model_name]
        self.model_scores = results
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best test RMSE: {results[best_model_name]['test_rmse']:.4f}")
        
        return results, best_model_name
    
    def hyperparameter_tuning(self, X, y, model_name='XGBoost'):
        """Perform hyperparameter tuning for the selected model"""
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate best model
        best_scores = self.evaluate_model(best_model, X_train, X_test, y_train, y_test)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best test RMSE: {best_scores['test_rmse']:.4f}")
        
        self.best_model = best_model
        return best_model, grid_search.best_params_, best_scores
    
    def save_model(self, filepath='models/house_price_model.pkl'):
        """Save the trained model and preprocessor"""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and preprocessor together
        model_package = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'feature_names': self.preprocessor.get_feature_names()
        }
        
        joblib.dump(model_package, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/house_price_model.pkl'):
        """Load a saved model and preprocessor"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_package = joblib.load(filepath)
        self.best_model = model_package['model']
        self.preprocessor = model_package['preprocessor']
        
        print(f"Model loaded from {filepath}")
        return self.best_model
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("No model has been trained or loaded")
        
        # Preprocess the data
        X_processed = self.preprocessor.transform(X)
        
        # Make predictions (remember to reverse log transform)
        predictions = self.best_model.predict(X_processed)
        return np.expm1(predictions)  # Reverse log1p transform
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from the best model"""
        if self.best_model is None:
            raise ValueError("No model has been trained or loaded")
        
        if hasattr(self.best_model, 'feature_importances_'):
            feature_names = self.preprocessor.get_feature_names()
            importances = self.best_model.feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df.head(top_n)
        else:
            print("Feature importance not available for this model type")
            return None