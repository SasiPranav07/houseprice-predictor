import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class HousePricePreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.feature_columns = None
        
    def clean_data(self, df):
        """Clean and prepare the dataset"""
        df = df.copy()
        
        # Handle missing values for specific columns
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
        
        # Fill categorical missing values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('None')
            
        # Fill numerical missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'SalePrice':  # Don't fill target variable
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def feature_engineering(self, df):
        """Create new features"""
        df = df.copy()
        
        # Total area features
        df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
        df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
        
        # Age features
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        
        # Quality features
        df['OverallScore'] = df['OverallQual'] * df['OverallCond']
        
        # Area per room
        df['AreaPerRoom'] = df['TotalSF'] / (df['TotRmsAbvGrd'] + 1)  # +1 to avoid division by zero
        
        # Garage features
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        df = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                unique_values = set(df[col].unique())
                known_values = set(self.label_encoders[col].classes_)
                unseen_values = unique_values - known_values
                
                if unseen_values:
                    # Map unseen values to most frequent class
                    most_frequent = df[col].mode()[0] if not df[col].mode().empty else 'None'
                    df[col] = df[col].replace(list(unseen_values), most_frequent)
                
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def select_features(self, df):
        """Select important features"""
        # Important features based on domain knowledge and correlation analysis
        important_features = [
            'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF',
            '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd',
            'MasVnrArea', 'Fireplaces', 'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF',
            'OpenPorchSF', '2ndFlrSF', 'HalfBath', 'LotArea', 'BsmtFullBath',
            'BsmtUnfSF', 'BedroomAbvGr', 'ScreenPorch', 'PoolArea', 'MiscVal',
            'LowQualFinSF', 'YrSold', 'OverallCond', 'MSSubClass', 'EnclosedPorch',
            'KitchenAbvGr', '3SsnPorch', 'BsmtHalfBath', 'MSZoning', 'LotShape',
            'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
            'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
            'Exterior1st', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'Heating',
            'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PavedDrive', 'SaleType', 'SaleCondition',
            # Engineered features
            'TotalSF', 'TotalBathrooms', 'TotalPorchSF', 'HouseAge', 'RemodAge',
            'OverallScore', 'AreaPerRoom', 'HasGarage', 'HasPool', 'Has2ndFloor', 'HasBasement'
        ]
        
        # Only keep columns that exist in the dataframe
        available_features = [col for col in important_features if col in df.columns]
        return df[available_features]
    
    def fit_transform(self, df):
        """Fit preprocessor and transform data"""
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df)
        df = self.select_features(df)
        
        # Store feature columns
        self.feature_columns = df.columns.tolist()
        
        # Scale numerical features
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df),
            columns=df.columns,
            index=df.index
        )
        
        return df_scaled
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical_features(df)
        df = self.select_features(df)
        
        # Ensure same columns as training data
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            df = df[self.feature_columns]
        
        # Scale features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        
        return df_scaled
    
    def get_feature_names(self):
        """Get list of feature names"""
        return self.feature_columns if self.feature_columns else []