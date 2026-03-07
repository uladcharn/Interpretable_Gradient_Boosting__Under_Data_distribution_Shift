import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

class BaseModel(ABC):
    """Abstract Base Class to ensure consistency across models."""
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    def evaluate(self, X_test, y_test):
        """Standard evaluation for any subpopulation or global test set."""
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        print(f"--- {self.model_name} Evaluation ---")
        print(classification_report(y_test, preds))
        return roc_auc_score(y_test, probs), preds, probs

    def performance_diagnostics(self, X_test, y_test):
        """Visual diagnostics for distribution and subpopulation analysis."""
        preds = self.model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        
        plt.figure(figsize=(10, 4))
        
        # Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {self.model_name}')
        
        # Feature Importance
        plt.subplot(1, 2, 2)
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.Series(self.model.feature_importances_, index=X_test.columns)
            importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importances')
        
        plt.tight_layout()
        plt.show()

class RFModel(BaseModel):
    def __init__(self):
        super().__init__("DecisionTree/RF")

    def train(self, X_train, y_train):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight='balanced'  # Similar to scale_pos_weight
        )
        self.model.fit(X_train, y_train)
        return self.model

class XGBModel(BaseModel):
    def __init__(self):
        super().__init__("XGBoost")

    def train(self, X_train, y_train):
        self.model = xgb.XGBClassifier(
            n_estimators=100,  # Changed from 1 to 100
            learning_rate=0.1,
            max_depth=6,
            reg_lambda=500,
            gamma=0,
            scale_pos_weight=3,
            subsample=0.9,
            colsample_bytree=0.5,
            objective='binary:logistic',
            random_state=42
        )
        self.model.fit(X_train, y_train, verbose=False)
        return self.model

class LGBMModel(BaseModel):
    def __init__(self):
        super().__init__("LightGBM")

    def train(self, X_train, y_train):
        self.model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=6, num_leaves=31, random_state=42
        )
        self.model.fit(X_train, y_train)
        return self.model

class GBTModel(BaseModel):
    def __init__(self):
        super().__init__("GradientBoosting")

    def train(self, X_train, y_train):
        # Using hyperparameters that reduce overfitting for distribution shifts
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=15,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        return self.model

class LogRegModel(BaseModel):
    def __init__(self):
        super().__init__("LogisticRegression")

    def train(self, X_train, y_train):
        # Logistic Regression benefits from 'balanced' weights in churn datasets
        # since churners are usually the minority class.
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        self.model.fit(X_train, y_train)
        return self.model