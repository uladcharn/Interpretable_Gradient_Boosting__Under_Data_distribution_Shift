from utils.preprocess import preliminary_preproc, separate_by_shift_categories
from utils.models import RFModel, XGBModel, LGBMModel

X_train, y_train, X_test, y_test, df, dummy_var_list = preliminary_preproc('./data/Telco_customer_churn.csv')

X_y_tenure, X_y_gender, X_y_city = separate_by_shift_categories(df, dummy_var_list)

# Distribution Analysis

# 1. Out of N categories, train your model on N-1 categories, excluding the Nth one

# 2. Then test on the Nth category

# 3. Visualize your results with SHAP Summary Plot:
## If Feature_A is positively correlated with the target in training but negatively in the test set, you have found Concept Drift.

# Subpopulation Analysis

# 1. Train your model on global data

# 2. Test your model on every provided category

# 3. Visualize your results with Heatmap (for model fails) and Residual Analysis (for discovering systematic bias)