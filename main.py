# %%


# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset (replace with your actual file path)
data = pd.read_csv('Final_healthcare_big_data.csv')

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Define numerical and categorical columns
numerical_features = ['Daily_Patient_Inflow', 'Emergency_Response_Time_Minutes', 'Staff_Count', 
                      'Bed_Occupancy_Rate', 'Visit_Frequency', 'Wait_Time_Minutes', 
                      'Length_of_Stay', 'Previous_Visits', 'Resource_Utilization', 'Age']
categorical_features = ['Treatment_Outcome', 'Equipment_Availability', 'Medicine_Stock_Level', 
                        'Comorbidities', 'Disease_Category']
ordinal_features = ['Satisfaction_Rating']  # Treat as numerical/ordinal

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # Updated parameter
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())  # Scale ordinal data
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)
    ])

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)

# Check for imbalanced target
print(y.value_counts())  # Ensure 'Outcome' is balanced or apply balancing

# %%


# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('Final_healthcare_big_data.csv')

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Encode the target variable (Outcome) to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Converts 'Improved', 'Unchanged', 'Worsened' to 0, 1, 2
# Store the mapping for later (optional, for interpretation)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Encoding Mapping:", label_mapping)

# Define numerical and categorical columns
numerical_features = ['Daily_Patient_Inflow', 'Emergency_Response_Time_Minutes', 'Staff_Count', 
                      'Bed_Occupancy_Rate', 'Visit_Frequency', 'Wait_Time_Minutes', 
                      'Length_of_Stay', 'Previous_Visits', 'Resource_Utilization', 'Age']
categorical_features = ['Treatment_Outcome', 'Equipment_Availability', 'Medicine_Stock_Level', 
                        'Comorbidities', 'Disease_Category']
ordinal_features = ['Satisfaction_Rating']  # Treat as numerical/ordinal

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # Updated parameter
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())  # Scale ordinal data
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features),
        ('ord', ordinal_transformer, ordinal_features)
    ])

# Fit and transform the features
X_processed = preprocessor.fit_transform(X)

# Handle imbalanced data with SMOTE (on encoded target)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_processed, y_encoded)

# Split the balanced data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Random Forest Model with class weights
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# XGBoost Model with class weights (use numerical labels)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, scale_pos_weight='balanced')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Decode predictions back to original labels for reporting
rf_pred_decoded = label_encoder.inverse_transform(rf_pred)
xgb_pred_decoded = label_encoder.inverse_transform(xgb_pred)
y_test_decoded = label_encoder.inverse_transform(y_test)

# Evaluate both models
print("Random Forest Classification Report:")
print(classification_report(y_test_decoded, rf_pred_decoded))
print("XGBoost Classification Report:")
print(classification_report(y_test_decoded, xgb_pred_decoded))

# Confusion Matrix for Random Forest (with decoded labels)
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_decoded, rf_pred_decoded)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Improved', 'Unchanged', 'Worsened'], 
            yticklabels=['Improved', 'Unchanged', 'Worsened'])
plt.title('Confusion Matrix (Random Forest)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature Importance (Random Forest)
importances = rf_model.feature_importances_
feature_names = (numerical_features + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)) + 
                 ordinal_features)
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False).head(10))

# %%
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy={1: 74966, 2: 74966})  # Target "Unchanged" and "Worsened" to match "Improved"


# %%
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
oversample = SMOTE(random_state=42)
undersample = RandomUnderSampler(random_state=42)
pipeline = ImbPipeline(steps=[('over', oversample), ('under', undersample)])
X_balanced, y_balanced = pipeline.fit_resample(X_processed, y_encoded)

# %%
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, scale_pos_weight=sum(y_encoded==0)/sum(y_encoded==2))  # Adjust for "Worsened"

# %%
from sklearn.model_selection import GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid_rf, cv=5, scoring='f1_weighted')
grid_search_rf.fit(X_train, y_train)
print("Best RF Params:", grid_search_rf.best_params_)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}
grid_search_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, scale_pos_weight='balanced'), param_grid_xgb, cv=5, scoring='f1_weighted')
grid_search_xgb.fit(X_train, y_train)
print("Best XGBoost Params:", grid_search_xgb.best_params_)

# %%


# %%



