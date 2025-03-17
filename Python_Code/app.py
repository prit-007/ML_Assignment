from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import time

app = Flask(__name__)
CORS(app)  # Allow Flutter to access this API


def convert_to_native_types(obj):
    """Convert NumPy and Pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj
    
    
def detect_problem_type(y):
    """Detect whether the target column is for classification or regression."""
    unique_values = y.nunique()
    
    # Handle categorical target variables
    if y.dtype == 'object':
        return 'classification'
    
    # If small number of unique numeric values, likely classification
    if unique_values <= 10:
        return 'classification'
    
    # If large number of unique values, likely regression
    if unique_values > 20:
        return 'regression'
    
    # For cases in between, check if values are mostly whole numbers
    if pd.api.types.is_numeric_dtype(y):
        # If most values are integers, likely classification
        if (y.dropna() % 1 == 0).mean() > 0.9:
            return 'classification'
    
    # Default to regression for other cases
    return 'regression'

def get_data_insights(data):
    """Get detailed insights about the dataset."""
    # Basic stats
    rows, cols = data.shape
    missing_values = data.isnull().sum().sum()
    missing_percentage = (missing_values / (rows * cols)) * 100
    
    # Feature types
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    numerical_features = data.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate correlation for numerical features
    correlations = {}
    if len(numerical_features) > 1:
        corr_matrix = data[numerical_features].corr().abs()
        # Get top 5 highly correlated pairs
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > 0.7:  # Only strong correlations
                    corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])  # Convert to float
                    })
        # Sort by correlation value and take top 5
        corr_pairs = sorted(corr_pairs, key=lambda x: x['correlation'], reverse=True)[:5]
        correlations['high_correlation_pairs'] = corr_pairs
    
    # Identify the target column (last column)
    target_column = data.columns[-1]
    
    # Calculate class distribution for target variable
    class_distribution = None
    if target_column in categorical_features or data[target_column].nunique() < 10:
        class_distribution = data[target_column].value_counts().to_dict()
    
    # Summary statistics for numerical features
    numerical_stats = {}
    for col in numerical_features:
        numerical_stats[col] = {
            'min': float(data[col].min()),  # Convert to float
            'max': float(data[col].max()),  # Convert to float
            'mean': float(data[col].mean()),  # Convert to float
            'median': float(data[col].median()),  # Convert to float
            'std': float(data[col].std())  # Convert to float
        }
    
    insights = {
        "rows": int(rows),  # Convert to int
        "columns": int(cols),  # Convert to int
        "missing_values": int(missing_values),  # Convert to int
        "missing_percentage": float(missing_percentage),  # Convert to float
        "categorical_features": int(len(categorical_features)),  # Convert to int
        "numerical_features": int(len(numerical_features)),  # Convert to int
        "target_column": target_column,
        "class_distribution": class_distribution,
        "high_correlations": correlations.get('high_correlation_pairs', []),
        "numerical_stats": numerical_stats
    }
    
    # Add suggested preprocessing steps
    preprocessing_suggestions = []
    if missing_percentage > 0:
        preprocessing_suggestions.append("Handle missing values (imputation or removal)")
    if len(categorical_features) > 0:
        preprocessing_suggestions.append("Encode categorical features")
    if any(data[col].max() - data[col].min() > 10 for col in numerical_features):
        preprocessing_suggestions.append("Scale numerical features")
    
    insights["preprocessing_suggestions"] = preprocessing_suggestions
    
    return insights
def preprocess_data(data):
    """Preprocess the data for modeling."""
    # Separate features and target
    X = data.iloc[:, :-1]  # Features (all columns except last)
    y = data.iloc[:, -1]   # Target (last column)
    
    # Handle categorical features in X
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    
    # Create a simple pipeline for preprocessing
    # Handle missing values
    if X[numerical_cols].isnull().sum().sum() > 0:
        num_imputer = SimpleImputer(strategy='mean')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    if X[categorical_cols].isnull().sum().sum() > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Encode categorical features
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    # Scale numerical features
    X[numerical_cols] = StandardScaler().fit_transform(X[numerical_cols])
    
    # Handle target variable if it's categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
    
    return X, y

def train_best_model(data, max_training_time=30):
    """Train multiple models and find the best one."""
    start_time = time.time()
    
    # Preprocess data
    X, y = preprocess_data(data)
    problem_type = detect_problem_type(data.iloc[:, -1])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_details = {}
    
    if problem_type == 'classification':
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }
        
        # Define metrics to evaluate
        metrics = {
            'accuracy': accuracy_score,
            'f1_score': lambda y_true, y_pred: float(f1_score(y_true, y_pred, average='weighted')),  # Convert to float
            'precision': lambda y_true, y_pred: float(precision_score(y_true, y_pred, average='weighted')),  # Convert to float
            'recall': lambda y_true, y_pred: float(recall_score(y_true, y_pred, average='weighted'))  # Convert to float
        }
        
        primary_metric = 'accuracy'
        
    else:  # regression
        models = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=42),
            "Lasso Regression": Lasso(random_state=42),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }
        
        # Define metrics to evaluate
        metrics = {
            'rmse': lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),  # Convert to float
            'r2_score': lambda y_true, y_pred: float(r2_score(y_true, y_pred))  # Convert to float
        }
        
        primary_metric = 'rmse'

    
    results = {}
    model_explanations = {
        "Random Forest": "Ensemble learning method using multiple decision trees. Good for complex relationships and handles non-linearity well.",
        "Gradient Boosting": "Advanced ensemble technique that builds trees sequentially to correct errors. Often achieves high accuracy.",
        "Logistic Regression": "Simple, interpretable model for classification tasks. Works well with linearly separable data.",
        "Linear Regression": "Basic regression model for linear relationships. Easy to interpret and implement.",
        "Ridge Regression": "Regularized version of linear regression that prevents overfitting by penalizing large coefficients.",
        "Lasso Regression": "Regularized regression that can perform feature selection by forcing some coefficients to zero.",
        "SVM": "Powerful classifier that works by finding the optimal hyperplane to separate classes. Effective in high-dimensional spaces.",
        "SVR": "Regression version of SVM. Good for capturing non-linear relationships.",
        "KNN": "Instance-based learning that predicts based on the k-nearest data points. Simple but effective.",
        "Decision Tree": "Tree-based model that makes decisions based on feature values. Easy to interpret and visualize."
    }
    
    # Train models while respecting time limit
    for name, model in models.items():
        # Check if time limit is exceeded
        if time.time() - start_time > max_training_time:
            results[name] = {"trained": False, "reason": "Time limit exceeded"}
            continue
        
        try:
            # Fit the model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            model_metrics = {}
            for metric_name, metric_func in metrics.items():
                model_metrics[metric_name] = float(metric_func(y_test, y_pred))
            
            # Calculate feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                # Get feature names
                feature_names = data.columns[:-1].tolist()
                importance = model.feature_importances_
                # Create a dictionary of feature importance
                feature_importance = dict(zip(feature_names, importance.tolist()))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
            
            results[name] = {
                "trained": True,
                "metrics": model_metrics,
                "feature_importance": feature_importance,
                "explanation": model_explanations.get(name, "")
            }
            
        except Exception as e:
            results[name] = {"trained": False, "reason": str(e)}
    
    # Find the best model based on the primary metric
    trained_models = {name: model_info for name, model_info in results.items() if model_info.get("trained", False)}
    
    if trained_models:
        if problem_type == 'classification':
            # For classification, higher metric is better
            best_model = max(
                trained_models.items(),
                key=lambda x: x[1]["metrics"][primary_metric]
            )[0]
        else:
            # For regression (RMSE), lower is better
            best_model = min(
                trained_models.items(),
                key=lambda x: x[1]["metrics"][primary_metric]
            )[0]
        
        score = results[best_model]["metrics"][primary_metric]
        
        # Calculate training time
        training_time = time.time() - start_time
        
        return {
            "problem_type": problem_type,
            "best_model": best_model,
            "score": score * 100 if problem_type == 'classification' else score,
            "all_results": results,
            "training_time": round(training_time, 2),
            "metrics_explanation": {
                "accuracy": "Percentage of correctly classified instances",
                "f1_score": "Harmonic mean of precision and recall",
                "rmse": "Root Mean Squared Error (lower is better)",
                "r2_score": "Coefficient of determination (higher is better)"
            }
        }
    else:
        return {
            "problem_type": problem_type,
            "best_model": "None",
            "score": 0,
            "error": "No models were successfully trained",
            "training_time": round(time.time() - start_time, 2)
        }
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    try:
        data = pd.read_csv(file)
        
        # Check if dataset has at least 2 columns (1 feature + 1 target)
        if data.shape[1] < 2:
            return jsonify({
                "error": "The CSV file must contain at least one feature column and one target column",
                "suggestion": "Ensure your CSV file has at least two columns. The last column will be treated as the target."
            }), 400
        
        # Check if dataset is too large
        if data.shape[0] > 10000 or data.shape[1] > 100:
            # Take a sample if too large
            data = data.sample(min(10000, data.shape[0]), random_state=42)
        
        # Get insights about the data
        insights = get_data_insights(data)
        
        # Train models and get recommendations
        result = train_best_model(data)
        
        # Add additional information for better UX
        result["tips"] = [
            "Consider feature engineering to improve model performance",
            "Always validate your model with cross-validation",
            "Explore hyperparameter tuning for the best model",
            "Check for class imbalance if accuracy seems suspiciously high"
        ]
        
        # Convert all NumPy and Pandas types to native Python types
        response_data = {
            "insights": convert_to_native_types(insights),
            "model_recommendation": convert_to_native_types(result)
        }
        
        return jsonify(response_data)
    
    except pd.errors.EmptyDataError:
        return jsonify({
            "error": "The CSV file is empty",
            "suggestion": "Ensure your CSV file contains data"
        }), 400
    except pd.errors.ParserError:
        return jsonify({
            "error": "The CSV file is not properly formatted",
            "suggestion": "Ensure your CSV file is correctly formatted and uses commas as delimiters"
        }), 400
    except Exception as e:
        return jsonify({
            "error": str(e),
            "suggestion": "Make sure your CSV file is properly formatted and contains at least one feature column and one target column"
        }), 500
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Server is running"}), 200

if __name__ == "__main__":
    app.run(debug=True)