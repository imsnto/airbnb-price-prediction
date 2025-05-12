# Airbnb Price Prediction 

## Overview
This project develops a machine learning model to predict Airbnb listing prices based on various features such as ratings, reviews, bedrooms, bathrooms, beds, and guest capacity. The process includes exploratory data analysis (EDA), data preprocessing, feature engineering, and model training using an Artificial Neural Network (ANN) and ensemble methods (Random Forest and Gradient Boosting).

## Dataset
- **Source**: The dataset is loaded from `airbnb.csv`.
- **Features**: Includes numerical features (e.g., `price`, `rating`, `reviews`, `bedrooms`, `bathrooms`, `beds`, `guests`) and categorical features (e.g., `country`, `host_name`).
- **Size**: Original shape varies; after preprocessing, outliers and missing values are handled, reducing the dataset size.

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Data Loading**: Loaded using `pandas` from `airbnb.csv`.
- **Data Inspection**: 
  - Displayed random samples, dataset shape, column names, and data types.
  - Checked for missing values and data distributions.
- **Visualizations**:
  - Histograms and boxplots for numerical features to analyze distributions and skewness.
  - Count plots for categorical features (`country`, `host_name`, `bedrooms`) to identify top categories.
  - Correlation heatmap to assess relationships between numerical features.
- **Key Findings**:
  - Features like `studios` and `toiles` were deemed unnecessary and excluded.
  - Outliers were detected and removed using the Interquartile Range (IQR) method.
  - Categorical features showed significant variation (e.g., top 20 countries and host names).

### 2. Data Preprocessing
- **Missing Values**:
  - Dropped rows with missing `host_name` (only 7 instances).
  - Filled missing `checkin` and `checkout` values with 'unknown'.
- **Type Conversion**:
  - Converted `rating` from object to float, replacing 'New' with 0.
  - Converted `reviews` to integer after removing special characters.
- **Outlier Removal**:
  - Applied IQR-based filtering on `price`, `rating`, `reviews`, `bedrooms`, `bathrooms`, `beds`, and `guests`.
- **Feature Encoding**:
  - Used `LabelEncoder` to encode `country` into `country_encode`.
- **Feature Scaling**:
  - Standardized `reviews` using `StandardScaler`.
- **Feature Selection**:
  - Dropped irrelevant columns: `Unnamed: 0`, `id`, `name`, `host_name`, `host_id`, `address`, `features`, `amenities`, `safety_rules`, `hourse_rules`, `img_links`, `checkin`, `checkout`, `country`, `price`, `studios`, `toiles`, `country_encode`.
  - Target variable: `price`.
  - Features: Remaining numerical columns.

### 3. Data Splitting
- **Train-Validation-Test Split**:
  - Training: 70% of the data.
  - Validation: 15% of the data.
  - Test: 15% of the data.
  - Used `train_test_split` with `random_state=42` for reproducibility.

### 4. Model Development
Three models were implemented to predict Airbnb prices:

#### a. Artificial Neural Network (ANN)
- **Architecture**:
  - Sequential model with 4 layers: 
    - Input layer: 128 neurons, ReLU activation.
    - Hidden layers: 64 and 32 neurons, ReLU activation, with 30% dropout.
    - Output layer: 1 neuron, linear activation.
- **Training**:
  - Optimizer: Adam.
  - Loss: Huber.
  - Metrics: Mean Absolute Error (MAE).
  - Batch size: 32.
  - Epochs: 100 (with early stopping on validation loss, patience=10).
- **Evaluation**:
  - Plotted training and validation loss curves.
  - Evaluated test loss and MAE.
  - Compared predicted vs. actual prices for the first 10 test samples.

#### b. Random Forest Regressor
- **Configuration**:
  - 100 estimators, `random_state=42`.
- **Training**: Fit on training data.
- **Evaluation**: Compared predicted vs. actual prices for the first 10 test samples.

#### c. Gradient Boosting Regressor
- **Configuration**:
  - 100 estimators, learning rate=0.1, `random_state=42`.
- **Training**: Fit on training data.
- **Evaluation**: Compared predicted vs. actual prices for the first 10 test samples.

## Results
- **ANN**: Provided a baseline with Huber loss and MAE metrics, showing reasonable convergence (visualized via loss curves).
- **Random Forest**: Demonstrated predictive capability with differences between actual and predicted prices analyzed.
- **Gradient Boosting**: Similar analysis performed, though a minor error in the code (referencing `y_pred[10]`) needs correction to `y_test[:10]`.
- **Limitations**:
  - The ANN may require hyperparameter tuning for better performance.
  - Ensemble models (Random Forest, Gradient Boosting) were not fully evaluated with metrics like MAE or RMSE.
  - Categorical features like `country_encode` were included but not extensively explored in the models.
