import subprocess
import sys
import importlib

# List of required packages
required_packages = [
    'joblib', 'gc', 'matplotlib', 'numpy', 'pandas', 'requests', 'seaborn', 'statsmodels', 'tqdm',
    'feature_engine', 'io', 'mlxtend', 'scipy', 'sklearn', 'zipfile'
]

# Function to check if a package is installed
def install_missing_packages(packages):
    for package in packages:
        try:
            # Try importing the package
            importlib.import_module(package)
        except ImportError:
            # If the package is not installed, install it using pip
            print(f"Package {package} is not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Run the function to check and install missing packages
install_missing_packages(required_packages)


import joblib, gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
import warnings
from tqdm import tqdm
from zipfile import ZipFile
from io import BytesIO
from mlxtend.plotting import plot_decision_regions
from scipy import stats
from sklearn import feature_selection
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from feature_engine.encoding import RareLabelEncoder
from feature_engine.outliers import OutlierTrimmer
from feature_engine.selection import DropCorrelatedFeatures


# Custom Transformer for Box-Cox Transformation
class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lambdas_ = None  # Store lambdas for each feature

    def fit(self, X, y=None):
        # Fit Box-Cox for each feature individually
        self.lambdas_ = [boxcox(X[:, i] + 1e-6)[1] for i in range(X.shape[1])]  # Add small value to avoid zeroes
        return self

    def transform(self, X, y=None):
        # Apply Box-Cox transformation using fitted lambdas
        return np.array([boxcox(X[:, i] + 1e-6, lmbda=self.lambdas_[i]) for i in range(X.shape[1])]).T

class IsolationForestOutlierTrimmer:
    def __init__(self, contamination=0.01, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.clf = IsolationForest(contamination=self.contamination, random_state=self.random_state)

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self

    def transform(self, X, y=None):
        y_pred = self.clf.predict(X)
        mask = y_pred != -1
        if y is not None:
            return X[mask], y[mask]
        return X[mask]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X, y)
    
class IQROutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound_ = None
        self.upper_bound_ = None
        self.outlier_share_ = None  # Store the share of outliers removed

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        self.lower_bound_ = Q1 - self.factor * IQR
        self.upper_bound_ = Q3 + self.factor * IQR
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        initial_row_count = len(X)
        
        is_within_bounds = (X >= self.lower_bound_) & (X <= self.upper_bound_)
        X_filtered = X[is_within_bounds.all(axis=1)]
        
        final_row_count = len(X_filtered)
        self.outlier_share_ = (initial_row_count - final_row_count) / initial_row_count
        
        if y is not None:
            y = pd.Series(y) if isinstance(y, np.ndarray) else y
            y_filtered = y.loc[X_filtered.index]
            return X_filtered, y_filtered
        
        return X_filtered
    
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def get_outlier_indices(self, X):
        X = pd.DataFrame(X)
        is_within_bounds = (X >= self.lower_bound_) & (X <= self.upper_bound_)
        outliers_mask = ~is_within_bounds.all(axis=1)
        return X[outliers_mask].index

    def get_bounds(self):
        bounds = pd.DataFrame({
            'Feature': self.lower_bound_.index,
            'Lower Bound': self.lower_bound_.values,
            'Upper Bound': self.upper_bound_.values
        })
        bounds.set_index('Feature', inplace=True)
        return bounds

    def print_outlier_share(self):
        if self.outlier_share_ is not None:
            print(f"Total share of outliers removed: {self.outlier_share_ * 100:.2f}%")
        else:
            print("Outlier share not calculated. Ensure `transform` method has been called.")

class StatmodelsWrapper(BaseEstimator):
    def __init__(self, model_type='ols', use_wls=False):
        self.model_type = model_type
        self.use_wls = use_wls
        self.model = None

    def fit(self, X, y):
        X = sm.add_constant(X)

        if self.model_type == 'ols':
            if self.use_wls:
                # Fit an OLS model to estimate residuals for WLS
                ols_model = sm.OLS(y, X).fit()
                residuals = ols_model.resid

                # Compute weights as the inverse of the squared residuals
                weights = 1 / (residuals ** 2 + np.finfo(float).eps)  # Adding epsilon to avoid division by zero

                # Fit the WLS model with the computed weights
                self.model = sm.WLS(y, X, weights=weights).fit()
            else:
                # Fit the OLS model
                self.model = sm.OLS(y, X).fit()

        elif self.model_type == 'logit':
            # Fit the logistic regression model
            self.model = sm.Logit(y, X).fit()
        
        self.summary = self.model.summary  # Store the summary as an attribute
        return self

    def predict(self, X):
        X = sm.add_constant(X)
        if self.model_type in ['ols', 'wls']:
            return self.model.predict(X)
        elif self.model_type == 'logit':
            # For logistic regression, return class labels (0 or 1) based on probability threshold of 0.5
            probabilities = self.model.predict(X)
            return (probabilities >= 0.5).astype(int)

    def predict_proba(self, X):
        if self.model_type == 'logit':
            X = sm.add_constant(X)
            probabilities = self.model.predict(X)
            # Return probabilities for class 0 and class 1
            return np.vstack((1 - probabilities, probabilities)).T
        else:
            raise NotImplementedError("Probability predictions are only available for logistic regression.")

class RemoveHighCorr(BaseEstimator, TransformerMixin):
    def __init__(self, method='spearman', thsld=0.8, p_val_tshld=0.05):
        self.method = method
        self.thsld = thsld
        self.p_val_tshld = p_val_tshld
        self.to_remove_ = []

    def fit(self, X, y=None):
        # Convert array to DataFrame if necessary and manage columns dynamically
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.columns = X.columns  # Set columns from current DataFrame
        
        cols = X.columns
        corr_counts = {col: 0 for col in cols}
        mean_corrs = {col: [] for col in cols}
        pairs = []

        # Compute correlations
        for i in range(len(cols)):
            for j in range(i):
                corr, p_val = stats.spearmanr(X.iloc[:, i], X.iloc[:, j]) if self.method == 'spearman' else \
                              (stats.pearsonr(X.iloc[:, i], X.iloc[:, j]) if self.method == 'pearson' else \
                               stats.kendalltau(X.iloc[:, i], X.iloc[:, j]))
                
                if abs(corr) > self.thsld and p_val < self.p_val_tshld:
                    corr_counts[cols[i]] += 1
                    corr_counts[cols[j]] += 1
                    mean_corrs[cols[i]].append(abs(corr))
                    mean_corrs[cols[j]].append(abs(corr))
                    pairs.append((cols[i], cols[j]))

        mean_corrs_avg = {col: np.mean(mean_corrs[col]) for col in cols if mean_corrs[col]}
        to_remove = set()

        for col_1, col_2 in pairs:
            if corr_counts[col_1] > corr_counts[col_2]:
                to_remove.add(col_1)
            else:
                to_remove.add(col_2)

        self.to_remove_ = list(to_remove)
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns)
        return X.drop(columns=self.to_remove_, errors='ignore')

class SMOTETomekCustom(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_strategy='auto', random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote_tomek = SMOTETomek(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
    
    def fit(self, X, y=None):
        self.smote_tomek.fit_resample(X, y)
        return self
    
    def transform(self, X, y=None):
        # Note: SMOTE is not applied here, it's only applied during fit
        return X

def download_zip_df(file_url):    
    try:
        response = requests.get(file_url)
        # Check if the request was successful
        if response.status_code == 200:
            # Open the response content as bytes
            with ZipFile(BytesIO(response.content)) as thezip:
                # Get a list of all archived file names from the zip
                all_files = thezip.namelist()
                # Find the first CSV file in the ZIP archive
                csv_file_name = next((s for s in all_files if ".csv" in s), None)
                if csv_file_name is not None:
                    # Open the CSV file within the zipfile
                    with thezip.open(csv_file_name) as csv_file:
                        df = pd.read_csv(csv_file)
                        display(df.head())
                        return df
                else:
                    print("No CSV file found in the ZIP archive.")
                    return None
        else:
            print('Failed to download the file: Status code', response.status_code)
            return None
    except Exception as e:
        print(f'An error occurred: {e}')
        return None

class TqdmCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total)
        self.update_interval = total // 20  # Update every 5%
        self.last_update = 0

    def __call__(self, res):
        current_progress = len(res.x_iters)
        if current_progress - self.last_update >= self.update_interval:
            self.pbar.update(current_progress - self.last_update)
            self.last_update = current_progress

def numerical_categorical_cols(df):
    """
    Separates numerical and categorical columns from a DataFrame.
    
    Parameters:
    df (pd.DataFrame or np.ndarray): The input DataFrame or array.
    
    Returns:
    tuple: A tuple containing two lists: the first list contains the names of numerical columns,
           and the second list contains the names of categorical columns.
    """
    # If the input is a NumPy array, convert it to a DataFrame
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    return numerical_cols, categorical_cols

def Standard_Outlier_Remover(X_train, y_train, num_cols, num_std=3):
    """
    Remove outliers from specified numeric columns in X_train and corresponding records in y_train.
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        The original DataFrame of features from which to remove outliers.
    y_train : pandas Series or DataFrame
        The original labels corresponding to X_train's records.
    num_cols : list of str
        List of column names in X_train that are numeric and should be checked for outliers.
    num_std : int or float, optional
        The number of standard deviations to use for defining outliers. Default is 3.
        
    Returns:
    --------
    X_train_clean : pandas DataFrame
        A new DataFrame of features with outliers removed.
    y_train_clean : pandas Series or DataFrame
        A new set of labels corresponding to the cleaned X_train.
    ratio : float
        The percentage of observations in the original X_train that were identified as outliers.
    outliers : list of int
        List of row indices in the original X_train that were identified as outliers.
    """
    
    X_train_clean = X_train.copy()
    outliers = []
    
    for col in num_cols:
        mean = X_train_clean[col].mean()
        std = X_train_clean[col].std()
        tol = num_std * std
        upper = mean + tol
        lower = mean - tol
        
        # Using vectorized operations to find outliers
        outlier_mask = (X_train_clean[col] > upper) | (X_train_clean[col] < lower)
        outliers.extend(X_train_clean.index[outlier_mask].tolist())
        
    # Remove duplicates
    outliers = list(set(outliers))
    
    # Calculate ratio of outliers
    ratio = round(len(outliers) / len(X_train_clean) * 100, 2)
    
    # Drop outliers from X_train and y_train
    X_train_clean = X_train_clean.drop(index=outliers)
    y_train_clean = y_train.drop(index=outliers)
    
    return X_train_clean, y_train_clean, ratio, outliers

def Winsorization_Method(df_source, columns,  lower, upper):
    """
    Remove outliers from the given DataFrame based on the provided lower and upper percentiles.

    Parameters:
    -----------
    df_source : pandas.DataFrame
        The original DataFrame from which outliers will be removed.
    columns : list of str
        The names of the columns to consider for outlier removal.
    lower : float
        The lower percentile below which data points are considered as outliers.
    upper : float
        The upper percentile above which data points are considered as outliers.

    Returns:
    --------
    ratio : float
        The ratio of outliers in the original DataFrame, rounded to two decimal places.
    df_win : pandas.DataFrame
        A new DataFrame with the outliers removed.
    
    Example:
    --------
    >>> ratio, df_win = Winsorization_Method(df, ['A', 'B'], 10, 90)
    
    Notes:
    ------
    - The function makes a copy of the original DataFrame, so the original DataFrame remains unchanged.
    - Outliers are determined separately for each column and are not considered across multiple columns.

    """
    #Data preparation
    df = df_source.copy()
    df = df[columns]

    #Determining records with outliers
    outliers=[]
    
    for col in columns:
        q1 = np.percentile(df[col], lower)
        q2 = np.percentile(df[col], upper)
        
        for pos in range(len(df)):
            if df[col].iloc[pos]>q2 or df[col].iloc[pos]<q1:
                outliers.append(pos) 
                
    outliers = set(outliers)                   
    outliers = list(outliers)
    
    ratio= round(len(outliers)/len(df)*100, 2) #calculating the ratio of outliers in the original data                     
    df_win = df.drop(df.index[outliers]) 
    
    return ratio, df_win

def analyze_columns(data, include_columns=None, plot=True):
    """
    Analyze specified columns of a DataFrame or Series for skewness, kurtosis, and visual distribution.

    Parameters:
    - data: DataFrame or Series to analyze.
    - include_columns: List of column names to include in the analysis. Default is None (takes all columns).
    - plot: Boolean to determine whether to show visual plots (default is True).

    Returns:
    - A DataFrame with skewness and kurtosis values for the specified columns.
    """
    # If input is a Series, convert it to a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
        # If Series has no name, set a default column name
        if data.columns[0] == 0:
            data.columns = ['Unnamed_Series']

    # If input is a DataFrame, proceed normally
    df = data

    # Use all columns if no specific columns are provided
    if include_columns is None:
        include_columns = df.columns

    # Create an empty DataFrame to store the results
    results = pd.DataFrame(columns=['Column', 'Skewness', 'Kurtosis'])
    
    for col in include_columns:
        if col in df.columns:
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            
            # Append results to the DataFrame
            new_row = pd.DataFrame({'Column': [col], 'Skewness': [skewness], 'Kurtosis': [kurtosis]})
            results = pd.concat([results, new_row], ignore_index=True)

            if plot:
                # Set up the matplotlib figure
                plt.figure(figsize=(14, 4))

                # Distribution plot
                plt.subplot(131)
                sns.histplot(df[col], kde=True)
                plt.title(f'Distribution of {col}')

                # Box plot
                plt.subplot(132)
                sns.boxplot(y=df[col])
                plt.title(f'Boxplot of {col}')

                # Q-Q plot
                plt.subplot(133)
                stats.probplot(df[col], plot=plt, rvalue=True, dist='norm')
                plt.title(f'Q-Q plot of {col}')

                plt.suptitle(f'Analysis of {col}')
                plt.show()
                
                # Printing the results
                print(f"Skewness of {col}: {skewness}")
                print(f"Kurtosis of {col}: {kurtosis}")
        else:
            print(f"Column '{col}' not found in DataFrame.")
    
    # Return the results DataFrame with skewness and kurtosis values
    return results

def normality_tests(data):
    """
    Perform various statistical tests to assess the normality of a given dataset.

    This function conducts the following normality tests on the provided data:
    1. Shapiro-Wilk Test
    2. Kolmogorov-Smirnov Test
    3. Anderson-Darling Test
    4. Lilliefors Test (if available)
    5. D’Agostino’s K-squared Test

    Parameters:
    - data (array-like or DataFrame): The data should be a 1D array-like structure (e.g., list, numpy array, pandas Series) 
                                      or a 2D pandas DataFrame.

    Returns:
    - results (DataFrame): A DataFrame containing the results of the normality tests for each column.
    """
    
    def run_tests(column_data):
        # Shapiro-Wilk Test
        shapiro_stat, shapiro_p = stats.shapiro(column_data)

        # Kolmogorov-Smirnov Test
        ks_stat, ks_p = stats.kstest(column_data, 'norm', args=(np.mean(column_data), np.std(column_data)))

        # Anderson-Darling Test
        anderson_result = stats.anderson(column_data, dist='norm')
        anderson_stat = anderson_result.statistic

        # Lilliefors Test (if available)
        try:
            from statsmodels.stats.diagnostic import lilliefors
            lilliefors_stat, lilliefors_p = lilliefors(column_data)
        except ImportError:
            lilliefors_stat, lilliefors_p = np.nan, np.nan  # Handle case where statsmodels is not available

        # D’Agostino’s K-squared Test
        dagostino_stat, dagostino_p = stats.normaltest(column_data)

        # Collect results in a dictionary
        return {
            'Shapiro-Wilk Stat': shapiro_stat,
            'Shapiro-Wilk p': shapiro_p,
            'KS Stat': ks_stat,
            'KS p': ks_p,
            'Anderson-Darling Stat': anderson_stat,
            'Lilliefors Stat': lilliefors_stat,
            'Lilliefors p': lilliefors_p,
            'D’Agostino Stat': dagostino_stat,
            'D’Agostino p': dagostino_p
        }

    # Check if the input is 1D (list, numpy array, pandas Series)
    if isinstance(data, (list, np.ndarray, pd.Series)):
        results = run_tests(data)
        return pd.DataFrame(results, index=[0])

    # Check if the input is 2D (pandas DataFrame)
    elif isinstance(data, pd.DataFrame):
        results = {}
        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                results[column] = run_tests(data[column])

        # Convert the results into a DataFrame
        return pd.DataFrame(results, index=[
            'Shapiro-Wilk Stat', 'Shapiro-Wilk p', 
            'KS Stat', 'KS p', 
            'Anderson-Darling Stat', 
            'Lilliefors Stat', 'Lilliefors p', 
            'D’Agostino Stat', 'D’Agostino p'
        ])

    else:
        raise TypeError("The data should be a 1D array-like structure (e.g., Python list, NumPy array, or Pandas Series) or a 2D pandas DataFrame.")

def heatmap_spearman_significance(df, annot=True):
    """
    Generate a heatmap displaying the Spearman's rank correlation coefficients 
    for each pair of variables in the provided DataFrame. If annot is True, 
    p-values and significance levels are annotated within each cell. Also, 
    return a DataFrame with the Spearman's rank correlation coefficients and 
    p-values for each pair of variables.

    Parameters:
    df (DataFrame): A pandas DataFrame containing the variables for which correlations 
                    are to be calculated. The DataFrame should contain only numeric 
                    columns.
    annot (bool): If True, display p-values and significance levels in the heatmap. 
                  Default is True.

    Returns:
    DataFrame: A DataFrame containing the Spearman's rank correlation coefficients and 
               p-values for each pair of variables.
    """
    # Initialize matrices
    correlations = df.corr(method='spearman')
    annotations = pd.DataFrame(index=df.columns, columns=df.columns)
    results = []

    # Calculate Spearman correlation and p-values, format annotations
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                corr, p_val = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                # Determine the significance level
                if p_val < 0.01:
                    sig = '***'
                elif p_val < 0.05:
                    sig = '**'
                elif p_val < 0.1:
                    sig = '*'
                else:
                    sig = ''
                # Format annotation with correlation and p-value
                annotations.at[col1, col2] = f'{corr:.2f}\n({p_val:.3f}){sig}'
                # Append results to list
                results.append({'Pair': f'{col1}-{col2}', 'Correlation': corr, 'P-value': p_val})
            else:
                annotations.at[col1, col2] = ''

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlations, dtype=bool))

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    if annot:
        sns.heatmap(correlations, annot=annotations, mask=mask, cmap='coolwarm', fmt='', cbar_kws={'label': 'Spearman Correlation Coefficient'})
    else:
        sns.heatmap(correlations, annot=False, mask=mask, cmap='coolwarm', cbar_kws={'label': 'Spearman Correlation Coefficient'})
    plt.title("Spearman's Rank Correlation Heatmap" + (" with P-value and Significance Annotations" if annot else ""))
    plt.show()

    # Convert results list to DataFrame and return
    results_df = pd.DataFrame(results)
    return results_df.set_index('Pair')

def plot_feature_importances(model, X, y, random_state=None):
    model.fit(X, y)
    importances = model.feature_importances_
    plt.figure(figsize=(10, len(X.columns) * 0.3))
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    mdi_importances = pd.Series(importances, index=X.columns).sort_values(ascending=True)
    ax = mdi_importances.plot.barh(yerr=std)
    ax.set_title("Random Forest Feature Importances (MDI)")
    ax.figure.tight_layout()
    plt.show()

def high_corr(df_source, method='spearman', thsld=0.8, perc=0.95):
    """
    Remove columns from a DataFrame that are highly correlated with other columns.

    Parameters:
    -----------
    df_source : pandas.DataFrame
        The original DataFrame for which to identify and remove highly correlated columns.

    method : str, optional
        The correlation method to be used. 'pearson', 'kendall', 'spearman' are supported. 
        Default is 'spearman'.

    thsld : float, optional
        The absolute correlation threshold. Pairs of columns with correlation higher than this value 
        will be considered for removal. Default is 0.8.

    perc : float, optional
        The percentile to use for deciding which variable to remove from a highly correlated pair.
        Should be between 0 and 1. Default is 0.95.

    Returns:
    --------
    df : pandas.DataFrame
        A new DataFrame with highly correlated columns removed.
    to_delete: list
        A list of removed variables. 

    Notes:
    ------
    - For each pair of highly correlated columns, the function removes the one with the higher
      percentile value of correlation, based on the 'perc' parameter.
    - Prints out the columns that are removed and their percentile values.
    """
    df =  df_source.copy()
    corr = df.corr(method)
    high_corr_vars = []
    to_delete = [] 
    
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > thsld:
                colname_i = corr.columns[i]
                colname_j = corr.columns[j]

                # Calculate correlation for each variable
                perc_i = np.percentile(corr[colname_i], perc)
                perc_j = np.percentile(corr[colname_j], perc)

                # Decide which variable to remove from each correlated pair based on its highest percentile value in the correlation matrix.
                max_abs_value = max(abs(perc_i), abs(perc_j))
                if max_abs_value >= thsld:
                    if abs(perc_i) > abs(perc_j):
                        high_corr_vars.append((colname_i, perc_i))
                    else:
                        high_corr_vars.append((colname_j, perc_j))
                else:
                    if abs(perc_i) <= abs(perc_j):
                        high_corr_vars.append((colname_i, perc_i))
                    else:
                        high_corr_vars.append((colname_j, perc_j))

                    

    # Remove the variable with the highest mean correlation from each pair
    for var, perc_val in high_corr_vars:
        if var in df.columns:
            print(f"Removing {var} with {perc}th percentile = {perc_val}")
            del df[var]
            to_delete.append(var)
    print("Remaining columns:", df.columns)
    return df, to_delete

def compare_distributions(df_1, df_2, features=None, show_plot=False):
    """
    Compares the distributions of specified features in two datasets, handles inf values,
    and returns a DataFrame with p-values. Optionally plots distribution comparisons.

    Parameters:
    - df_1: The first dataset (pandas DataFrame or Series).
    - df_2: The second dataset (pandas DataFrame or Series).
    - features: A list of feature names to compare. If None, all features in df_1 are compared.
    - show_plot: Boolean, if True, plots the distribution comparison for each feature.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    def filter_numeric(df):
        if isinstance(df, pd.Series):
            if pd.api.types.is_numeric_dtype(df):
                return df
            else:
                return pd.Series(dtype='float64')
        else:
            return df.select_dtypes(include=[np.number])
    
    df_1 = filter_numeric(df_1)
    df_2 = filter_numeric(df_2)
    
    if isinstance(df_1, pd.Series) and isinstance(df_2, pd.Series):
        clean_df_1 = df_1.dropna()
        clean_df_2 = df_2.dropna()
        u_statistic, p_value = stats.mannwhitneyu(clean_df_1, clean_df_2)
        if show_plot:
            plt.figure(figsize=(10, 6))
            sns.histplot(clean_df_1, color='blue', label='Dataset 1', kde=True, stat="density", linewidth=0, bins=30)
            sns.histplot(clean_df_2, color='red', label='Dataset 2', kde=True, stat="density", linewidth=0, bins=30)
            plt.title('Distribution Comparison')
            plt.legend()
            plt.show()
        return pd.DataFrame([{'Feature': 'Series Comparison', 'P-Value': p_value}])
    
    if features is None:
        features = df_1.columns.tolist()

    results = []

    for column in features:
        clean_df_1 = df_1[column].dropna()
        clean_df_2 = df_2[column].dropna()
        
        # Ensure there are no non-numeric values
        clean_df_1 = clean_df_1[clean_df_1.apply(lambda x: np.isreal(x))]
        clean_df_2 = clean_df_2[clean_df_2.apply(lambda x: np.isreal(x))]
        
        if len(clean_df_1) > 0 and len(clean_df_2) > 0:
            u_statistic, p_value = stats.mannwhitneyu(clean_df_1, clean_df_2)
            results.append({'Feature': column, 'P-Value': p_value})
            if show_plot:
                plt.figure(figsize=(10, 6))
                sns.histplot(clean_df_1, color='blue', label='Dataset 1', kde=True, stat="density", linewidth=0, bins=30)
                sns.histplot(clean_df_2, color='red', label='Dataset 2', kde=True, stat="density", linewidth=0, bins=30)
                plt.title(f'Distribution of {column}')
                plt.legend()
                plt.show()

    return pd.DataFrame(results)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Create a mask to avoid division by zero
    non_zero_mask = y_true != 0

    # Calculate the MAPE
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def test_regression(model, X_train, X_test, y_train, y_test, cv):    
    #initialize the model
    model.fit(X_train, y_train)
    train_mse = -np.mean(cross_val_score(model, X_train, y_train, cv = cv, \
                                n_jobs = -1, scoring='neg_mean_squared_error'))
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    #Calculate scores
    test_mse = mean_squared_error(y_test, y_pred)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
    test_mape = mean_absolute_percentage_error(y_test, y_pred)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred)
    
    #
    score_names = ['train_mse', 'test_mse', 'train_mae', 'test_mae', 'train_mape',\
                   'test_mape', 'train_r2', 'test_r2']
    results = [train_mse, test_mse, train_mae, test_mae, train_mape, test_mape, train_r2, test_r2]
    results_dict = {score_names[i]: [results[i]] for i in range(len(score_names))}
    return results_dict

def test_model_with_cv(model, X, y, cv, model_type):
    results_dict = {}

    if model_type == 'regression':
        # Define regression scorers
        scorers = {
            'MSE': make_scorer(mean_squared_error, greater_is_better=False),
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
            'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            'R2': make_scorer(r2_score)
            
        }

    elif model_type == 'classification':
        # Define classification scorers
        scorers = {
            'Accuracy': make_scorer(accuracy_score),
            'Precision': make_scorer(precision_score, average='weighted'),
            'Recall': make_scorer(recall_score, average='weighted'),
            'F1': make_scorer(f1_score, average='weighted')
            # Note: ROC AUC needs to be handled separately due to its requirement for probability scores
        }

    else:
        raise ValueError("Invalid model_type specified. Choose 'regression' or 'classification'.")

    # Use cross_validate to calculate both test (CV) and train scores
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scorers, return_train_score=True)

    # Store the mean of the scores
    for score_name in scorers.keys():
        train_score_key = 'train_' + score_name
        test_score_key = 'test_' + score_name
        results_dict[score_name + ' Train Score'] = np.mean(cv_results[train_score_key])
        results_dict[score_name + ' Test Score'] = np.mean(cv_results[test_score_key])

    # Handle ROC AUC for classification separately
    if model_type == 'classification':
        try:
            y_probas = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=-1)
            roc_auc = roc_auc_score(y, y_probas[:, 1])  # Assuming y_probas[:, 1] are the probabilities for the positive class
            results_dict['ROC AUC CV Score'] = roc_auc
        except AttributeError as e:
            print("The classifier does not support predict_proba, and ROC AUC cannot be calculated.", e)
    
    model.fit(X, y)
    return model, results_dict

def build_and_train_pipeline(X, y, problem_type, sort_by,
                             model_steps,
                             scoring,
                             outlier_trimmer_step=None,
                             preprocessing_steps=None,
                             n_folds=3,
                             SEED=23):

    # Apply outlier trimming if an OutlierTrimmer instance is provided
    if outlier_trimmer_step is not None:
        outlier_trimmer_step.fit(X)
        X_trim, y_trim = outlier_trimmer_step.transform_x_y(X, y)
    else:
        X_trim, y_trim = X, y
    
    # Run the preprocessing pipeline
    if preprocessing_steps is not None:
        preprocessing = Pipeline(steps = preprocessing_steps)
        preprocessing.fit(X_trim, y_trim)
        X_preprocessed = preprocessing.transform(X_trim)
    else:
        X_preprocessed = X_trim
    
    results = []
    for name, model in model_steps:
        # Perform cross-validation and collect the metrics
        cv_results = cross_validate(model, X_preprocessed, y_trim, cv=n_folds, scoring=scoring, n_jobs=-1, return_train_score=True)
        
        results_dict = {'model': name}
        for metric_name in scoring.keys():
            train_metric_mean = np.mean(cv_results[f'train_{metric_name}'])
            train_metric_std = np.std(cv_results[f'train_{metric_name}'])
            test_metric_mean = np.mean(cv_results[f'test_{metric_name}'])
            test_metric_std = np.std(cv_results[f'test_{metric_name}'])

            results_dict[metric_name + "_train"] = abs(train_metric_mean)
            results_dict[metric_name + "_train_std"] = abs(train_metric_std)
            results_dict[metric_name + "_test"] = abs(test_metric_mean)
            results_dict[metric_name + "_test_std"] = abs(test_metric_std)

        results.append(results_dict)

    results_df = pd.DataFrame(results).sort_values(by=sort_by, ascending=False)
    
    # Rebuild and fit the best pipeline based on the selected sort criteria
    best_model_name = results_df.iloc[0]['model']
    best_model = [model for name, model in model_steps if name == best_model_name][0]
    best_pipeline_steps = preprocessing_steps + [(best_model_name, best_model)]
    best_pipeline = Pipeline(steps=best_pipeline_steps)
    best_pipeline.fit(X_trim, y_trim)

    return results_df, best_pipeline

def preprocessing_pipeline(X, y, 
                             outlier_trimmer=None,
                             preprocessing_steps=None,
                             n_folds=3,
                             SEED=23):

    # Apply outlier trimming if an OutlierTrimmer instance is provided
    if outlier_trimmer is not None:
        outlier_trimmer.fit(X)
        X_trim, y_trim = outlier_trimmer.transform_x_y(X, y)
    else:
        X_trim, y_trim = X, y
    
    # Run the preprocessing pipeline
    if preprocessing_steps is not None:
        proc_pipe = Pipeline(steps = preprocessing_steps)
        proc_pipe.fit(X_trim, y_trim)
        X_preprocessed = proc_pipe.transform(X_trim)
    else:
        X_preprocessed = X_trim
        proc_pipe = None
    
    return X_preprocessed, y_trim, outlier_trimmer, proc_pipe

def train_pipeline(X, y, problem_type, sort_by, sort_ascending,
                             model_steps,
                             scoring,
                             n_folds=3,
                             SEED=23):
    results = []
    for name, model in model_steps:
        # Perform cross-validation and collect the metrics
        cv_results = cross_validate(model, X, y, cv=n_folds, scoring=scoring, n_jobs=-1, return_train_score=True)
        
        results_dict = {'model': name}
        for metric_name in scoring.keys():
            train_metric_mean = np.mean(cv_results[f'train_{metric_name}'])
            train_metric_std = np.std(cv_results[f'train_{metric_name}'])
            test_metric_mean = np.mean(cv_results[f'test_{metric_name}'])
            test_metric_std = np.std(cv_results[f'test_{metric_name}'])

            results_dict[metric_name + "_train"] = abs(train_metric_mean)
            results_dict[metric_name + "_train_std"] = abs(train_metric_std)
            results_dict[metric_name + "_test"] = abs(test_metric_mean)
            results_dict[metric_name + "_test_std"] = abs(test_metric_std)

        results.append(results_dict)

    results_df = pd.DataFrame(results).sort_values(by=sort_by, ascending=sort_ascending)
    
    # Rebuild and fit the best pipeline based on the selected sort criteria
    best_model_name = results_df.iloc[0]['model']
    best_model = [model for name, model in model_steps if name == best_model_name][0]

    return results_df, best_model

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def check_false_categorical(df):
    """
    Analyze each column in the DataFrame to count distinct numeric and non-numeric values.

    This function iterates through each column in the provided DataFrame and checks if the column's
    data type is 'object'. For such columns, it categorizes each value as either numeric or non-numeric.
    It then counts the number of distinct numeric and non-numeric values and lists them if needed.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be analyzed.

    Returns
    -------
    results : dict
        A dictionary where each key is a column name from the DataFrame, and each value is another dictionary
        containing the following information:
        - 'Distinct Numeric Values': int
            The number of distinct numeric values in the column.
        - 'Distinct Non-Numeric Values': int
            The number of distinct non-numeric values in the column.
        - 'Numeric Values': numpy.ndarray
            An array of the distinct numeric values found in the column.
        - 'Non-Numeric Values': numpy.ndarray
            An array of the distinct non-numeric values found in the column.

    Example
    -------
    >>> data = {
    ...     'column1': ['123', '456', 'abc', '789', 'def', '123$', '45.6', '78.9'],
    ...     'column2': ['100', '200', '300', '400', '500', '600', '700', 'abc'],
    ...     'column3': [1, 2, 3, 4, 5, 6, 7, 8]
    ... }
    >>> df = pd.DataFrame(data)
    >>> result = analyze_columns(df)
    >>> for column, analysis in result.items():
    ...     print(f"Analysis of column '{column}':")
    ...     print(f"Distinct Numeric Values: {analysis['Distinct Numeric Values']}")
    ...     print(f"Distinct Non-Numeric Values: {analysis['Distinct Non-Numeric Values']}")
    ...     print(f"Numeric Values: {analysis['Numeric Values']}")
    ...     print(f"Non-Numeric Values: {analysis['Non-Numeric Values']}")
    ...     print()
    """
    results = {}
    
    for column in df.columns:
        if df[column].dtype == 'object':
            # Apply the is_numeric function to categorize values
            df['is_numeric'] = df[column].apply(is_numeric)
            
            # Count distinct numeric and non-numeric values
            distinct_numeric_values = df[df['is_numeric']][column].nunique()
            distinct_non_numeric_values = df[~df['is_numeric']][column].nunique()
            
            # Store the results in a dictionary
            results[column] = {
                'Distinct Numeric Values': distinct_numeric_values,
                'Distinct Non-Numeric Values': distinct_non_numeric_values,
                'Numeric Values': df[df['is_numeric']][column].unique(),
                'Non-Numeric Values': df[~df['is_numeric']][column].unique()
            }
    
    # Drop the temporary column used for checks
    if 'is_numeric' in df.columns:
        df.drop(columns=['is_numeric'], inplace=True)
    
    # Print the analysis results
    for column, analysis in results.items():
        print(f"Analysis of column '{column}':")
        print(f"Distinct Numeric Values: {analysis['Distinct Numeric Values']}")
        print(f"Distinct Non-Numeric Values: {analysis['Distinct Non-Numeric Values']}")
        print(f"Numeric Values: {analysis['Numeric Values']}")
        print(f"Non-Numeric Values: {analysis['Non-Numeric Values']}")

    return results

def calculate_skewness_kurtosis(df, nan_policy = 'omit'):
    """
    Calculate skewness and kurtosis for each numeric column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be analyzed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the skewness and kurtosis of each numeric column.
    """
    skew_kurtosis = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            col_skewness = skew(df[column], nan_policy = nan_policy)
            col_kurtosis = kurtosis(df[column], nan_policy = nan_policy)
            skew_kurtosis[column] = {
                'Skewness': col_skewness,
                'Kurtosis': col_kurtosis
            }
    return pd.DataFrame(skew_kurtosis).T

def linreg_p_values(model, X, y):
    """
    Calculate p-values for a linear regression model's coefficients.

    This function fits a linear regression model using the provided data and
    calculates the p-values for each predictor's coefficient. It assumes 
    that the model passed as an argument is an instance of a linear regression
    model from a library like sklearn, and that `X` and `y` are compatible
    with this model.

    Parameters:
    model: A linear regression model instance from sklearn or similar library.
           The model should not have been previously fitted.
    X: DataFrame or 2D array-like
       The input variables (predictors) for the regression model. If a DataFrame
       is used, columns should have names for predictor identification in the output.
    y: Array-like
       The target variable (response) for the regression model.

    Returns:
    final_df: DataFrame
              A pandas DataFrame containing the predictor names, estimated coefficients,
              t-statistics, and p-values for each predictor in the model.
    """
    # Fit the model to the data
    model.fit(X, y)
    coefficients = model.coef_
    predictions = model.predict(X)
    residuals = y - predictions
    
    # Calculate degrees of freedom
    n, k = X.shape
    df = n - k - 1  # degrees of freedom for the residuals
    
    # Calculate Residual Sum of Squares (RSS)
    rss = np.sum(residuals**2)
    
    # Calculate standard errors of the coefficients
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])  # Add intercept to X
    inv_XTX = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    stderr = np.sqrt(np.diag(inv_XTX) * rss / df)
    
    # Calculate t-statistics and p-values in a vectorized manner
    t_stat = coefficients / stderr[1:]  # ignoring intercept term in stderr
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
    
    # Prepare final output
    predictors = X.columns if isinstance(X, pd.DataFrame) else [f'X{i+1}' for i in range(k)]
    data = {'Predictor': predictors, 'coef': coefficients, 't-stat': t_stat, 'p-values': p_values}
    final_df = pd.DataFrame(data)
    
    return final_df