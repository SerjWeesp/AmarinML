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
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split, cross_validate, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method=None):
        """
        Initialize the FlexibleOutlierHandler with a custom method.
        
        Parameters:
        -----------
        method : object
            An object with `fit` and `transform` methods for outlier handling.
            This can be any user-defined or external outlier detection framework.
        """
        if method is None or not hasattr(method, 'fit') or not hasattr(method, 'transform'):
            raise ValueError("The method must have both `fit` and `transform` methods.")
        self.method = method
        self.outlier_share_ = None  # Store the share of outliers removed

    def fit(self, X, y=None):
        self.method.fit(X, y)
        return self

    def transform(self, X, y=None):
        initial_row_count = len(X)
        X_transformed, y_transformed = self.method.transform(X, y) if y is not None else (self.method.transform(X), None)
        final_row_count = len(X_transformed)
        self.outlier_share_ = (initial_row_count - final_row_count) / initial_row_count
        return (X_transformed, y_transformed) if y is not None else X_transformed

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def get_outlier_indices(self, X):
        """
        Get the indices of outliers based on the method used.
        The method should implement a `get_outlier_indices` function.
        """
        if hasattr(self.method, "get_outlier_indices"):
            return self.method.get_outlier_indices(X)
        raise NotImplementedError("The selected method does not support `get_outlier_indices`.")

    def get_bounds(self):
        """
        Get the bounds for outlier detection (if supported by the method).
        The method should implement a `get_bounds` function.
        """
        if hasattr(self.method, "get_bounds"):
            return self.method.get_bounds()
        raise NotImplementedError("The selected method does not support `get_bounds`.")

    def print_outlier_share(self):
        if self.outlier_share_ is not None:
            print(f"Total share of outliers removed: {self.outlier_share_ * 100:.2f}%")
        else:
            print("Outlier share not calculated. Ensure `transform` method has been called.")

class IQRHandler:
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound_ = None
        self.upper_bound_ = None

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
        is_within_bounds = (X >= self.lower_bound_) & (X <= self.upper_bound_)
        X_filtered = X[is_within_bounds.all(axis=1)]
        if y is not None:
            y = pd.Series(y) if isinstance(y, np.ndarray) else y
            y_filtered = y.loc[X_filtered.index]
            return X_filtered, y_filtered
        return X_filtered

    def get_outlier_indices(self, X):
        X = pd.DataFrame(X)
        is_within_bounds = (X >= self.lower_bound_) & (X <= self.upper_bound_)
        return X[~is_within_bounds.all(axis=1)].index

    def get_bounds(self):
        """
        Return a DataFrame containing the bounds for each feature.
        """
        bounds = pd.DataFrame({
            'Feature': self.lower_bound_.index,
            'Lower Bound': self.lower_bound_.values,
            'Upper Bound': self.upper_bound_.values
        })
        bounds.set_index('Feature', inplace=True)
        return bounds

class IsolationForestHandler:
    def __init__(self, contamination=0.01, random_state=23):
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

    def get_outlier_indices(self, X):
        y_pred = self.clf.predict(X)
        return X.index[y_pred == -1]

    def get_bounds(self):
        """
        IsolationForest does not compute explicit bounds.
        """
        raise NotImplementedError("IsolationForest does not define specific bounds for features.")

class StdDevHandler:
    def __init__(self, num_std=3):
        self.num_std = num_std
        self.lower_bound_ = None
        self.upper_bound_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        means = X.mean()
        stds = X.std()
        self.lower_bound_ = means - self.num_std * stds
        self.upper_bound_ = means + self.num_std * stds
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        outliers = []
        for col in X.columns:
            mean = X[col].mean()
            std = X[col].std()
            tol = self.num_std * std
            upper = mean + tol
            lower = mean - tol
            outliers.extend(X.index[(X[col] > upper) | (X[col] < lower)].tolist())
        outliers = list(set(outliers))
        X_filtered = X.drop(index=outliers)
        if y is not None:
            y = pd.Series(y) if isinstance(y, np.ndarray) else y
            y_filtered = y.loc[X_filtered.index]
            return X_filtered, y_filtered
        return X_filtered

    def get_outlier_indices(self, X):
        X = pd.DataFrame(X)
        outliers = []
        for col in X.columns:
            mean = X[col].mean()
            std = X[col].std()
            tol = self.num_std * std
            upper = mean + tol
            lower = mean - tol
            outliers.extend(X.index[(X[col] > upper) | (X[col] < lower)].tolist())
        return list(set(outliers))

    def get_bounds(self):
        """
        Return a DataFrame containing the bounds for each feature.
        """
        bounds = pd.DataFrame({
            'Feature': self.lower_bound_.index,
            'Lower Bound': self.lower_bound_.values,
            'Upper Bound': self.upper_bound_.values
        })
        bounds.set_index('Feature', inplace=True)
        return bounds

class DBSCANHandler:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.clf = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self

    def transform(self, X, y=None):
        labels = self.clf.labels_
        mask = labels != -1  # Points labeled -1 are considered noise (outliers)
        if y is not None:
            return X[mask], y[mask]
        return X[mask]

    def get_outlier_indices(self, X):
        labels = self.clf.labels_
        return X.index[labels == -1]

    def get_bounds(self):
        """
        DBSCAN does not compute explicit bounds for features.
        """
        raise NotImplementedError("DBSCAN does not define specific bounds for features.")

class OneClassSVMHandler:
    def __init__(self, kernel="rbf", gamma="scale", nu=0.05):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.clf = OneClassSVM(kernel=self.kernel, gamma=self.gamma, nu=self.nu)

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self

    def transform(self, X, y=None):
        y_pred = self.clf.predict(X)
        mask = y_pred == 1
        if y is not None:
            return X[mask], y[mask]
        return X[mask]

    def get_outlier_indices(self, X):
        y_pred = self.clf.predict(X)
        return X.index[y_pred == -1]

    def get_bounds(self):
        """
        One-Class SVM does not provide explicit feature-wise bounds.
        """
        raise NotImplementedError("One-Class SVM does not define specific bounds for features.")

class EllipticEnvelopeHandler:
    def __init__(self, contamination=0.01):
        self.contamination = contamination
        self.clf = EllipticEnvelope(contamination=self.contamination)

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self

    def transform(self, X, y=None):
        y_pred = self.clf.predict(X)
        mask = y_pred != -1
        if y is not None:
            return X[mask], y[mask]
        return X[mask]

    def get_outlier_indices(self, X):
        y_pred = self.clf.predict(X)
        return X.index[y_pred == -1]

    def get_bounds(self):
        """
        Elliptic Envelope does not provide explicit feature-wise bounds.
        """
        raise NotImplementedError("EllipticEnvelope does not define specific bounds for features.")

class PCAOutlierHandler:
    def __init__(self, n_components=None, contamination=0.05, threshold_method='reconstruction_error', dynamic_threshold=False):
        """
        Parameters:
        -----------
        dynamic_threshold : bool, optional
            If True, contamination is ignored, and outliers are flagged using a fixed reconstruction error or Mahalanobis threshold.
        """
        self.n_components = n_components
        self.contamination = contamination
        self.threshold_method = threshold_method
        self.dynamic_threshold = dynamic_threshold
        self.pca = PCA(n_components=n_components)
        self.threshold_ = None
        self.errors_ = None

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        X_pca = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        reconstruction_error = np.mean((X - X_reconstructed) ** 2, axis=1)
        self.errors_ = reconstruction_error

        if self.threshold_method == 'reconstruction_error':
            if self.dynamic_threshold:
                # Fixed threshold example (adjust as needed)
                self.threshold_ = 0.01 * np.var(X.values)
            else:
                # Contamination-based threshold
                self.threshold_ = np.percentile(reconstruction_error, (1 - self.contamination) * 100)
            mask = reconstruction_error <= self.threshold_
        elif self.threshold_method == 'mahalanobis':
            cov = np.cov(X_pca.T)
            inv_cov = np.linalg.inv(cov)
            mean = np.mean(X_pca, axis=0)
            mahalanobis_distance = np.array([np.dot(np.dot((x - mean), inv_cov), (x - mean).T) for x in X_pca])
            self.errors_ = mahalanobis_distance
            if self.dynamic_threshold:
                # Fixed Mahalanobis distance threshold example
                self.threshold_ = 3  # Arbitrary cutoff, tune as needed
            else:
                # Contamination-based threshold
                self.threshold_ = chi2.ppf(1 - self.contamination, df=self.n_components or X.shape[1])
            mask = mahalanobis_distance <= self.threshold_
        else:
            raise ValueError("Unsupported threshold method. Use 'reconstruction_error' or 'mahalanobis'.")

        if y is not None:
            return X[mask], y[mask]
        return X[mask]

    def get_outlier_indices(self, X):
        """
        Get the indices of outliers.
        """
        if self.errors_ is None:
            raise ValueError("Model has not been fitted or transformed. Call `fit` or `transform` first.")
        return np.where(self.errors_ > self.threshold_)[0]

    def get_bounds(self):
        """
        PCA does not define explicit feature-wise bounds, but we can provide the threshold.
        """
        return pd.DataFrame({
            'Threshold Method': [self.threshold_method],
            'Threshold Value': [self.threshold_]
        })

    def get_reconstruction_errors(self):
        """
        Return the reconstruction errors for all data points.
        """
        if self.errors_ is None:
            raise ValueError("Model has not been fitted or transformed. Call `fit` or `transform` first.")
        return self.errors_

class WinsorizerHandler:
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        """
        Initialize the WinsorizerHandler.

        Parameters:
        -----------
        lower_quantile : float, optional
            The lower quantile threshold for Winsorization. Default is 0.01 (1st percentile).
        upper_quantile : float, optional
            The upper quantile threshold for Winsorization. Default is 0.99 (99th percentile).
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.lower_bound_y_ = None
        self.upper_bound_y_ = None

        self.outlier_share_ = None  # Store the proportion of Winsorized points (X + y combined)

    def fit(self, X, y=None):
        """
        Compute the quantile bounds for Winsorization.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Input data to compute bounds.
        y : array-like or Series, optional
            Target data to compute bounds if we want to winsorize y as well.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        # 1) Handle X
        X = pd.DataFrame(X)
        self.lower_bounds_ = X.quantile(self.lower_quantile)
        self.upper_bounds_ = X.quantile(self.upper_quantile)

        # 2) Handle y if provided
        if y is not None:
            y_series = pd.Series(y).astype(float)
            self.lower_bound_y_ = y_series.quantile(self.lower_quantile)
            self.upper_bound_y_ = y_series.quantile(self.upper_quantile)
        else:
            self.lower_bound_y_ = None
            self.upper_bound_y_ = None

        return self

    def transform(self, X, y=None):
        """
        Apply Winsorization (clip) to the dataset (X) and optionally the target (y).
        Also compute the outlier share.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Input data to Winsorize.
        y : array-like or Series, optional
            Target data to Winsorize if needed.

        Returns:
        --------
        X_winsorized : pandas DataFrame
            The Winsorized dataset for X.
        y_winsorized : pandas Series or None
            The Winsorized target if y was provided, otherwise None.
        """
        X = pd.DataFrame(X)
        # Identify where values are below or above bounds for X
        outlier_mask_X = (X < self.lower_bounds_) | (X > self.upper_bounds_)
        total_outliers_X = outlier_mask_X.sum().sum()  # total outlier cells in X

        # Winsorize (clip) the data
        X_winsorized = X.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)

        y_winsorized = None
        total_outliers_y = 0  # we'll count how many values in y are clipped if y is provided

        # If y is provided, winsorize y as well
        if y is not None and self.lower_bound_y_ is not None and self.upper_bound_y_ is not None:
            y_series = pd.Series(y).astype(float)
            outlier_mask_y = (y_series < self.lower_bound_y_) | (y_series > self.upper_bound_y_)
            total_outliers_y = outlier_mask_y.sum()
            y_winsorized = y_series.clip(lower=self.lower_bound_y_, upper=self.upper_bound_y_)

        # Calculate the total share of outliers (X + y)
        total_values_X = X.size
        total_values_y = len(y) if y is not None else 0
        total_values = total_values_X + total_values_y

        total_outliers = total_outliers_X + total_outliers_y
        self.outlier_share_ = total_outliers / total_values if total_values > 0 else 0

        if y is not None:
            return X_winsorized, y_winsorized
        else:
            return X_winsorized

    def fit_transform(self, X, y=None):
        """
        Fit and transform the dataset (X) and optionally the target (y) in a single step.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Input data to Winsorize.
        y : array-like or Series, optional
            Target data to Winsorize if needed.

        Returns:
        --------
        X_winsorized : pandas DataFrame
            The Winsorized dataset for X.
        y_winsorized : pandas Series or None
            The Winsorized target if y was provided, otherwise None.
        """
        self.fit(X, y=y)
        return self.transform(X, y=y)

    def get_bounds(self):
        """
        Get the lower and upper quantile bounds for each feature in X.
        If y was included, also return the bounds for y.
        
        Returns:
        --------
        bounds : dict
            Dictionary with:
              - "X": DataFrame of X's lower & upper bounds
              - "y": (lower_bound, upper_bound) if y was fit
        """
        bounds_X = pd.DataFrame({
            'Lower Bound': self.lower_bounds_,
            'Upper Bound': self.upper_bounds_
        })

        if self.lower_bound_y_ is not None and self.upper_bound_y_ is not None:
            bounds_y = (self.lower_bound_y_, self.upper_bound_y_)
        else:
            bounds_y = None

        return {
            'X': bounds_X,
            'y': bounds_y
        }

    def get_outlier_indices(self, X, y=None):
        """
        Get the indices of potential outliers that were Winsorized in X and optionally y.

        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Input data to check for Winsorized outliers.
        y : array-like or Series, optional
            Target data to check for Winsorized outliers.

        Returns:
        --------
        outlier_indices_X : list
            List of indices where X was Winsorized in at least one column.
        outlier_indices_y : list or None
            List of indices where y was Winsorized (if y was provided), otherwise None.
        """
        X = pd.DataFrame(X)
        outlier_mask_X = (X < self.lower_bounds_) | (X > self.upper_bounds_)
        outlier_indices_X = X[outlier_mask_X.any(axis=1)].index.tolist()

        outlier_indices_y = None
        if y is not None and self.lower_bound_y_ is not None and self.upper_bound_y_ is not None:
            y_series = pd.Series(y).astype(float)
            outlier_mask_y = (y_series < self.lower_bound_y_) | (y_series > self.upper_bound_y_)
            outlier_indices_y = y_series[outlier_mask_y].index.tolist()

        return outlier_indices_X, outlier_indices_y

    def print_outlier_share(self):
        """
        Print the proportion of data points (cells) that were Winsorized in X plus any y.
        """
        if self.outlier_share_ is not None:
            print(f"Total share of Winsorized cells: {self.outlier_share_ * 100:.2f}%")
        else:
            print("Outlier share not calculated. Ensure `transform` or `fit_transform` has been called.")

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
            'Shapiro-Wilk p': round(shapiro_p, 6),
            'KS Stat': ks_stat,
            'KS p': round(ks_p,6),
            'Anderson-Darling Stat': anderson_stat,
            'Lilliefors Stat': lilliefors_stat,
            'Lilliefors p': round(lilliefors_p, 6),
            'D’Agostino Stat': dagostino_stat,
            'D’Agostino p': round(dagostino_p, 6)
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
        ]).T

    else:
        raise TypeError("The data should be a 1D array-like structure (e.g., Python list, NumPy array, or Pandas Series) or a 2D pandas DataFrame.")

def corr_heatmap_significance(df, method = 'spearman', annot=True):
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
    correlations = df.corr(method=method)
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
        sns.heatmap(correlations, annot=annotations, mask=mask, cmap='coolwarm', fmt='', cbar_kws={'label': 'Correlation Coefficient'})
    else:
        sns.heatmap(correlations, annot=False, mask=mask, cmap='coolwarm', cbar_kws={'label': 'Spearman Correlation Coefficient'})
    plt.title("Correlation Heatmap" + (" with P-value and Significance Annotations" if annot else ""))
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
            col_skewness = stats.skew(df[column], nan_policy = nan_policy)
            col_kurtosis = stats.kurtosis(df[column], nan_policy = nan_policy)
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

def plot_grid_search(cv_results, grid_params):
    """
    Plots the test scores from a Grid Search as either a line plot or heatmap based on the number of hyperparameters.

    Parameters:
    -----------
    cv_results : dict
        A dictionary of cross-validation results, usually obtained from `GridSearchCV.cv_results_`.
    
    grid_params : list of str
        A list of hyperparameter names that are part of the grid search.

    Returns:
    --------
    None

    Notes:
    ------
    - If there's only one hyperparameter, the function will plot a line plot.
    - If there are two hyperparameters, the function will plot a heatmap.
    - If there are more than two hyperparameters, the function will print a message indicating its limitation.
    
    Example:
    --------
    >>> plot_grid_search(cv_results, ['param1', 'param2'])
    [Heatmap is displayed]

    >>> plot_grid_search(cv_results, ['param1'])
    [Line plot is displayed]
    """
    #Convert to DataFrame
    results = pd.DataFrame(cv_results)

    # If only one hyperparameter, plot a line plot
    if len(grid_params) == 1:
        param = 'param_' + grid_params[0]
        plt.figure(figsize=(8, 6))
        plt.plot(results[param], results['mean_test_score'], marker='o')
        plt.xlabel(grid_params[0])
        plt.ylabel('Mean Test Score')
        plt.title(f"Grid Search Test Scores")
        plt.show()
    
    # If two hyperparameters, plot a heatmap
    elif len(grid_params) == 2:
        pivot_table = results.pivot(index=f'param_{grid_params[0]}', 
                                    columns=f'param_{grid_params[1]}', 
                                    values='mean_test_score')
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
        plt.xlabel(grid_params[1])
        plt.ylabel(grid_params[0])
        plt.title(f"Grid Search Test Score Heatmap")
        plt.show()

    # For more than two hyperparameters, print a message
    else:
        print("The function is designed to plot up to two hyperparameters.")

def grid_log_c(X_train, y_train, X_test, y_test, C_values, penalty='l1', solver = 'liblinear', random_state=23):
    """
    Evaluates and plots the train and test accuracy of a logistic regression model for various C values.

    Parameters:
    -----------
    X_train : array-like or DataFrame
        The features for the training set.
    y_train : array-like or DataFrame
        The target variable for the training set.
    X_test : array-like or DataFrame
        The features for the test set.
    y_test : array-like or DataFrame
        The target variable for the test set.
    C_values : list of floats
        The list of C values to evaluate. C is the inverse regularization strength.
    penalty : str, optional
        The penalty type to use in the logistic regression model ('l1' or 'l2'). Default is 'l1'.
    solver: str, optional
        Algorithm to use in the optimization problem. Default is ‘liblinear’.
    random_state : int, optional
        The number used to initialize a pseudorandom number generator, which is used for reproducibility of the results.

    Returns:
    --------
    train_accuracies : list of floats
        The train accuracies for the different C values.
    test_accuracies : list of floats
        The test accuracies for the different C values.
    
    Notes:
    ------
    - This function uses a log scale for the C values on the x-axis of the plot.

    Example:
    --------
    >>> C_values = [0.001, 0.01, 0.1, 1, 10]
    >>> grid_log_c(X_train, y_train, X_test, y_test, C_values, 32)

    """
    train_accuracies = []
    test_accuracies = []

    # Implement a model
    for c in C_values:
        lr = LogisticRegression(C=c, solver=solver, penalty=penalty, random_state=random_state)
        lr.fit(X_train, y_train)

        train_accuracy = lr.score(X_train, y_train)
        test_accuracy = lr.score(X_test, y_test)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # Draw a plot
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(C_values, test_accuracies, label='Test Accuracy', marker='^')
    plt.xscale('log')
    plt.xlabel('C Value (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy for Different C Values')
    plt.legend()
    plt.show()

    return train_accuracies, test_accuracies

def feature_importance_logreg(classifier, X_train, y_train, cut_off=0.05):
    
    """
    Plots feature importances for a logistic regression model using L1 regularisation.

    Parameters:
    ----------
    classifier : sklearn.linear_model object
        The logistic regression model. 
    X_train : array-like or DataFrame
        The features for the training set.
    y_train : array-like or DataFrame
        The target variable for the training set. 
    cut_off : float, optional
        The cut-off value to consider for feature importance.

    Returns:
    -------
    feature_df : pandas DataFrame
        A DataFrame containing the features sorted by their importance.
        
    omitted_df : pandas DataFrame
        A DataFrame containing the features that were omitted based on the cut-off value.
    """
    # Create the model
    classifier.fit(X_train, y_train)
    
    # Get feature importances
    feature_importance_values = np.abs(classifier.coef_)
    
    # Sort feature importances
    indices = np.argsort(feature_importance_values[0])[::-1]
    
    sorted_feature_names = [X_train.columns[i] for i in indices]
    sorted_importance_values = [feature_importance_values[0][i] for i in indices]
    
    # Apply cutoff if specified
    if cut_off is not None:
        omitted_feature_names = [name for i, name in enumerate(sorted_feature_names) if sorted_importance_values[i] < cut_off]
        omitted_importance_values = [value for value in sorted_importance_values if value < cut_off]
        
        sorted_feature_names = [name for name in sorted_feature_names if name not in omitted_feature_names]
        sorted_importance_values = [value for value in sorted_importance_values if value >= cut_off]
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(sorted_importance_values)), sorted_importance_values, align="center")
    plt.xticks(range(len(sorted_importance_values)), sorted_feature_names, rotation=90)  # Add rotation for better visibility
    plt.xlabel("Feature Name")
    plt.ylabel("Absolute Coefficient Value")
    plt.show()
    
    # Create a DataFrame for sorted feature importances
    feature_df = pd.DataFrame({
        'Feature': sorted_feature_names,
        'Importance': sorted_importance_values
    })
    
    # Create a DataFrame for omitted feature importances
    omitted_df = pd.DataFrame({
        'Omitted Feature': omitted_feature_names,
        'Omitted Importance': omitted_importance_values
    })
    
    return feature_df, omitted_df

def plot_omitted_accuracy(classifier, X_train, y_train, X_test, y_test, high_pvalues, low_imp_df, SEED=42):
    """
    Plots the train and test accuracy of a Logistic Regression model when omitting certain variables.
    Performs this operation with and without replacement.

    Parameters:
    -----------
    classifier : sklearn.linear_model object
        The logistic regression model. 
    X_train, y_train : Training data and labels
    X_test, y_test : Test data and labels
    high_pvalues : list of features to be omitted based on high p-values
    low_imp_df : DataFrame containing features to be omitted based on low importance
    SEED : random state seed for reproducibility

    Returns:
    --------
    acc_df : DataFrame containing train and test accuracies

    """
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))

    # Omitting variables with replacement
    to_delete = set(high_pvalues + low_imp_df['Omitted Feature'].to_list())
    acc_dict = {}
    for var in to_delete:
        X_train_fe = X_train.copy()
        X_train_fe.drop(columns=var, inplace=True)
        classifier.fit(X_train_fe, y_train)
        train_score = classifier.score(X_train_fe, y_train)
        y_pred_fe = classifier.predict(X_test[X_train_fe.columns])
        test_score = accuracy_score(y_pred_fe, y_test)
        acc_dict[var] = [train_score, test_score]
    acc_df = pd.DataFrame(acc_dict).transpose()
    acc_df.columns = ['train_acc', 'test_acc']
    
    # Creating a lineplot 0
    acc_df.plot(ax=axs[0], title='With Replacement')
    axs[0].set_xticks(np.arange(len(acc_df.index)))
    axs[0].set_xticklabels(acc_df.index, rotation=90)

    # Omitting variables without replacement
    to_delete = list(acc_df.sort_values('test_acc', ascending=False).index)
    acc_dict = {}
    X_train_fe = X_train.copy()
    for var in to_delete:
        X_train_fe.drop(columns=var, inplace=True)
        classifier.fit(X_train_fe, y_train)
        train_score = classifier.score(X_train_fe, y_train)
        y_pred_fe = classifier.predict(X_test[X_train_fe.columns])
        test_score = accuracy_score(y_pred_fe, y_test)
        acc_dict[var] = [train_score, test_score]
    acc_df = pd.DataFrame(acc_dict).transpose()
    acc_df.columns = ['train_acc', 'test_acc']
    
    # Creating the list of omitted variables
    acc_delete = []
    # Finding the last occurrence of the max test accuracy
    max_test_acc = acc_df['test_acc'].max()
    last_max_nm = acc_df[::-1]['test_acc'].idxmax()
    last_max_index = acc_df.index.get_loc(last_max_nm)

    for var in acc_df.index:
        if var != last_max_nm:
            acc_delete.append(var)
        else: 
            break
        
    # Creating a lineplot 1
    acc_df.plot(ax=axs[1], title='Without Replacement')
    axs[1].set_xticks(np.arange(len(acc_df.index)))
    axs[1].set_xticklabels(acc_df.index, rotation=90)
    axs[1].axvline(last_max_index, color='red', linestyle='--', label='Max Test Accuracy')
    axs[1].legend()
    plt.show()
        
    return acc_df, acc_delete

def plot_roc(X, y, classifier, n_splits, custom_threshold=0.5):
    """
    Plots the Receiver Operating Characteristic (ROC) curve using Stratified K-Fold 
    cross-validation and highlights a given custom threshold on the curve.

    Parameters:
    -----------
    X : pandas DataFrame
        The feature matrix.
    y : pandas Series
        The target vector.
    classifier : scikit-learn classifier
        A classifier that has a `fit` and `predict_proba` method.
    n_splits : int
        The number of folds for Stratified K-Fold cross-validation.
    custom_threshold : float, optional (default=0.8)
        The custom probability threshold to highlight on the ROC curve.
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # Clone the classifier so each fold starts with a fresh model
        clf_fold = clone(classifier)
        clf_fold.fit(X.iloc[train_idx], y.iloc[train_idx])

        # Predict probabilities for the positive class
        y_prob = clf_fold.predict_proba(X.iloc[test_idx])[:, 1]

        # 1) Compute standard ROC for all thresholds
        fpr, tpr, thresholds = roc_curve(y.iloc[test_idx], y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 2) Interpolate TPR for "mean_fpr" to average across folds
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        
        # Plot each fold’s ROC
        ax.plot(fpr, tpr, lw=1, alpha=0.3, 
                label=f'ROC fold {i} (AUC = {roc_auc:.2f})')

        # 3) Find the point on ROC closest to custom_threshold
        #    This means: find the index in `thresholds` where threshold is ~ custom_threshold
        #    Then we can mark that point on the curve.
        #    Note: thresholds are sorted descending from 1 to 0
        idx = np.argmin(np.abs(thresholds - custom_threshold))
        
        # Mark the point (fpr[idx], tpr[idx]) on the current fold’s ROC
        ax.scatter(fpr[idx], tpr[idx], color='black', s=30, 
                   label=(f"Threshold={custom_threshold:.2f}, fold={i}")
                   if i == 0 else None,  # So the legend entry only appears once
                   zorder=5)

    # Plot the "chance" line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Chance')

    # Compute mean and std of TPR for the "mean_fpr"
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    # +/- 1 std. deviation
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(
        xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        xlabel='False Positive Rate', ylabel='True Positive Rate',
        title="Receiver Operating Characteristic (Custom Threshold Marked)"
    )
    ax.legend(loc="lower right")
    plt.show()

def deviance_residuals(y_test, y_pred_proba):

    #Compute deviance residuals
    y_log = np.log(y_pred_proba)
    nlog = np.log(1 - y_pred_proba)
    dev_res = np.sign(y_test - y_pred_proba) * np.sqrt(-2 * (y_test * y_log + (1 - y_test) * nlog))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_proba, dev_res, color='blue', alpha=0.2, label='Deviance Residuals')
    plt.axhline(0, color='red', linestyle='--', linewidth=1.2, label='Zero Line')
    plt.xlabel('Predicted Probabilities')
    plt.ylabel('Deviance Residuals')
    plt.title('Deviance Residuals vs. Predicted Probabilities')
    plt.legend()
    plt.show()
    
    return dev_res

def best_clf_tshld(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    thresholds = np.clip(thresholds, 0, 1)
    
    # Compute F1 scores for different thresholds
    f1_scores = [
        f1_score(y_test, (y_pred_proba > t).astype(int))
        for t in thresholds
    ]
    
    # Identify the threshold with the maximum F1 score
    best_threshold_f1 = thresholds[np.argmax(f1_scores)]
    
    # Plotting the F1 score for each threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.scatter(best_threshold_f1, np.max(f1_scores), color='red', label='Best threshold (F1 max)')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score across different thresholds')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print('Best threshold by F1-score: ', best_threshold_f1)
    return best_threshold_f1

def plot_lift_curve(y_test, y_probs):
    # 5. Sort instances by predicted probability (descending)
    data_test = pd.DataFrame({'y_true': y_test, 'y_prob': y_probs})
    data_test.sort_values(by='y_prob', ascending=False, inplace=True)
    
    # 6. Calculate cumulative gains
    data_test['cum_positives'] = data_test['y_true'].cumsum()
    total_positives = data_test['y_true'].sum()
    
    # 7. Compute % of samples
    data_test['pct_samples'] = np.arange(1, len(data_test) + 1) / len(data_test)
    
    # 8. Compute cumulative capture rate of positives
    data_test['capture_rate'] = data_test['cum_positives'] / total_positives
    
    # 9. Plot the Lift Curve (Cumulative Gains Curve)
    plt.figure(figsize=(8,6))
    
    # Plot model's capture rate
    plt.plot(data_test['pct_samples'], data_test['capture_rate'], label='Model')
    
    # Plot baseline (random) line
    plt.plot([0, 1], [0, 1], '--', color='red', label='Random')
    
    plt.title('Lift Curve (Cumulative Gains Curve)')
    plt.xlabel('Percentage of Samples')
    plt.ylabel('Cumulative Capture of Positives')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()