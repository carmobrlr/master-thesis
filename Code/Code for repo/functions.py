import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score, mean_absolute_error, r2_score, log_loss
from xgboost import XGBRegressor, XGBClassifier
from sklearn.naive_bayes import GaussianNB
import re
from sklearn.svm import SVC, SVR
from imblearn.pipeline import make_pipeline
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE

def load_and_prepare_data(train_file='train_data.xlsx',
                          test_file='test_data_corrected.xlsx'):
    # Load training data
    X_train = pd.read_excel(train_file, sheet_name='x')
    y_train = pd.read_excel(train_file, sheet_name='y')
    X_train['y'] = y_train

    # Load testing data
    X_test = pd.read_excel(test_file, sheet_name='x')
    y_test = pd.read_excel(test_file, sheet_name='y')
    X_test['y'] = y_test

    # Concatenate train and test data
    X = pd.concat([X_train, X_test], axis=0)

    # Drop NaNs and duplicates
    X = X.dropna().drop_duplicates().reset_index(drop=True)
    y = X['y']

    return X, y


def select_features_by_pearson(X, target_col='y', p_value_threshold=0.005):
    cols_selected = []
    num_vars = 0
    for col in X.columns:
        if col == target_col:
            continue
        corr_coeff, p_value = pearsonr(X[col], X[target_col])
        if p_value <= p_value_threshold:
            cols_selected.append(col)
            num_vars += 1
    print(f'Number of variables with p-value <= {p_value_threshold}: {num_vars}')
    return cols_selected


def select_features_by_anova(X, y, threshold=0.05, high_score_threshold=90):
    # Categorize the target
    y_categorized = np.where(y.ravel() >= high_score_threshold, 0, 1)

    X1 = X.drop(columns='y', errors='ignore')
    cols_selected = []
    num_vars = 0

    for col in X1.columns:
        feature = X1[col].values.reshape(-1, 1)
        F, p = f_classif(feature, y_categorized)
        if p <= threshold:
            cols_selected.append(col)
            num_vars += 1

    print(f'Number of variables with p-value <= {threshold}: {num_vars}')
    return cols_selected

def get_feature_sets():
    return {
        # From literature
        'FS1_RF': ['Average DO concentration dd0', 'Average DO concentration gradient d0', 'Average DO concentration gradient dd6',
                   'd0 Average pH Gradient', 'dd0 DO 2nd derivative/cell count', 'dd1 Cell Density', 'dd2 Average pH', 
                   'dd3 Average pH Gradient', 'dd5-dd7 Aggregate Size Gradient', 'dd7 Cell Density', 
                   'DO concentration/cell count dd7', 'DO gradient/cell count dd2', 'DO gradient/cell count dd7'],
        
        # Biclustering CCCB1 and variations
        'CCCB1': ['dd2 Cell Density', 'dd3 Cell Density ', 'dd5 Cell Density', 'dd7 Cell Density',
                  'Average DO concentration dd0', 'Average DO concentration dd1', 'Average DO concentration dd2', 
                  'Average DO concentration dd3', 'Average DO concentration dd4', 'DO gradient/cell count dd1',
                  'DO gradient/cell count dd2', 'DO gradient/cell count dd3', 'DO gradient/cell count dd5'],
        
        'CCCB2': ['dd0 Cell Density', 'dd1 Cell Density','dd2 Cell Density', 'dd3 Cell Density ', 'dd5 Cell Density',
                  'Average DO concentration dd0', 'Average DO concentration dd1', 'Average DO concentration dd2',
                  'DO gradient/cell count dd1', 'DO gradient/cell count dd2', 'DO gradient/cell count dd3',
                  'DO gradient/cell count dd5', 'dd3 Glucose Concentration', 'dd5 Glucose Concentration',
                  'dd7 Glucose Concentration'],
        
        # CCCB1 + pH
        'CCCB1_1': ['dd2 Cell Density', 'dd3 Cell Density ', 'dd5 Cell Density', 'dd7 Cell Density',
                    'Average DO concentration dd0', 'Average DO concentration dd1', 'Average DO concentration dd2', 
                    'Average DO concentration dd3', 'Average DO concentration dd4', 'DO gradient/cell count dd1',
                    'DO gradient/cell count dd2', 'DO gradient/cell count dd3', 'DO gradient/cell count dd5',
                    'dd2 Average pH', 'dd3 Average pH', 'dd4 Average pH', 'dd5 Average pH', 'dd6 Average pH', 'dd7 Average pH',
                    'dd0 Average pH Gradient','dd1 Average pH Gradient','dd2 Average pH Gradient','dd3 Average pH Gradient'],
        
        # CCCB2 + pH
        'CCCB2_1': ['dd0 Cell Density', 'dd1 Cell Density','dd2 Cell Density', 'dd3 Cell Density ', 
                    'dd5 Cell Density', 'Average DO concentration dd0', 'Average DO concentration dd1', 
                    'Average DO concentration dd2', 'DO gradient/cell count dd1', 'DO gradient/cell count dd2', 
                    'DO gradient/cell count dd3', 'DO gradient/cell count dd5', 'dd3 Glucose Concentration', 
                    'dd5 Glucose Concentration', 'dd7 Glucose Concentration', 'dd2 Average pH', 'dd3 Average pH', 
                    'dd4 Average pH', 'dd5 Average pH', 'dd6 Average pH', 'dd7 Average pH', 'd1 Average pH Gradient', 
                    'dd0 Average pH Gradient', 'dd1 Average pH Gradient', 'dd2 Average pH Gradient', 
                    'dd3 Average pH Gradient', 'dd4 Average pH Gradient'],
    }

def get_dataframe_with_selected_features(X, feature_set_name, extra_sets={}):
    all_sets = get_feature_sets()
    all_sets.update(extra_sets)

    if feature_set_name not in all_sets:
        raise ValueError(f"Feature set '{feature_set_name}' not found.")
    
    selected_cols = all_sets[feature_set_name]
    return X[selected_cols]


def clean_column_names(df):
    # Clean the column names (for XGBoost)
    df.columns = [re.sub(r'[^\w]', '_', col) for col in df.columns]
    return df

def Classifier_calculate_metrics_FINAL2(model_str: str, X: pd.DataFrame, y: pd.Series,
                                       model_params: dict, 
                                       cross_validation_technique: str, 
                                       n_splits: int = 10,
                                       random_state: int = 42):
    
    # Get number of features of X, C is the number of classes
    F = X.shape[1]
    C = 2

    # Initialize model based on the provided model string and parameters
    if model_str == 'RandomForestClassifier':
        model = RandomForestClassifier(random_state=random_state, **model_params)
    elif model_str == 'XGBoostClassifier':
        model = XGBClassifier(random_state=random_state, **model_params)
    elif model_str == 'GaussianNaiveBayesClassifier':
        model = GaussianNB()
    elif model_str == 'SMOTE+GaussinNaiveBayesClassifier':
        model = make_pipeline(
            SMOTE(random_state=42, k_neighbors=5),
            GaussianNB()
        )
    elif model_str == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(random_state=random_state, **model_params)
    elif model_str == 'Scaler+SVMClassifier':
        svm_C = model_params['svc__C']
        svm_gamma = model_params['svc__gamma']
        svm_kernel = model_params['svc__kernel']
        model = make_pipeline(
            StandardScaler(),
            SVC(random_state=random_state, probability=True, C = svm_C, kernel=svm_kernel, gamma=svm_gamma)
        )
    else:
        raise NotImplementedError(f"Model '{model_str}' is not yet implemented.")


    # LOOCV (Leave-One-Out Cross Validation)
    if cross_validation_technique == 'loocv':
        loo = LeaveOneOut()
        
        all_y_test = []
        all_y_pred = []

        log_likelihood_total = 0
        AIC_scores, BIC_scores = [], []

        for train_index, test_index in loo.split(X):  
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            log_likelihood = -log_loss(y_test, y_pred_proba, normalize=False, labels=[0,1])
            log_likelihood_total += log_likelihood
            
            # Compute number of parameters for each model
            if model_str == 'DecisionTreeClassifier':
                k = model.get_n_leaves()
            elif model_str == 'RandomForestClassifier':
                k = sum(tree.get_n_leaves() for tree in model.estimators_)
            elif model_str == 'XGBoostClassifier':
                k = sum(tree.count('\n') for tree in model.get_booster().get_dump())
            elif model_str == 'GaussianNaiveBayesClassifier' or model_str == 'SMOTE+GaussinNaiveBayesClassifier':
                k = C * 2 * F + (C - 1)
            elif model_str == 'Scaler+SVMClassifier':
                n_support_vectors = len(model[1].support_)
                if model[1].kernel == 'rbf':
                    k = n_support_vectors + 1 # gamma
                elif model[1].kernel == 'linear':
                    k = n_support_vectors + F

            n = 1
            AIC_score = 2 * k - 2 * log_likelihood
            AIC_scores.append(AIC_score)
            BIC_score = k * np.log(n) - 2 * log_likelihood
            BIC_scores.append(BIC_score)  
            
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)

        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)

        accuracy = accuracy_score(all_y_test, all_y_pred)
        precision = precision_score(all_y_test, all_y_pred)
        recall = recall_score(all_y_test, all_y_pred)
        f1 = f1_score(all_y_test, all_y_pred)
        MCC = matthews_corrcoef(all_y_test, all_y_pred)

        mean_AIC = np.mean(AIC_scores)
        mean_BIC = np.mean(BIC_scores)
        std_AIC = np.std(AIC_scores)
        std_BIC = np.std(BIC_scores)

        print(f'Accuracy: {accuracy:.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Recall: {recall:.3f}')
        print(f'f1 score: {f1:.3f}')
        print(f'MCC score: {MCC:.3f}')
        print(f'Mean AIC score: {mean_AIC:.3f} ± {std_AIC:.3f}')
        print(f'Mean BIC score: {mean_BIC:.3f} ± {std_BIC:.3f}')

        return (accuracy, precision, recall, f1, MCC, mean_AIC, mean_BIC), (std_AIC, std_BIC)

    # Stratified K-Fold Cross Validation (skfcv)
    elif cross_validation_technique == 'skfcv':
        accuracies, precisions, recalls, f1_scores, MCC_scores, AIC_scores, BIC_scores = [], [], [], [], [], [], []

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            n = len(y_test)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            log_likelihood = -log_loss(y_test, y_pred_proba, normalize=False)

            # Compute number of parameters for each model
            if model_str == 'DecisionTreeClassifier':
                k = model.get_n_leaves()
            elif model_str == 'RandomForestClassifier':
                k = sum(tree.get_n_leaves() for tree in model.estimators_)
            elif model_str == 'XGBoostClassifier':
                k = sum(tree.count('\n') for tree in model.get_booster().get_dump())
            elif model_str == 'GaussianNaiveBayesClassifier' or model_str == 'SMOTE+GaussinNaiveBayesClassifier':
                k = C * 2 * F + (C - 1)
            elif model_str == 'Scaler+SVMClassifier':
                n_support_vectors = len(model[1].support_)
                if model[1].kernel == 'rbf':
                    k = n_support_vectors + 1 # gamma
                elif model[1].kernel == 'linear':
                    k = n_support_vectors + F
            
            accuracies.append(accuracy_score(y_test, y_pred))
            precision = precision_score(y_test, y_pred, zero_division=0)
            if np.sum(y_pred) == 0:
                precisions.append(np.nan)
            else:
                precisions.append(precision)

            recall = recall_score(y_test, y_pred, zero_division=0)
            if np.sum(y_test) == 0:
                recalls.append(np.nan)
            else:
                recalls.append(recall)

            f1 = f1_score(y_test, y_pred, zero_division=0)
            if np.sum(y_pred) == 0 or np.sum(y_test) == 0:
                f1_scores.append(np.nan)
            else:
                f1_scores.append(f1)

            MCC_scores.append(matthews_corrcoef(y_test, y_pred))

            AIC_score = 2 * k - 2 * log_likelihood
            AIC_scores.append(AIC_score)
            BIC_score = k * np.log(n) - 2 * log_likelihood
            BIC_scores.append(BIC_score)

        # Calculate mean and standard deviation
        mean_accuracy = np.mean(accuracies)
        mean_precision = np.nanmean(precisions)
        mean_recall = np.nanmean(recalls)
        mean_f1 = np.nanmean(f1_scores)
        mean_MCC = np.mean(MCC_scores)
        mean_AIC = np.mean(AIC_scores)
        mean_BIC = np.mean(BIC_scores)

        std_accuracy = np.std(accuracies)
        std_precision = np.nanstd(precisions)
        std_recall = np.nanstd(recalls)
        std_f1 = np.nanstd(f1_scores)
        std_MCC = np.std(MCC_scores)
        std_AIC = np.std(AIC_scores)
        std_BIC = np.std(BIC_scores)

        print(f'Mean accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}')
        print(f'Mean precision: {mean_precision:.3f} ± {std_precision:.3f}')
        print(f'Mean recall: {mean_recall:.3f} ± {std_recall:.3f}')
        print(f'Mean f1 score: {mean_f1:.3f} ± {std_f1:.3f}')
        print(f'Mean MCC score: {mean_MCC:.3f} ± {std_MCC:.3f}')
        print(f'Mean AIC score: {mean_AIC:.3f} ± {std_AIC:.3f}')
        print(f'Mean BIC score: {mean_BIC:.3f} ± {std_BIC:.3f}')

        return (mean_accuracy, mean_precision, mean_recall, mean_f1, mean_MCC, mean_AIC, mean_BIC), \
               (std_accuracy, std_precision, std_recall, std_f1, std_MCC, std_AIC, std_BIC)
    
def Regressor_calculate_metrics(model_str: str, X: pd.DataFrame, y: pd.Series,
                                model_params: dict, 
                                cross_validation_technique: str, 
                                n_splits: int = 10,
                                random_state: int = 42):
    
    # Initialize model based on the provided model string and parameters
    if model_str == 'RandomForestRegressor':
        model = RandomForestRegressor(random_state=random_state, **model_params)
    elif model_str == 'XGBoostRegressor':
        model = XGBRegressor(random_state=random_state, **model_params)
    elif model_str == 'LinearRegression':
        model = LinearRegression()
    elif model_str == 'ROS+Scaler+SVMRegressor':
        model = make_pipeline(
            StandardScaler(),
            SVR(random_state=random_state, **model_params)
        )
    elif model_str == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor(random_state=random_state, **model_params)
    else:
        raise NotImplementedError(f"Model '{model_str}' is not yet implemented.")

    # Leave-One-Out Cross Validation (loocv)
    if cross_validation_technique == 'loocv':
        loo = LeaveOneOut()
        
        all_y_test = []
        all_y_pred = []

        for train_index, test_index in loo.split(X):  
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)

        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)

        MAE_metric = mean_absolute_error(all_y_test, all_y_pred)
        MSE_metric = mean_squared_error(all_y_test, all_y_pred)
        RMSE_metric = np.sqrt(MSE_metric)
        r2_metric = r2_score(all_y_test, all_y_pred)

        print(f'MAE: {MAE_metric:.4f}')
        print(f'MSE: {MSE_metric:.4f}')
        print(f'RMSE: {RMSE_metric:.4f}')
        print(f'r2 score: {r2_metric:.4f}')

        return (MAE_metric, MSE_metric, RMSE_metric, r2_metric)

    # K-Fold Cross Validation (kfcv)
    elif cross_validation_technique == 'kfcv':
        MAEs, MSEs, RMSEs, r2_metrics = [], [], [], []

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            MAEs.append(mean_absolute_error(y_test, y_pred))
            MSEs.append(mean_squared_error(y_test, y_pred))
            RMSEs.append(np.sqrt())
            r2_metrics.append(r2_score(y_test, y_pred))

        # Calculate mean and standard deviation
        mean_MAEs = np.mean(MAEs)
        mean_MSEs = np.nanmean(MSEs)
        mean_RMSEs = np.nanmean(RMSEs)
        mean_r2metrics = np.nanmean(r2_metrics)

        std_MAEs = np.std(MAEs)
        std_MSEs = np.nanstd(MSEs)
        std_RMSEs = np.nanstd(RMSEs)
        std_r2metrics = np.nanstd(r2_metrics)

        print(f'Mean accuracy: {mean_MAEs:.4f} ± {std_MAEs:.4f}')
        print(f'Mean precision: {mean_MSEs:.4f} ± {std_MSEs:.4f}')
        print(f'Mean recall: {mean_RMSEs:.4f} ± {std_RMSEs:.4f}')
        print(f'Mean f1 score: {mean_r2metrics:.4f} ± {std_r2metrics:.4f}')

        return (mean_MAEs, mean_MSEs, mean_RMSEs, mean_r2metrics), \
               (std_MAEs, std_MSEs, std_RMSEs, std_r2metrics)