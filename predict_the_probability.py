"""
Dimon Yin
dimonyin@163.com
City University of Hong Kong

三态电子商务有限公司面试题
"""
import os
import sys
import numpy as np
import pandas as pd

from sklearn import metrics
import imblearn.over_sampling
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier

# Print parameters
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)

# Check if directories are existed
if not os.path.isdir("generated_Data"):
    os.mkdir("generated_Data")
if not os.path.isdir("result_data"):
    os.mkdir("result_data")
if not os.path.isdir("result_data/BEFORE SMOTE"):
    os.mkdir("result_data/BEFORE SMOTE")
if not os.path.isdir("result_data/AFTER SMOTE"):
    os.mkdir("result_data/AFTER SMOTE")


def normalize_data(x, variables):
    """
    Normalize data
    :param x: data
    :param variables: the columns you want to normalize
    :return: normalized data
    """
    return x[variables].apply(lambda num: (num - num.min()) / (num.max() - num.min()))


def apply_smote(x, y, random_state):
    # Resample
    smote = imblearn.over_sampling.SMOTE(random_state=random_state)
    return smote.fit_sample(x, y.ravel())


def write_to_file(file_name, path, result):
    """
    --------------------------------------------
    A function that writes result to a txt file.
    --------------------------------------------
    :param file_name: str
    :param path: str
    :param result: str
    """
    with open(path + "/" + file_name, "w+") as file:
        file.write(result)


def get_best_config(search_model, model, parameters, criterion, numeric_variables, x, y, random_state=None,
                    do_smote=False, do_normalize=True, n_jobs=-1, n_iter=100):
    """
    ---------------------------------------------------------------------------------------------------
    This function is used to get the best configuration for a model by using the grid search algorithm.
    ---------------------------------------------------------------------------------------------------
    :param search_model: str .... GridSearchCV or RandomizedSearchCV
    :param model: the created model object
    :param parameters: the parameters need to be tuned
    :param criterion: the criteria used for model selection
    :param numeric_variables:
    :param do_smote: true or false... Is SMOTE algorithm applied?
    :param x: all the data points of x
    :param y: all the data points of y
    :param do_normalize: normalize data?
    :param random_state: random seed
    :param n_jobs: CPUs used
    :param n_iter: for RandomizedSearchCV
    :return: dic of parameters
    """
    # Apply SMOTE algorithm for data imbalance?
    if do_smote:
        x, y = apply_smote(x=x, y=y, random_state=random_state)

    # Normalize the columns by using the min-max method
    if do_normalize:
        x[numeric_variables] = normalize_data(x=x, variables=numeric_variables)

    k = 5  # k-fold

    # Do the normal search
    if search_model == "gs":
        grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring=criterion, cv=k, n_jobs=n_jobs,
                                   verbose=2)

        # Fit all the data
        grid_search.fit(x, y)
        print(grid_search.best_estimator_)  # Debug

        # Return the best parameters
        return grid_search.best_params_

    # Do the random search
    elif search_model == "rs":
        random_search = RandomizedSearchCV(estimator=model, param_distributions=parameters, scoring=criterion,
                                           n_iter=n_iter, cv=k, verbose=2, random_state=random_state, n_jobs=n_jobs)

        # Fit all the data
        random_search.fit(x, y)
        print(random_search.best_estimator_)  # Debug

        # Return the best parameters
        return random_search.best_params_


def get_average_score(model, scoring, x, y, k, n_jobs):
    """
    -----------------------------------------------------------
    A function that evaluates a model by using K-fold validation
    -----------------------------------------------------------
    :param model: initialized model
    :param scoring: criteria
    :param x: all X variables
    :param y: target variables
    :param k: k-fold
    :param n_jobs: CPUs used
    :return int:score
    """
    # A list of scores
    all_scores = cross_val_score(estimator=model, scoring=scoring, X=x, y=y, cv=k, n_jobs=n_jobs)
    average_score = all_scores.mean() * 100  # Average

    # Return the average score
    return average_score


def evaluate(algorithm, model, best_parameters, search_model, numeric_variables, x, y, random_state=None,
             do_normalize=True, do_smote=False, k=10, n_jobs=-1):
    """
    --------------------------------------------------------------
    This is an evaluation function for machine learning algorithms.
    --------------------------------------------------------------
    :param algorithm: str
    :param model: the built model
    :param best_parameters: dic of best parameters
    :param search_model: str .... GridSearchCV or RandomizedSearchCV
    :param numeric_variables: list contains all numeric variables of x
    :param do_smote: true or false... Is SMOTE algorithm applied?
    :param x: true values of all x
    :param y; true values of all y
    :param k: int, k-fold
    :param do_normalize: do normalize for data?
    :param random_state: random seed
    :param n_jobs: CPUs used
    :return: str/result
    """
    if do_smote:
        # Apply SMOTE algorithm for data imbalance?
        x, y = apply_smote(x=x, y=y, random_state=random_state)

    if do_normalize:
        # Normalize the columns by using the min-max method
        x[numeric_variables] = normalize_data(x=x, variables=numeric_variables)

    # Get evaluation criteria ------------------------------------Cross_validation approach
    average_accuracy = get_average_score(model=model, scoring="accuracy", x=x, y=y, k=k, n_jobs=n_jobs)
    average_roc_auc_score = get_average_score(model=model, scoring="roc_auc", x=x, y=y, k=k, n_jobs=n_jobs)
    average_recall_score = get_average_score(model=model, scoring="recall", x=x, y=y, k=k, n_jobs=n_jobs)
    average_precision_score = get_average_score(model=model, scoring="precision", x=x, y=y, k=k, n_jobs=n_jobs)
    average_f1_score = get_average_score(model=model, scoring="f1", x=x, y=y, k=k, n_jobs=n_jobs)

    # Get evaluation criteria ------------------------------------Train_test split approach
    # Split x and y into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=random_state)
    # This is the best model
    model.fit(x_train, y_train)  # Train using normal split approach
    y_pred = model.predict(x_test)  # Test

    accuracy_score = metrics.accuracy_score(y_test, y_pred) * 100
    roc_auc_score = metrics.roc_auc_score(y_test, y_pred) * 100
    recall_score = metrics.recall_score(y_test, y_pred) * 100
    precision_score = metrics.precision_score(y_test, y_pred) * 100
    f1_score = metrics.f1_score(y_test, y_pred) * 100

    # The result
    result = "---------------------------------------------------------------\n" +\
             "Algorithm: " + algorithm + "\n" + \
             "Is data normalized? " + str(do_normalize) + "\n" + \
             "Is SMOTE algorithm applied? " + str(do_smote) + "\n" + \
             "Search Model: " + search_model + "\n"\
             "Best Parameters: " + str(best_parameters) + "\n"\
             "___________________Cross_validation approach___________________\n" +\
             "Average test accuracy: %.2f%%" % average_accuracy + "\n"\
             "Average ROC AUC score: %.2f%%" % average_roc_auc_score + "\n"\
             "Average recall score: %.2f%%" % average_recall_score + "\n" \
             "Average precision score: %.2f%%" % average_precision_score + "\n"\
             "Average f1 score: %.2f%%" % average_f1_score + "\n" \
             "___________________Train_test_split approach___________________\n" +\
             "Test accuracy: %.2f%%" % accuracy_score + "\n"\
             "ROC AUC score: %.2f%%" % roc_auc_score + "\n"\
             "Confusion matrix:\n" +\
             str(confusion_matrix(y_test, y_pred)) + "\n"\
             "Recall score: %.2f%%" % recall_score + "\n"\
             "Precision score: %.2f%%" % precision_score + "\n"\
             "F1 score: %.2f%%" % f1_score + "\n"\
             "____________________________End________________________________"

    # Return it
    return result


# Pre-defined parameters
random_seed = 2020  # By defining this, the results can be reproduced
class_weight = "balanced"
do_SMOTE = True
do_NORMALIZE = True

# Load dataset
data = pd.read_csv("provided_data/case2_training.csv")

# In order to check missing values - no missing values
print(data.isnull().sum())
print(data.describe())

# Check if its a balanced dataset
print(data["Accept"].value_counts())  # Well its imbalanced

# Target variable
target = "Accept"

# Numeric variables
numeric_features = ["Date", "Beds", "Review", "Price"]  # The columns we want to normalize

# Get X and Y
X = data.drop(["ID", target], axis=1)
Y = data[target]
print("----------------------------------------------------------")
print("X before one-hot encoding")
print(X.columns)  # debug

# One-hot encoding
# Get dummy variables
dummies_regions = pd.get_dummies(X["Region"]).rename(columns=lambda name: "Region_" + str(name))
dummies_weekdays = pd.get_dummies(X["Weekday"]).rename(columns=lambda name: "Weekday_" + str(name))

# bring the dummies back into the original dataset
X = pd.concat([X.drop(["Region", "Weekday"], axis=1), dummies_regions, dummies_weekdays], axis=1)
print("----------------------------------------------------------")
print("X after one-hot encoding")
print(X.columns)  # debug


########################################################################################################################
# Logistic Regression
# ----------------------------------------------------------------------------------------------------------------------
# Parameters of LR model  penalty/solver/C/l1_ratio
C = np.logspace(-4, 4, 20)  # Inverse of regularization strength
l1_ratio = np.linspace(0, 1, 10, endpoint=True)  # The Elastic-Net mixing parameter

# Dic them
parameters_LR = [
    {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': C},
    {'penalty': ['l2'], 'solver': ['liblinear', 'newton-cg', 'sag', 'lbfgs', 'saga'], 'C': C},
    {'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': l1_ratio, 'C': C},
    {'penalty': ['none'], 'solver': ['newton-cg', 'sag', 'lbfgs', 'saga'], 'C': C}
]

# Create the LR model
logistic_regression = LogisticRegression(class_weight=class_weight, random_state=random_seed, n_jobs=-1)

# Get the best parameters
best_params_LR = get_best_config(search_model="gs", model=logistic_regression, parameters=parameters_LR,
                                 criterion="roc_auc", numeric_variables=numeric_features, do_smote=do_SMOTE,
                                 x=X, y=Y, do_normalize=do_NORMALIZE, random_state=random_seed)
print(best_params_LR)  # Debug

# Load the best parameters for LR
logistic_regression_best = LogisticRegression(**best_params_LR,
                                              class_weight=class_weight,
                                              random_state=random_seed,
                                              n_jobs=-1)
print(logistic_regression_best)  # Debug

# Evaluate
best_result_LR = evaluate(algorithm="Logistic Regression", model=logistic_regression_best,
                          best_parameters=best_params_LR, search_model="GridSearchCV",
                          numeric_variables=numeric_features, do_smote=do_SMOTE, x=X, y=Y,
                          do_normalize=do_NORMALIZE, random_state=random_seed)

# Print the result
print(best_result_LR)

# Finally, we record it to txt file
write_to_file("GridSearchCV_LR.txt", "result_data/AFTER SMOTE", best_result_LR)


########################################################################################################################
# Random Forest
# ----------------------------------------------------------------------------------------------------------------------
# Parameters of RF n_estimators/criterion/max_depths
# Number of trees in random forest
n_estimators_RF = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# Number of features to consider at every split
max_features_RF = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depths_RF = [int(x) for x in np.linspace(10, 110, num=11)]
max_depths_RF.append(None)

# Minimum number of samples required to split a node
min_samples_splits_RF = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leafs_RF = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap_RF = [True, False]


# Parameter space
parameters_RF = [
    {'n_estimators': n_estimators_RF,
     'max_features': max_features_RF,
     'max_depth': max_depths_RF,
     'min_samples_split': min_samples_splits_RF,
     'min_samples_leaf': min_samples_leafs_RF,
     'bootstrap': bootstrap_RF}
]

# Create the RF model
random_forest = RandomForestClassifier(class_weight=class_weight, random_state=random_seed, n_jobs=-1)

# Get the best parameters
best_params_RF = get_best_config(search_model="rs", model=random_forest,
                                 parameters=parameters_RF, criterion="roc_auc",
                                 numeric_variables=numeric_features, do_smote=do_SMOTE, x=X, y=Y,
                                 do_normalize=do_NORMALIZE, random_state=random_seed)

# Load the best parameters for RF
random_forest_best = RandomForestClassifier(**best_params_RF, class_weight=class_weight, random_state=random_seed,
                                            n_jobs=-1)

# Evaluate
best_result_RF = evaluate(algorithm="Random Forest", model=random_forest_best, best_parameters=best_params_RF,
                          search_model="RandomizedSearchCV", numeric_variables=numeric_features, do_smote=do_SMOTE, x=X,
                          y=Y, do_normalize=do_NORMALIZE, random_state=random_seed)

# Print the result
print(best_result_RF)

# Finally, we record it to txt file
write_to_file("RandomizedSearchCV_RF.txt", "result_data/AFTER SMOTE", best_result_RF)


########################################################################################################################
# Support Vector Machine
# ----------------------------------------------------------------------------------------------------------------------
# Parameters of SVM Kernel/gamma/C/degree
kernels = ['linear', 'sigmoid', 'rbf', "poly"]
gammas = [0.1, 0.2, 0.3, 'scale', 'auto']
C_svm = [0.1, 1, 3, 5, 7, 10]
degrees = list(range(0, 4))

# Parameter space
parameters_SVM = [
    {"C": C_svm, "kernel": ["linear"]},
    {"C": C_svm, "kernel": ["poly"], "degree": degrees, "gamma": gammas},
    {"C": C_svm, "kernel": ["sigmoid"], "gamma": gammas},
    {"C": C_svm, "kernel": ["rbf"], "gamma": gammas}
]

# Create the SVM model
svm_classifier = SVC(class_weight=class_weight, random_state=random_seed)

# Get the best parameters
best_params_SVM = get_best_config(search_model="rs", model=svm_classifier,
                                  parameters=parameters_SVM, criterion="roc_auc",
                                  numeric_variables=numeric_features, do_smote=do_SMOTE, x=X,
                                  y=Y, do_normalize=do_NORMALIZE, random_state=random_seed)

# Load the best parameters for SVM
svm_classifier_best = SVC(**best_params_SVM, class_weight=class_weight, random_state=random_seed)

# Evaluate
best_result_SVM = evaluate(algorithm="Support Vector Machine", model=svm_classifier_best,
                           best_parameters=best_params_SVM, search_model="RandomizedSearchCV",
                           numeric_variables=numeric_features, do_smote=do_SMOTE, x=X, y=Y,
                           do_normalize=do_NORMALIZE, random_state=random_seed)
# Print the result
print(best_result_SVM)

# Finally, we record it to txt file
write_to_file("RandomizedSearchCV_SVM.txt", "result_data/BEFORE SMOTE", best_result_SVM)


########################################################################################################################
# Gradient Boosting Classification Tree
# ----------------------------------------------------------------------------------------------------------------------
# Parameters of GBCT
# Learn rate
learn_rates_GBCT = np.linspace(0.01, 0.2, num=20)

# The number of trees in the forest
n_estimators_GBCT = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# Maximum number of levels in tree
max_depths_GBCT = [int(x) for x in np.linspace(1, 15, num=15)]
max_depths_GBCT.append(None)

# Minimum number of samples required to split a node
min_samples_splits_GBCT = np.linspace(0.1, 1.0, 10, endpoint=True)

# Minimum number of samples required at each leaf node
min_samples_leafs_GBCT = np.linspace(0.1, 0.5, 5, endpoint=True)

# Number of features to consider at every split
max_features_GBCT = list(range(1, X.shape[1]))

# Parameter space
parameters_GBCT = [
    {
        "learning_rate": learn_rates_GBCT,
        'n_estimators': n_estimators_GBCT,
        'max_features': max_features_GBCT,
        'max_depth': max_depths_GBCT,
        'min_samples_split': min_samples_splits_GBCT,
        'min_samples_leaf': min_samples_leafs_GBCT
    }
]

# Create the GBCT model
GB_classifier = GradientBoostingClassifier(random_state=random_seed)

# Get the best parameters
best_params_GBCT = get_best_config(search_model="rs", model=GB_classifier,
                                   parameters=parameters_GBCT, criterion="roc_auc",
                                   numeric_variables=numeric_features, do_smote=do_SMOTE, x=X, y=Y,
                                   do_normalize=do_NORMALIZE, random_state=random_seed)

# Load the best parameters for GBCT
GB_classifier_best = GradientBoostingClassifier(**best_params_GBCT, random_state=random_seed)

# Evaluate
best_result_GBCT = evaluate(algorithm="Gradient Boosting Classification Tree", model=GB_classifier_best,
                            best_parameters=best_params_GBCT, search_model="RandomizedSearchCV",
                            numeric_variables=numeric_features, do_smote=do_SMOTE, x=X, y=Y,
                            do_normalize=do_NORMALIZE, random_state=random_seed)

# Print the result
print(best_result_GBCT)

# Finally, we record it to txt file
write_to_file("RandomizedSearchCV_GBCT.txt", "result_data/AFTER SMOTE", best_result_GBCT)


########################################################################################################################
# xgboost
# ----------------------------------------------------------------------------------------------------------------------
# Parameter of xgb
# Learn_rate
etas_xgb = np.linspace(0.01, 0.2, num=20)

# Gamma
gammas_xgb = [i for i in range(5)]

# Min_child_weight
min_child_weights_xgb = [i for i in range(5)]

# Max_depth
max_depths_xgb = [i for i in range(3, 11)]

# n_estimators
n_estimators_xgb = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# subsample
subsamples_xgb = np.linspace(0.8, 1, 10)

# colsample_bytree
colsample_bytrees_xgb = np.linspace(0.3, 1, 10)

# Parameter space
parameters_xgb = [
    {
        "eta": etas_xgb,
        "n_estimators": n_estimators_xgb,
        "max_depth": max_depths_xgb,
        "subsample": subsamples_xgb,
        "colsample_bytree": colsample_bytrees_xgb,
        "gamma": gammas_xgb,
        "min_child_weight": min_child_weights_xgb,
    }
]

# Create the model
xgb_classifier = XGBClassifier(class_weight=class_weight, random_state=random_seed, n_jobs=-1)

# Get the best parameters
best_params_xgb = get_best_config(search_model="rs", model=xgb_classifier,
                                  parameters=parameters_xgb, criterion="roc_auc",
                                  numeric_variables=numeric_features, do_smote=do_SMOTE, x=X, y=Y,
                                  do_normalize=do_NORMALIZE, random_state=random_seed)

# Load the best parameters for xgb
xgb_classifier_best = XGBClassifier(**best_params_xgb,
                                    class_weight=class_weight,
                                    random_state=random_seed,
                                    n_jobs=-1)

# Evaluate
best_result_xgb = evaluate(algorithm="XGboost", model=xgb_classifier_best,
                           best_parameters=best_params_xgb, search_model="RandomizedSearchCV",
                           numeric_variables=numeric_features, do_smote=do_SMOTE, x=X, y=Y,
                           do_normalize=do_NORMALIZE, random_state=random_seed)

# Print the result
print(best_result_xgb)

# Finally, we record it to txt file
write_to_file("RandomizedSearchCV_xgb.txt", "result_data/AFTER SMOTE", best_result_xgb)


########################################################################################################################
# Load dataset
data_test = pd.read_csv("provided_data/case2_testing.csv")

# In order to check missing values - no missing values
print(data_test.isnull().sum())
print(data_test.describe())

# Get X and Y
result = data_test["ID"].to_frame()
X_test = data_test.drop(["ID"], axis=1)

# Normalize data
X[numeric_features] = normalize_data(X, numeric_features)
X_test[numeric_features] = normalize_data(X_test, numeric_features)

# One-hot encoding
# Get dummy variables
dummies_regions = pd.get_dummies(X_test["Region"]).rename(columns=lambda name: "Region_" + str(name))
dummies_weekdays = pd.get_dummies(X_test["Weekday"]).rename(columns=lambda name: "Weekday_" + str(name))

# bring the dummies back into the original dataset
X_test = pd.concat([X_test.drop(["Region", "Weekday"], axis=1), dummies_regions, dummies_weekdays], axis=1)

# Build the best model
selected_parameters = {'C': 0.08858667904100823, 'penalty': 'l1', 'solver': 'liblinear'}  # Best parameters

# LR
LR = LogisticRegression(**selected_parameters,
                        class_weight=class_weight,
                        random_state=random_seed,
                        n_jobs=-1)

# Fit the model
LR.fit(X, Y)

# Predict the result
result["possibility"] = LR.predict_proba(X_test)[:, 1]

# Output configured dataset
result.to_csv("generated_Data/final_result.csv", index=False)
