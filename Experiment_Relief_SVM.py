import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from skrebate import ReliefF
from sklearn import metrics
from sklearn.metrics import auc, f1_score, accuracy_score, precision_score
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("DATA.csv")
print(data)

# Selecting target
target = "target"

X = data.iloc[:, data.columns != target].values
y = data.iloc[:, data.columns == target].values

# X, y as dataframe not numpy array
X_df = data.iloc[:, data.columns != target]
y_df = data.iloc[:, data.columns == target]

# Select categorical and Continuous features
numeric_features = list(X_df._get_numeric_data().columns)
categorical_features = list(set(X_df.columns) - set(X_df._get_numeric_data().columns))

# Create pipeline for both Continuous and Categorical Variables
cnts_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categ_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Apply Column Transformers

preprocess_pipeline = ColumnTransformer([
    ('continuous', cnts_pipeline, numeric_features),
    ('cat', categ_pipeline, categorical_features)
    ])

# Function to get all ct columns
# https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api
def get_feature_out(estimator, feature_in):
    if hasattr(estimator, 'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}' \
                    for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in

def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != 'remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == 'passthrough':
            output_features.extend(ct._feature_names_in[features])

    return output_features




# Create the classifier object
selected_classifier = "SVC"
classifier = SVC(kernel="linear", probability=True, class_weight="balanced")
selector = ReliefF(n_features_to_select=10, n_neighbors=100)

# A pipeline chains two algorithms together so that the training process for both can be done in a single step and data is passed automatically from one to the other
pipeline = Pipeline([("preprocessor", preprocess_pipeline), ("ReliefF", selector), ("classifier", classifier)])

# Dictionary that contains the values for the parameter sweep
param_grid = dict(ReliefF__n_features_to_select=[10, 20, 30, 40, 50], classifier__C=[0.001, 0.01, 0.1, 1, 10, 100, 1000], classifier__gamma=[1, 0.1, 0.001, 0.0001])
#param_grid = dict(ReliefF__n_features_to_select=[2,3,4,5], classifier__max_depth=[2, 3, 4, 10], classifier__n_estimators=[100, 200, 500])

scores = []
accuracy_scores = []
f1_scores = []
precision_scores = []

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 1

# Initialise the 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X_df):
    # Generate the training and test partitions of X and Y for each iteration of CV

    X_train, X_test = X_df.iloc[train_index], X_df.iloc[test_index]
    y_train, y_test = y_df.iloc[train_index], y_df.iloc[test_index]

    # Increasing the value of the verbose parameter will give more messages of the internal grid search process
    # Increasing n_jobs will tell it to use multiple cores to parallelize the computation
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring="roc_auc", verbose=0, n_jobs=4)
    print(f"Now tuning {selected_classifier}. Let's play the waiting game.")
    print("KFold " + str(i))
    grid_search.fit(X_train, np.ravel(y_train))

    # Printing the values of the parameters chosen by grid search
    estimator = grid_search.best_estimator_

    all_features = [i for i in get_ct_feature_names(estimator.named_steps["preprocessor"])]
    num_feat = estimator.named_steps['ReliefF'].n_features_to_select
    selected_features = [g for g, x in zip(all_features, estimator.named_steps['ReliefF'].top_features_[:num_feat]) if
                         x]

    print("Number of selected features {0}".format(num_feat))
    print("Selected features {0}".format(selected_features))
    print("Selected features {0}".format(estimator.named_steps['ReliefF'].top_features_[:num_feat]))

    # Additional data to csv
    other_data = pd.DataFrame(selected_features, columns=["Selected Features"])
    other_data["No. Selected Features"] = len(selected_features)
    other_data.to_csv(path_or_buf="PATH/" + str(i) + "Relief-SVC_AdditionalData.csv")

    # Predicting the test data with the optimised models
    predictions = estimator.predict(X_test)

    # binarizing predictions
    le = preprocessing.LabelEncoder()
    predictions = le.fit_transform(predictions)

    score = metrics.roc_auc_score(y_test, predictions)
    print("AUC score for this test set: {0}".format(score))
    scores.append(score)


    # testing implementing curve png
    def get_scores():
        score = metrics.roc_auc_score(y_test, predictions)
        return score

    # Feature Importance Graph
    feature_importance = pd.DataFrame(selected_features, columns=["Selected Feature"])
    feature_importance["Feature Importance"] = estimator.named_steps['classifier'].coef_[0]

    # Sort by feature importance
    feature_importance1 = feature_importance.sort_values(by="Feature Importance", ascending=False)

    def feature_importance_plot():
        sns.set(font_scale=1.00)
        sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
                       "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
                       'ytick.color': '0.4'})

        # Set figure size and create barplot
        plt.figure(1)
        f, ax = plt.subplots(figsize=(12, 9))
        sns.barplot(x="Feature Importance", y="Selected Feature",
                    palette=reversed(sns.color_palette('YlOrRd', 15)), data=feature_importance1)

        # Generate a bolded horizontal line at y = 0
        ax.axvline(x=0, color='black', linewidth=4, alpha=.7)

        # Turn frame off
        ax.set_frame_on(False)

        # Tight layout
        plt.tight_layout()

        # Save Fig
        filename = "PATH/" + str(get_scores()) + "Feature_Importance_Relief-SVC" + ".png"
        plt.savefig(filename, dpi=1080)
        plt.clf()


    feature_importance_plot()

    # Report the overall score
    print("Overall AUC score: {0}".format(np.average(scores)))
    print()

    # Other scores
    y_true = le.fit_transform(np.ravel(y_test))

    # Accuracy
    accuracy = accuracy_score(y_true, predictions)
    print("Accuracy Score of this test: {0}".format(accuracy))
    accuracy_scores.append(accuracy)

    # F1
    f1 = f1_score(y_true, predictions)
    print("F1 Score of this test: {0}".format(f1))
    f1_scores.append(f1)

    # Precision
    prec = precision_score(y_true, predictions)
    print("Precision Score of this test: {0}".format(prec))
    precision_scores.append(prec)
    print()

    # Overall Scores
    # Report the overall AUC score
    print("Overall AUC score: {0}".format(np.average(scores)))

    # Overall Accuracy Score
    print("Overall Accuracy score: {0}".format(np.average(accuracy_scores)))

    # Overall F1
    print("Overall F1 score: {0}".format(np.average(f1_scores)))

    # Overall Precision
    print("Overall Precision score: {0}".format(np.average(precision_scores)))
    print()

    # Test New_ROC
    predictions_poslabel = estimator.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions_poslabel, pos_label="OA")

    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.figure(2)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i = i + 1

plt.plot([0, 1], [0, 1], linestyle='--', color='black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), alpha=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Relief-SVC')
plt.legend(loc="lower right")
# Tight layout
plt.tight_layout()
filename = "PATH/" + "ROC Relief-SVC" + ".png"
plt.savefig(filename)
#plt.show()

# Compile this data
compiled_data = pd. DataFrame([1, 2, 3, 4, 5], columns=["KFold"])
compiled_data["AUC Score"] = scores
compiled_data["Accuracy Score"] = accuracy_scores
compiled_data["F1 Score"] = f1_scores
compiled_data["Precision Score"] = precision_scores
compiled_data.to_csv("PATH/Relief-SVC Scores.csv")
print(compiled_data)
print()








