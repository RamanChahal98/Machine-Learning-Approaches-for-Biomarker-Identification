# Machine-Learning-Approaches-for-Biomarker-Identification
A Machine learning approach for the identification of biomarkers in osteoarthritis, although methodology is generic enough to applied for other biomedical data. Dataset is provided by CHECK (Cohort Hip &amp; Cohort Knee) - https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:62981.  In total 8 Machine Learning models are attached using feature selectors RFE and Relief, with classifiers Random Forest (RF), Logistic Regression (LR), Support Vector Classifier (SVC) and Xgboost (Xgb). Python 3.8

These are split into 8 experiments named:

1. Experiment_RFE_RF
2. Experiment_Relief_RF
3. Experiment_RFE_LR
4. Experiment_Relief_LR
5. Experiment_RFE_SVC
6. Experiment_Relief_SVC
7. Experiment_RFE_Xgb
8. Experiment_Relief_Xgb


--- Installation ---

Packages used and how to install in Linux (using pip):

numpy: 
	>> pip install numpy
pandas:
	>> pip install pandas
sklearn:
	>>  pip install sklearn
xgboost:
	>>  pip install xgboost
matplotlib:
	>>  pip install matplotlib
seaborn:
	>>  pip install seaborn

--- Usage ---

To use these models with the data provided - or another biomedical dataset, you may need to change the loading location where the data is loaded:

data = pd.read_csv("data location here")

Data locations will need to be changed by user a total of 5 times. These are the 1. data location just mentioned,

2. when gathering data about feature selector and classifier after fitting:
other_data.to_csv(path_or_buf="where you want the csv to be located" + str(i) + ".csv")

3. feature importance plot:
filename = "where you want plots to be located" + str(get_scores()) + "Feature_Importance" + ".png"

4. ROC plots:
filename = "where you want ROC plots to be located" + "ROC RFE-RF" + ".png"

5. Complied data, with metrics AUC, Accuracy, F1 and precision
compiled_data.to_csv("where you want csv to be located/named.csv")

Each python script must have the changes mentioned above

If done correctly, you should have generated feature importance plots for each round of CV and a single combined ROC plot for each model, along with other performance metrics.     
