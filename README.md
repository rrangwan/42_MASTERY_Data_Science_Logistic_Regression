# 42_MASTERY_Data_Science_Logistic_Regression


Given a training dataset of Harry Potter students categorized into one of four houses, with features representing courses and skills, the objective was to predict house allocation for a test dataset. A logistic regression model was employed, as the target variable was categorical rather than continuous. Data preprocessing involved analyzing general statistics and visualizations to determine which features to retain or exclude for optimal model performance.<br><br>

# usage


<br><br>


Requirement 1 <br><br>

python describe.py ../datasets/dataset_train.csv <br><br>


Requirement 2 <br><br>

python histogram.py datasets\dataset_train.csv <br>
python scatter_plot.py datasets\dataset_train.csv <br>
python pair_plot.py datasets\dataset_train.csv <br><br>

Requirement 3 <br><br>

python logreg_train.py datasets\dataset_train.csv <br>
python logreg_predict.py datasets\dataset_test.csv weights.csv <br><br>


Bonus <br><br>

python bonus_visualize.py datasets\dataset_train.csv <br><br>
