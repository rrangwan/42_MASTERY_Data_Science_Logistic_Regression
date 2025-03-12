# 42_MASTERY_Data_Science_Logistic_Regression


Given a training set of data for Harry Potter students belonging to one of four houses; with features of courses and skills; predict the allocation to house given test data.
This was a logictic model, because the houses were categories and not continous data. <br><br>

The data was analyzed to determine which field to use and which to drop by looking at the general statistics as well as assessing visualizations. <br><br>

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
