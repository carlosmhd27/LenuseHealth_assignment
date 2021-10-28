from Cleaning_data import cleaning_data
from numpy import shape
from os.path import join, isfile, isdir
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import  metrics, model_selection
from sklearn.inspection import permutation_importance

## Get the data an clean it
path_to_dataset = "./"
dataset_name = join(path_to_dataset, "Data_Scientist_-_Case_Dataset_splitted.xlsx")

data = cleaning_data(dataset_name,
                    save_csv = False,
                    print_info = True)
print(data.dtypes)

# data.pop('gender')
# data.pop('branch')
# data.pop('credit_account_id')

## Separate in x and y
convert = data.pop("converted")
## Shuffle and separate into training and validation sets
## test size is 25 by default
X_train, X_val, y_train, y_val = train_test_split(
     data, convert, random_state=0)

## define the models, AdaB and XGB as they are simple and precise models
## for classification, making a higher order model, as a Neural Network,
## would not increase much further the result and would make it worse to
## look for the important features
model_ada = AdaBoostClassifier(n_estimators=100, random_state=0)
model_xgb = XGBClassifier(objective = 'binary:logistic', use_label_encoder=False,
                          max_depth = 2, learning_rate = 1.0,
                          n_estimators = 5, random_state=0)
model_ada.fit(X_train, y_train)
score_ada = model_ada.score(X_val, y_val)

model_xgb = model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_val)
score_xgb = metrics.accuracy_score(y_val, y_pred)
print(f"The accuracy of the AdaBoostClassifier is {score_ada:.3f}")
print(f"The accuracy of the XGBClassifier is {score_xgb:.3f}")

## As we can see, this method can give us some good results

## Now, let's look at the most important feature, for that
## We will use the permutation_importance method from scikit-learn
## Which basically erases different variables to see how the precission drops

r_ada = permutation_importance(model_ada, X_val, y_val,
                                n_repeats=30,
                                random_state=0)

r_xgb = permutation_importance(model_xgb, X_val, y_val,
                                n_repeats=30,
                                random_state=0)

for r in [r_ada, r_xgb]:
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{data.keys()[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
