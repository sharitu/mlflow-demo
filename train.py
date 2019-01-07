import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import mlflow
import mlflow.sklearn

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv("./WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # print(df.head())
    plt.bar(df['Dependents'].value_counts().index, df['Dependents'].value_counts().values)
    # print(df.corr())
    df['TotalCharges'].replace(to_replace=' ', value=np.NaN, inplace=True) # find and replace missing value with np.NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges']) # convert the data column to numeric dtype
    tc_median = df['TotalCharges'].median() # calculate median
    df['TotalCharges'].fillna(tc_median, inplace=True) # replace missing value with median value
    ndf = df.copy() # create a new copy of dataframe
    # print(tc_median)
    bool_cols = [col for col in df.columns if col not in ['gender','SeniorCitizen'] and len(df[col].unique()) == 2] # identify boolean columns
    # print(bool_cols) # boolean columns

    for col in bool_cols: # iterate through boolean columns
        ndf[col] = np.where(ndf[col]=='No',0, 1) # replace Yes/No values with 1/0

    ndf['gender'] = np.where(ndf['gender']=='Female', 0, 1) # replace Female/Male with 0/1, this is also known as binary encoding

    ndf.drop('customerID', axis=1, inplace=True) # drop primary key / id column from the table
    other_cat_cols = [col for col in ndf.select_dtypes('object').columns if col not in bool_cols and col not in ['customerID', 'gender', 'SeniorCitizen']] # find other categorical column
    # print(other_cat_cols)

    ndf_dummies = pd.get_dummies(ndf) # One hot encode categorical columns which have more than 2 classes
    # print(ndf_dummies.head())
    from sklearn.model_selection import train_test_split
    X = ndf_dummies.drop('Churn', axis=1).copy() # create X | feature columns
    y = ndf_dummies['Churn'].copy() # create y | target column
    X_cv = X.iloc[-100:].copy() # cross validation feature set
    y_cv = y.iloc[-100:].copy() # cross validation target set
    X = X.iloc[:-100].copy() # remove cross validation feature set from X
    y = y.iloc[:-100].copy() # remove cross validation target set from y
    # print(X.shape, y.shape, X_cv.shape, y_cv.shape) # matrix shapes

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42) # train test split for model training and testing
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape) # matrix shapes

    from sklearn.linear_model import ElasticNet # simple logistic regression model for binary classification
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        y_cv_pred = model.predict(X_cv)
        (rmse, mae, r2) = eval_metrics(y_cv, y_cv_pred)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")
        print("  Params: alpha: %s  l1_ratio: %s" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

# accuracy = model.score(X_test, y_test) # accuracy score on test set
# print('Accuracy %.3f' %(accuracy*100)+' %') # print accuracy score %
# print('Recall score %.3f' %recall_score(y_cv, y_cv_pred))
# print('f1 %f' %f1_score(y_cv, y_cv_pred))