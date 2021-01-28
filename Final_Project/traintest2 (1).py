import sklearn
import argparse
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.dataset import Dataset

#importing the libraries that we use
import pandas as pd 
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import SVC
#sns.set(color_codes=True) # adds a nice background to the graphs

#dat = "https://toriaworstorage3dbdc84f5.blob.core.windows.net/azureml-blobstore-55d46784-4d95-4e48-a295-f8417ee4bb2e"
dat = "https://raw.githubusercontent.com/vikkyfama/Project_Capstone/toribranch/Personal_loans.csv"
ds = TabularDatasetFactory.from_delimited_files(dat)

def clean_data(data):
    #dataset = Dataset.get_by_name(ws, name = 'Personal_loan')
    x_df = data.to_pandas_dataframe()

    avg_Experience = x_df["Experience"].mean()
    print(f"Average Experience {avg_Experience}")
    x_df["Experience"] = x_df["Experience"].apply(lambda x : avg_Experience if x<0 else x)

    q1 = x_df["Income"].quantile(0.25)
    q3 = x_df["Income"].quantile(0.75)
    iqr = q3 - q1
    border = q1 + 1.5 * iqr
    P_inc = x_df["Income"].apply(lambda x : border if x>border else x)


    q1 = x_df["Mortgage"].quantile(0.25)
    q3 = x_df["Mortgage"].quantile(0.75)
    iqr = q3 - q1
    border = q1 + 1.5 * iqr
    P_mortgage = x_df["Mortgage"].apply(lambda x : border if x>border else x)
    

    x_df["Income"]=P_inc
    x_df["Mortgage"]=P_mortgage


    x_df["Mortgage"]= np.log1p(x_df["Mortgage"])
    X_df = x_df.drop(["Personal Loan","Age","ZIP Code","CreditCard","ID","Online"],axis=1)     # Predictor feature columns
    Target = x_df["Personal Loan"]
    #Y = Target   # Predicted class (1, 0) 

     
    y_df = Target   # Predicted class (1, 0)
    return X_df, y_df

X, Y = clean_data(ds)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    # 1 is just any random seed number
normalizer = preprocessing.Normalizer().fit(x_train)
normalizer.transform(x_train)
normalizer = preprocessing.Normalizer().fit(x_test)
normalizer.transform(x_test)   


# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
#dat = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

#ds = TabularDatasetFactory.from_delimited_files(dat)

# x, y = clean_data(ds)

# TODO: Split data into train and test sets.

# ## YOUR CODE HERE ###a

# run = Run.get_context()


def main(): 


    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations to converge")

    args = parser.parse_args(args = [])

    run = Run.get_context()
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    accuracy = model.score(x_test, y_test)

    run = Run.get_context()
    run.log("Accuracy", np.float(accuracy))

    #Save the trained model
    os.makedirs('outputs', exist_ok=True)
    print('model saved')
    joblib.dump(value=model, filename='outputs/model.pkl')
    print('dumped')

    run.complete()




if __name__ == '__main__':
    main()
