import pandas as pd
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression


class LogisticModel:
    def __init__(self, train_csv, test_csv, predictors):
        self.train_df = self.gen_train_data(train_csv)
        self.test_df = self.gen_submit_data(test_csv)
        self.cross_valid_scores = 0.0
        self.predictors = predictors
        self.alg = None

    def gen_train_data(self, data_csv, verbose=False):
        df = pd.read_csv(data_csv)
        if verbose:
            print 'Initial data %s description:' % data_csv
            print df.describe()

        # fill missing age with the median of the ages
        df.Age.fillna(df.Age.median(), inplace=True)

        # Replace all the occurence of male with the number 0.
        df.loc[df["Sex"] == "male", "Sex"] = 0
        df.loc[df["Sex"] == "female", "Sex"] = 1

        df.Embarked.fillna('S', inplace=True)
        df.loc[df.Embarked == 'S', 'Embarked'] = 0
        df.loc[df.Embarked == 'C', 'Embarked'] = 1
        df.loc[df.Embarked == 'Q', 'Embarked'] = 2

        return df

    def gen_submit_data(self, data_csv, verbose=False):
        df = self.gen_train_data(data_csv, verbose)
        df.Fare.fillna(df.Fare.median(), inplace=True)
        return df

    def fit(self):
        # Initialize our algorithm
        self.alg = LogisticRegression(random_state=1)
        # Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
        scores = cross_validation.cross_val_score(self.alg, self.train_df[self.predictors],
                                                  self.train_df["Survived"], cv=3)
        # Take the mean of the scores (because we have one for each fold)
        self.cross_valid_scores = scores.mean()
        self.alg.fit(self.train_df[self.predictors], self.train_df['Survived'])

    def logistic_reg_submit(self):
        predictions = self.alg.predict(self.test_df[self.predictors])
        # Create a new data frame with only the columns Kaggle wants from the dataset.
        return pd.DataFrame({"PassengerId": self.test_df["PassengerId"], "Survived": predictions})


def submit_sanity_check(submit_csv):
    df = pd.read_csv(submit_csv)
    assert df.shape == (418, 2)
    assert all(df.columns == ['PassengerId', 'Survived'])

if __name__ == '__main__':
    # The columns we'll use to predict the target
    titanic_train_csv = 'data/train.csv'
    titanic_test_csv = 'data/test.csv'
    titanic_submit_csv = 'data/submit.csv'
    logistics_predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    lm = LogisticModel(titanic_train_csv, titanic_test_csv, logistics_predictors)
    lm.fit()
    print 'Scores of train data cross validation: %f' % lm.cross_valid_scores
    lm.logistic_reg_submit().to_csv(titanic_submit_csv, index=False)
    submit_sanity_check(titanic_submit_csv)