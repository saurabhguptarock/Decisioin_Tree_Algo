from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import csv

data = pd.read_csv('Train.csv')
column_to_drop = ['cabin', 'name', 'embarked', 'ticket', 'home.dest', 'boat']

data = data.drop(column_to_drop, axis=1)
data = data.fillna(data['age'].mean())

le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['body'] = le.fit_transform(data['body'])

output = ['survived']

x = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'body']]
y = data[output]


def entropy(col):
    counts = np.unique(col, return_counts=True)
    N = float(col.shape[0])

    entropy = 0.0
    for ix in counts[1]:
        p = ix / N
        entropy += (-1.0 * p * np.log2(p))
    return entropy


def divide_data(x_data, fkey, fval):
    x_right = pd.DataFrame([], columns=x_data.columns)
    x_left = pd.DataFrame([], columns=x_data.columns)

    for ix in range(x_data.shape[0]):
        val = x_data[fkey].loc[ix]
        if val > fval:
            x_right = x_right.append(x_data.loc[ix])
        else:
            x_left = x_left.append(x_data.loc[ix])
    return x_left, x_right


def information__gain(x_data, fkey, fval):
    left, right = divide_data(x_data, fkey, fval)

    l = float(left.shape[0]) / x_data.shape[0]
    r = float(right.shape[0]) / x_data.shape[0]

    if left.shape[0] == 0 or right.shape[0] == 0:
        return -100000000
    i_gain = entropy(x_data.survived) - (l * entropy(left.survived) + r * entropy(right.survived))
    return i_gain


class DecisionTree:

    def __init__(self, depth=0, max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None

    def train(self, x_data):
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'body']
        info_gain = []
        for ix in features:
            i_gain = information__gain(x_data, ix, x_data[ix].mean())
            info_gain.append(i_gain)

        self.fkey = features[np.argmax(info_gain)]
        self.fval = x_data[self.fkey].mean()
        print(f'Making Tree Feature is {self.fkey}')
        left, right = divide_data(x_data, self.fkey, self.fval)
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)

        if left.shape[0] == 0 or right.shape[0] == 0:
            if x_data.survived.mean() >= 0.5:
                self.target = 1.0
            else:
                self.target = 0.0
            return
        if self.depth >= self.max_depth:
            if x_data.survived.mean() >= 0.5:
                self.target = 1.0
            else:
                self.target = 0.0
            return

        self.left = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.left.train(left)

        self.right = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.right.train(right)

        if x_data.survived.mean() >= 0.5:
            self.target = 1.0
        else:
            self.target = 0.0
        return

    def predict(self, test):
        if test[self.fkey] > self.fval:
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)


dt = DecisionTree()
dt.train(data)


data = pd.read_csv('Test.csv')
column_to_drop = ['cabin', 'name', 'embarked', 'ticket', 'home.dest', 'boat']

data = data.drop(column_to_drop, axis=1)
data = data.fillna(data['age'].mean())

le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['body'] = le.fit_transform(data['body'])

x = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'body']]


ypred = []
for i in range(x.shape[0]):
    ypred.append(dt.predict(x.loc[i]))

le = LabelEncoder()
le.fit_transform(ypred)
ypred = np.array(ypred).reshape((-1,))


with open('submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Id', 'survived'])
    for i in range(300):
        w.writerow([i, ypred[i]])
