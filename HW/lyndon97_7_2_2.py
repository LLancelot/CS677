import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from texttable import Texttable

url = r'https://archive.ics.uci.edu/ml/'  + \
           r'machine-learning-databases/iris/iris.data'

data = pd.read_csv(url, names=['sepal-length', 'sepal-width',
                     'petal-length', 'petal-width', 'Class'])

features = ['sepal-length', 'sepal-width','petal-length', 'petal-width']
features_rm1 = ['sepal-width','petal-length', 'petal-width']
features_rm2 = ['sepal-length','petal-length', 'petal-width']
features_rm3 = ['sepal-length', 'sepal-width', 'petal-width']
features_rm4 = ['sepal-length', 'sepal-width','petal-length']

class_labels = ['Iris-setosa', 'Iris-versicolor', "Iris-virginica"]

data_c1 = data.copy()
data_c2 = data.copy()
data_c3 = data.copy()
for index, row in data_c1.iterrows():
    if row['Class'] != 'Iris-setosa':
        data_c1.loc[index,'Class'] = 'others'

for index, row in data_c2.iterrows():
    if row['Class'] != 'Iris-versicolor':
        data_c2.loc[index, 'Class'] = 'others'

for index, row in data_c3.iterrows():
    if row['Class'] != 'Iris-virginica':
        data_c3.loc[index, 'Class'] = 'others'

le = LabelEncoder()
Y_c1 = le.fit_transform(data_c1['Class'].values)
Y_c2 = le.fit_transform(data_c2['Class'].values)
Y_c3 = le.fit_transform(data_c3['Class'].values)





def calculate(feature_value, class_value):
    X_train, X_test, Y_train, Y_test = train_test_split(feature_value, class_value, test_size=0.5, random_state=3)
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(X_train, Y_train)
    prediction = log_reg_classifier.predict(X_test)
    accuracy = np.mean(prediction == Y_test)
    return accuracy

contr_c1_all = calculate(data_c1[features].values, class_value=Y_c1)
contr_c1_f1 = calculate(data_c1[features_rm1].values, class_value=Y_c1)
contr_c1_f2 = calculate(data_c1[features_rm2].values, class_value=Y_c1)
contr_c1_f3 = calculate(data_c1[features_rm3].values, class_value=Y_c1)
contr_c1_f4 = calculate(data_c1[features_rm4].values, class_value=Y_c1)

contr_c2_all = calculate(data_c2[features].values, class_value=Y_c2)
contr_c2_f1 = calculate(data_c2[features_rm1].values, class_value=Y_c2)
contr_c2_f2 = calculate(data_c2[features_rm2].values, class_value=Y_c2)
contr_c2_f3 = calculate(data_c2[features_rm3].values, class_value=Y_c2)
contr_c2_f4 = calculate(data_c2[features_rm4].values, class_value=Y_c2)

contr_c3_all = calculate(data_c3[features].values, class_value=Y_c3)
contr_c3_f1 = calculate(data_c3[features_rm1].values, class_value=Y_c3)
contr_c3_f2 = calculate(data_c3[features_rm2].values, class_value=Y_c3)
contr_c3_f3 = calculate(data_c3[features_rm3].values, class_value=Y_c3)
contr_c3_f4 = calculate(data_c3[features_rm4].values, class_value=Y_c3)

res_table = pd.DataFrame(
    {
        'Flower':['sepal length ∆','sepal width ∆','petal length ∆','petal width ∆'],
        'Versicolor':[str(round((contr_c2_all - contr_c2_f1)*100,2))+"%",
                      str(round((contr_c2_all - contr_c2_f2)*100,2))+"%",
                      str(round((contr_c2_all - contr_c2_f3)*100,2))+"%",
                      str(round((contr_c2_all - contr_c2_f4)*100,2))+"%"],
        'Setosa':[str(round((contr_c1_all - contr_c1_f1)*100,2))+"%",
                      str(round((contr_c1_all - contr_c1_f2)*100,2))+"%",
                      str(round((contr_c1_all - contr_c1_f3)*100,2))+"%",
                      str(round((contr_c1_all - contr_c1_f4)*100,2))+"%"],
        'Virginica':[str(round((contr_c3_all - contr_c3_f1)*100,2))+"%",
                      str(round((contr_c3_all - contr_c3_f2)*100,2))+"%",
                      str(round((contr_c3_all - contr_c3_f3)*100,2))+"%",
                      str(round((contr_c3_all - contr_c3_f4)*100,2))+"%"]
    }
)
tb=Texttable()
tb.set_cols_align(['l','r','r','r'])
tb.set_cols_dtype(['t','i','i','i'])
tb.header(res_table.columns)
tb.add_rows(res_table.values,header=False)

print(tb.draw())