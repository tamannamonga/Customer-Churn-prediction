import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
df=pd.read_csv(r"c:\Users\hp\Downloads\archive (3)\WA_Fn-UseC_-Telco-Customer-Churn.csv")
#DataCleaning
# print(df.describe())
# print(df.info())
# print(df.isnull().sum().sum())
# print(df.shape)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
print(df['TotalCharges'].isnull().sum())
print(df['Churn'].value_counts())
print(df.duplicated().sum())
df.drop('customerID', axis=1, inplace=True)
df.replace("No internet service", "No", inplace=True)
df.replace("No phone service", "No", inplace=True)
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0}) #converted into num value for encoding.
#label encoding is used for binary columns and one hot for nominal columns.
#(because labelEncoder assume orders like 3>2>1.)
df = pd.get_dummies(df, drop_first=True)
print(df.info())
print(df.isnull().sum().sum())
x = df.drop("Churn", axis=1)
y = df["Churn"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#Trying different models.
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
rf = RandomForestClassifier(class_weight="balanced") 
# dataset is imbalanced with more signifacnt non-churn customers.
# to ensure that the model do not bias towards majority class,
# class weight = balance is used here.
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
dt = DecisionTreeClassifier(random_state=42)
dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
importance = pd.Series(model.coef_[0], index=x.columns) # links features with model coefficients.
# coefficient tells how strongly that feature affects churn prediction.
# +ive = feature increase prob of churn. -ive = less prob of churn.
print(importance.sort_values(ascending=False)) #sorts them so we can see the most important factors.
# comparing all 3 logistic regression works best.
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(len(x.columns))