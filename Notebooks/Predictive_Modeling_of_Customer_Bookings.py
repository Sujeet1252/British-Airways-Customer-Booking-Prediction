#!/usr/bin/env python
# coding: utf-8

# # Predictive modeling of customer bookings

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)


# In[2]:


df = pd.read_csv("customer_booking.csv", encoding="ISO-8859-1")
df.head()


# In[3]:


print("Shape :", df.shape)

df.info()

df.describe(include="all")


# In[4]:


df.isnull().sum()


# In[5]:


print("Duplicate Rows :", df.duplicated().sum())


# In[6]:


plt.figure(figsize=(5,3))

sns.countplot(x="booking_complete", data=df)

plt.title("Booking Completion")

plt.show()


# In[7]:


df = df.drop_duplicates()

print("New Shape :", df.shape)


# In[8]:


df["booking_complete"].value_counts()


# In[9]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_columns = [
    "sales_channel",
    "trip_type",
    "flight_day",
    "route",
    "booking_origin"
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

df.head()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))

sns.heatmap(
    df.corr(),
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")

plt.show()


# In[11]:


X = df.drop("booking_complete", axis=1)

y = df["booking_complete"]


# In[12]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[13]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# In[15]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Accuracy : {accuracy*100:.2f}%")


# In[16]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[21]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4,3))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# In[29]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, threshold = roc_curve(y_test, y_prob)

plt.figure(figsize=(4,3))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()


# In[32]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print(scores)
print("Average Accuracy :", scores.mean())


# In[30]:


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance


# In[33]:


plt.figure(figsize=(8,5))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.show()


# In[ ]:




