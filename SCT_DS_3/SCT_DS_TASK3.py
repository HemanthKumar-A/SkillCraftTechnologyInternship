from google.colab import files
uploaded = files.upload()
df = pd.read_csv('bank-full.csv', sep=';')
print(df.head())

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Step 1: Download & load
!wget https://github.com/selva86/datasets/raw/master/bank-full.csv -P datasets
df = pd.read_csv('datasets/bank-full.csv', sep=';')

# Step 2: Preprocess
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('y', axis=1)
y = df['y']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Train Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
clf.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Visualize the tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['no','yes'], filled=True)
plt.show()
