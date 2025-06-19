from google.colab import files
files.upload()
# Move kaggle.json to the right location and set permissions
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c titanic

!unzip titanic.zip


import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv('train.csv')

# Data Cleaning
# Fill missing Age with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin and Name columns
df.drop(columns=['Cabin', 'Name'], inplace=True)

# Convert categorical to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Exploratory Data Analysis

# 1. Survival Count
df['Survived'].value_counts().plot(kind='bar')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Number of Passengers')
plt.show()

# 2. Survival by Gender
survived_gender = df.groupby('Sex')['Survived'].value_counts().unstack()
survived_gender.plot(kind='bar', stacked=True)
plt.title('Survival by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.show()

# 3. Age Distribution
plt.hist(df['Age'], bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# 4. Survival by Passenger Class
pclass_survival = df.groupby('Pclass')['Survived'].value_counts().unstack()
pclass_survival.plot(kind='bar', stacked=True)
plt.title('Survival by Passenger Class')
plt.xlabel('Pclass (1 = Upper, 2 = Middle, 3 = Lower)')
plt.ylabel('Count')
plt.show()

# 5. Correlation Matrix
correlation = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", correlation)

fig, ax = plt.subplots(figsize=(10, 6))
cax = ax.matshow(correlation, cmap='coolwarm')
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
plt.yticks(range(len(correlation.columns)), correlation.columns)
fig.colorbar(cax)
plt.title('Correlation Matrix', pad=20)
plt.show()

# Optional Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch']
df['IsAlone'] = (df['FamilySize'] == 0).astype(int)

print("\nSample of engineered features:")
print(df[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head())
