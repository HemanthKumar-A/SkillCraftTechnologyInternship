import matplotlib.pyplot as plt
import pandas as pd
data = {
    "Name": [f"Person{i+1}" for i in range(20)],
    "Age": [23, 25, 31, 22, 28, 35, 29, 24, 30, 26, 33, 27, 40, 38, 34, 21, 37, 36, 32, 39],
    "Gender": ['Male', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male',
               'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Male', 'Female', 'Male']
}

df = pd.DataFrame(data)
plt.subplot(1,2,2)
gender_counts = df['Gender'].value_counts()
plt.bar(gender_counts.index, gender_counts.values, color=['lightcoral', 'lightblue'])
plt.title('Bar Chart of Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()