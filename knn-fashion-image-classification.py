#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print(sys.version)


# In[60]:


import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report as cls_report


# In[9]:


D_train = pd.read_csv(filedialog.askopenfilename(), header=None)


# In[11]:


D_test = pd.read_csv(filedialog.askopenfilename(), header = None)


# In[12]:


column_names = ["F_Type"] + [f"Pix{i}" for i in range(1, 785)]


# In[13]:


D_train.columns = column_names


# In[14]:


D_test.columns = column_names


# In[20]:


# Define the mapping dictionary
train_label_mapping = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Replace the values in the column with their corresponding labels
D_train['F_Type'] = D_train['F_Type'].map(train_label_mapping)


# In[21]:


# Define the mapping dictionary
test_label_mapping = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Replace the values in the column with their corresponding labels
D_test['F_Type'] = D_test['F_Type'].map(test_label_mapping)


# In[23]:


D_test.head()


# In[24]:


# Compute the frequency table
D_train['F_Type'] = D_train['F_Type'].astype('category')
frequency_table = D_train['F_Type'].value_counts()
print(frequency_table)


# Compute the proportion table
proportion_table = frequency_table / frequency_table.sum()

# Compute the rounded percentages
rounded_percentages = np.round(proportion_table * 100, decimals=1)

print(rounded_percentages)


# In[25]:


D_train.shape


# In[26]:


#Normalizing Train Data

# Select the numeric columns in the DataFrame
train_numeric_columns = D_train.iloc[:, 1:785]

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Normalize the data using the MinMaxScaler
train_normalized_data = scaler.fit_transform(train_numeric_columns)

# Create a new DataFrame with the normalized data
train_NormData = pd.DataFrame(train_normalized_data, columns=train_numeric_columns.columns)


# In[27]:


#Normalizing Test Data

# Select the numeric columns in the DataFrame
test_numeric_columns = D_test.iloc[:, 1:785]

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Normalize the data using the MinMaxScaler
test_normalized_data = scaler.fit_transform(test_numeric_columns)

# Create a new DataFrame with the normalized data
test_NormData = pd.DataFrame(test_normalized_data, columns=test_numeric_columns.columns)


# In[28]:


X_train = train_NormData
y_train = D_train.iloc[:,0]
X_test = test_NormData
y_test = D_test.iloc[:,0]


# In[69]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time

# Specify the range of k values
start_k = 2
end_k = 25



# Initialize lists to store k values, elapsed times, and accuracies
k_values = []
elapsed_times = []
accuracies = []

# Loop over different values of k
for k in range(start_k, end_k+1):
    
    # Start the timer
    start_time = time.time()
    # Train the KNN classifier
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)


    # Print the elapsed time and the current k value
    print("k =", k, "- Elapsed Time:", elapsed_time, "seconds")

    # Make predictions on the testing set
    D_pred = knn_model.predict(X_test)

    # Calculate the elapsed time for each iteration
    elapsed_time = time.time() - start_time

    # Calculate evaluation metrics
    classification_results = cls_report(y_test, D_pred)
    report_lines = classification_results.split('\n')
    accuracy_line = next((line for line in report_lines if 'accuracy' in line.lower()), None)

    if accuracy_line is not None:
        # Extract accuracy value
        accuracy = accuracy_line.split()[1]
        print("Accuracy:", accuracy)
    else:
        print("Accuracy information not found in the classification report.")
        continue

    # Store the k value, elapsed time, and accuracy in the lists
    k_values.append(k)
    elapsed_times.append(elapsed_time)
    accuracies.append(float(accuracy))

# Plot the line graph
# plt.plot(k_values, elapsed_times, label='Elapsed Time')
plt.plot(k_values, accuracies*100, label='Accuracy')
plt.xlabel('k value')
plt.ylabel('Accuracy %')
plt.title('KNN Performance')
plt.legend()
plt.show()



# In[71]:


# Plot the line graph
plt.plot(k_values, elapsed_times, label='Elapsed Time')
#plt.plot(k_values, accuracies, label='Accuracy')
plt.xlabel('k value')
plt.ylabel('Elapsed Time / Accuracy')
plt.title('KNN Performance')
plt.legend()
plt.show()

