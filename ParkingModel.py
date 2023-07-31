import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# prepare data
input_dir = './clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train/test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]  # 12 different classifier and choose the best based on the performance with the combination of gamma & C
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print('{}% of sample were correctly classified'.format(str(score*100)))

# Evaluating the model performance
accuracy = metrics.accuracy_score(y_test, y_prediction)
precision = metrics.precision_score(y_test, y_prediction)
recall = metrics.recall_score(y_test, y_prediction)
f1_score = metrics.f1_score(y_test, y_prediction)
confusion_matrix = metrics.confusion_matrix(y_test, y_prediction)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Confusion Matrix:\n", confusion_matrix)

# Plotting the confusion matrix
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Plotting the metrics
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1_score]

plt.bar(labels, values)
plt.title("Model Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)  # Set the y-axis limit to the range of [0, 1]
plt.show()
#pickle.dump(best_estimator,open('./ParkingModel.p','wb'))