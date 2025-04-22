def decisionTree():
    print("""
# Decision Tree Classifier - Using Entropy on Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv("iris.data", names=col_names)

print("Summary Statistics:")
print(df.describe())
print("\\nClass Distribution:")
print(df['class'].value_counts())

X = df.drop('class', axis=1)
y = df['class']

# Train Decision Tree with entropy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_entropy.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(15, 10))
plot_tree(clf_entropy, feature_names=col_names[:-1], class_names=clf_entropy.classes_, filled=True)
plt.title("Decision Tree (Entropy)")
plt.show()

# Evaluate
y_pred = clf_entropy.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy (Entropy):", accuracy)

cm = confusion_matrix(y_test, y_pred, labels=clf_entropy.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_entropy.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Entropy")
plt.show()

# Predict custom input
custom_input = [[5.1, 3.5, 1.5, 0.2]]
prediction = clf_entropy.predict(custom_input)
print(f"Prediction for input {custom_input[0]}: {prediction[0]}")

# Prune the tree
clf_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf_pruned.fit(X_train, y_train)
y_pruned_pred = clf_pruned.predict(X_test)
pruned_accuracy = accuracy_score(y_test, y_pruned_pred)
print("Accuracy after pruning (max_depth=3):", pruned_accuracy)""")