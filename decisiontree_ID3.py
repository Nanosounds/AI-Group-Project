import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import metrics
from io import StringIO  
from IPython.display import Image  
import pydotplus


#Defining column names
col_names = ["Fever?", "Swelling?", "Pain?", "Is it okay to be drowsy?", "Allergy symptoms?", "Nausea?", "My tummy hurts", "Suspected heart attack?", "Heartburn?", "label"]

#Loading the dataset
try:
    otc = pd.read_csv("ScenarioDataset.csv", header=None, names=col_names, delimiter=",")
except Exception as e:
    print(f"An error occurred: {e}")
    
#Defining features and labels 
feature_cols = ["Fever?", "Swelling?", "Pain?", "Is it okay to be drowsy?", "Allergy symptoms?", "Nausea?", "My tummy hurts", "Suspected heart attack?", "Heartburn?"]
X = otc[feature_cols]
y = otc.label

#Creating Decision Tree classifier object
clf = DecisionTreeClassifier(criterion="entropy")

#Training Decision Tree Classifier
clf = clf.fit(X, y)

#Predict the response for the dataset
y_pred = clf.predict(X)

#In-sample accuracy
accuracy = metrics.accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

#Visualizing the decision tree
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=clf.classes_)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('entropy_tree.png')
Image(graph.create_png())

