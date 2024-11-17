import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.tree import export_graphviz
from io import StringIO  
from IPython.display import Image  
import pydotplus

col_names = ["Fever?", "Swelling?", "Pain?", "Is it okay to be drowsy?", "Allergy symptoms?", "Nausea?", "My tummy hurts", "Suspected heart attack?", "Heartburn?", "label"]

# Load dataset
try:
    otc = pd.read_csv("ScenarioDataset.csv", header=None, names=col_names, delimiter=",")
    print(otc.head())
except Exception as e:
    print(f"An error occurred: {e}")
    
feature_cols = ["Fever?", "Swelling?", "Pain?", "Is it okay to be drowsy?", "Allergy symptoms?", "Nausea?", "My tummy hurts", "Suspected heart attack", "Heartburn?"]
X= otc[feature_cols]
y= otc.label

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1) # 99% training and 1% test

#Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy")

#Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Making a picture of the decision tree
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['Ibuprofen', 'Acetaminophen', 'Acetylsalicylic acid', 'Diphenhydramine', 'Cetrizine hydrocholoride', 'Calcium carbonate','Ginger', 'Dimenhydrinate'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('scenario_entropy.png')
Image(graph.create_png())