import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.tree import export_graphviz
from io import StringIO  
from IPython.display import Image  
import pydotplus

col_names = ["Lowers fever?", "Reduces swelling?", "pain relief?", "drowsy?", "Histamine-blocker?", "Prevents nausea?", "stomach issues", "heartburn", "label"]

# Load dataset
try:
    otc = pd.read_csv("otc2.csv", header=None, names=col_names, delimiter=";")
    print(otc.head())
except Exception as e:
    print(f"An error occurred: {e}")
    
feature_cols = ["Lowers fever?", "Reduces swelling?", "pain relief?", "drowsy?", "Histamine-blocker?", "Prevents nausea?", "stomach issues", "heartburn"]
X= otc[feature_cols]
y= otc.label

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini")

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Making a picture of the decision tree

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['Advil', 'Tylenol', 'Aleve', 'ASA', 'Benadryl', 'Reactine', 'Zyrtec', 'Claratin', 'Aerius', 'Imodium', 'Tums', 'Alka-Seltzer', 'Rolaid', 'Gravol', 'Dramamine', 'Pepto-Bismol'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('otc_gini.png')
Image(graph.create_png())