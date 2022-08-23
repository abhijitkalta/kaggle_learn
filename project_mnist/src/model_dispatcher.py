from sklearn import tree
from sklearn import ensemble

models = { 
    "decision_tree_ginni" : tree.DecisionTreeClassifier(criterion = 'ginni'),
    "decision_tree_entropy" : tree.DecisionTreeClassifier(criterion = 'entropy'),
    "rf" : ensemble.RandomForestClassifier()
}