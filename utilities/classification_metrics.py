import unittest
from sklearn import metrics
import matplotlib.pyplot as plt

class ClassificationMetrics:
    
    def accuracy(self, y_true, y_pred):
        '''
        Function to calculate the accuracy
        :param y_true list of true y values
        :param y_pred list of pred y values
        :return accuracy score
        '''
        count = 0
        for yt, yp in zip(y_true, y_pred):
            if yt == yp:
                count += 1
        
        return count / len(y_true)
    
    def true_positive(self, y_true, y_pred):
        tp = 0
        for yt, yp in zip(y_true, y_pred):
            if yp == 1 and yt == 1:
                tp += 1
        return tp
    
    def true_negative(self, y_true, y_pred):
        tn = 0
        for yt, yp in zip(y_true, y_pred):
            if yp == 0 and yt == 0:
                tn += 1
        return tn
    
    def false_positive(self, y_true, y_pred):
        fp = 0
        for yt, yp in zip(y_true, y_pred):
            if yp == 1 and yt == 0:
                fp += 1
        return fp
    
    def false_negative(self, y_true, y_pred):
        fn = 0
        for yt, yp in zip(y_true, y_pred):
            if yp == 0 and yt == 1:
                fn += 1
        return fn
    
    def accuracy_v2(self, y_true, y_pred):
        accuracy = 0
        tp = self.true_positive(y_true, y_pred)
        tn = self.true_negative(y_true, y_pred)
        fp = self.false_positive(y_true, y_pred)
        fn = self.false_negative(y_true, y_pred)
        
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        return accuracy
     
    def precision(self, y_true, y_pred):
        precision = 0
        tp = self.true_positive(y_true, y_pred)
        fp = self.false_positive(y_true, y_pred)
        precision = tp / (tp + fp)
        return precision
    
    def recall(self, y_true, y_pred):
        recall = 0
        tp = self.true_positive(y_true, y_pred)
        fn = self.false_negative(y_true, y_pred)
        recall = tp / (tp + fn)
        return recall
    
    def plot_precision_recall(self):
        y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0] 
        y_prob = [0.02638412, 0.11114267, 0.31620708, 0.0490937,  0.0191491, 0.17554844,            
                  0.15952202, 0.03819563, 0.11639273, 0.079377,   0.08584789, 0.39095342, 0.27259048, 
                  0.03447096, 0.04644807, 0.03543574, 0.18521942, 0.05934905, 0.61977213, 0.33056815]
        
        thresholds = [0.0490937 , 0.05934905, 0.079377,  0.08584789, 0.11114267, 0.11639273,  
                      0.15952202, 0.17554844, 0.18521942,  0.27259048, 0.31620708, 0.33056815,  
                      0.39095342, 0.61977213] 
        
        precisions = []
        recalls = []
        
        for thres in thresholds:
            y_pred = [ 1 if x >= thres else 0 for x in y_prob]
            precision = self.precision(y_true, y_pred)
            recall = self.recall(y_true, y_pred)
            precisions.append(precision)
            recalls.append(recall)
        
        return precisions, recalls
        
    
    def f1_score(self, y_true, y_pred):
        f1_score = 0
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        
        f1_score = (2 * precision * recall) / ( precision + recall)
        return f1_score
    
    def tpr_sensitivity(self, y_true, y_pred):
        return self.recall(y_true, y_pred)
    
    
class TestClassificationMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.l1 = [0, 1, 1, 1, 0, 0, 0, 1]
        self.l2 = [0, 1, 0, 1, 0, 1, 0, 0]
        self.classification_metrics = ClassificationMetrics()
    
    def test_accuracy(self):
        self.assertEqual(self.classification_metrics.accuracy(self.l1, self.l2), 
                         metrics.accuracy_score(self.l1, self.l2),
                        "Incorrect accuracy score")
        
    def test_accuracy_v2(self):
        self.assertEqual(self.classification_metrics.accuracy_v2(self.l1, self.l2), 
                         metrics.accuracy_score(self.l1, self.l2),
                        "Incorrect accuracy score")
    
    
    def test_true_positive(self):
        tp = metrics.confusion_matrix(self.l1, self.l2).ravel()[3]
        self.assertEqual(self.classification_metrics.true_positive(self.l1, self.l2),
                         tp,
                         "Incorrect true positive")
    
    def test_true_negative(self):
        tn = metrics.confusion_matrix(self.l1, self.l2).ravel()[0]
        self.assertEqual(self.classification_metrics.true_negative(self.l1, self.l2),
                         tn,
                         "Incorrect true negative")
        
    def test_false_positive(self):
        fp = metrics.confusion_matrix(self.l1, self.l2).ravel()[1]
        self.assertEqual(self.classification_metrics.false_positive(self.l1, self.l2),
                         fp,
                         "Incorrect false positive")
    
    def test_false_negative(self):
        fn = metrics.confusion_matrix(self.l1, self.l2).ravel()[2]
        self.assertEqual(self.classification_metrics.false_negative(self.l1, self.l2),
                         fn,
                         "Incorrect false negative")

    def test_precision(self):
        precision = metrics.precision_score(self.l1, self.l2)
        self.assertEqual(self.classification_metrics.precision(self.l1, self.l2),
                         precision,
                         "Incorrect precision")
        
    def test_recall(self):
        recall = metrics.recall_score(self.l1, self.l2)
        self.assertEqual(self.classification_metrics.recall(self.l1, self.l2),
                         recall,
                         "Incorrect recall")
     
    def test_f1_score(self):
        f1_score = metrics.f1_score(self.l1, self.l2)
        self.assertEqual(self.classification_metrics.f1_score(self.l1, self.l2),
                         f1_score,
                         "Incorrect f1 score")
    
    def test_tpr_sensitivity(self):
        tpr = metrics.recall_score(self.l1, self.l2)
        self.assertEqual(self.classification_metrics.tpr_sensitivity(self.l1, self.l2),
                         tpr,
                         "Incorrect sensitivity")


        
if __name__ == '__main__':
    precisions, recalls = ClassificationMetrics().plot_precision_recall()
    #plt.plot(recalls, precisions)
    #plt.xlabel('Recalls')
    #plt.ylabel('Precisions')
    #plt.show()
    unittest.main()