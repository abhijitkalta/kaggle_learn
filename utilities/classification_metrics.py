import unittest
from sklearn import metrics

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



        
if __name__ == '__main__':
    unittest.main()
    