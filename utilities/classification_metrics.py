import unittest
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

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
    
    def fpr(self, y_true, y_pred):
        fp = self.false_positive(y_true, y_pred)
        tn = self.false_negative(y_true, y_pred)
        
        fpr = fp / ( fp + tn )
        return fpr
    
    def plot_roc(self):
        y_true = [0, 0, 0, 0, 1, 0, 1,  0, 0, 1, 0, 1, 0, 0, 1] 
        y_prob = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,  0.9, 0.5, 0.3, 0.66, 0.3, 0.2,  0.85, 0.15, 0.99]
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0] 
        
        tprs = []
        fprs = []
        
        for thres in thresholds:
            y_pred = [ 1 if x >= thres else 0 for x in y_prob ]
            tpr = self.tpr_sensitivity(y_true, y_pred)
            fpr = self.fpr(y_true, y_pred)
            tprs.append(tpr)
            fprs.append(fpr)
        
        data = {
            'tpr': tprs,
            'fpr': fprs,
            'thresholds': thresholds
        }
        roc_df = pd.DataFrame(data)
        
        return roc_df
    
    
    def log_loss(self, y_true, y_prob):
        losses = []
        epsilon = 1e-15
        for yt, yp in zip(y_true, y_prob):
            yp = np.clip(yp, epsilon, 1-epsilon)
            loss = -1 * (yt * np.log(yp) + (1-yt)*(np.log( 1 - yp)))
            losses.append(loss)
        
        return np.mean(losses)
    
    def macro_precision(self, y_true, y_pred):
        num_classes = len(pd.unique(y_true))
        
        precision = 0
        
        for class_ in range(num_classes):
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            
            tp = self.true_positive(temp_true, temp_pred)
            fp = self.false_positive(temp_true, temp_pred)
            
            temp_precision = tp / (tp + fp)
            precision += temp_precision
        
        return precision/(num_classes)
    
    def micro_precision(self, y_true, y_pred):
        num_classes = len(pd.unique(y_true))
        
        tp = 0
        fp = 0
        
        for class_ in range(num_classes):
            
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            
            temp_tp = self.true_positive(temp_true, temp_pred)
            temp_fp = self.false_positive(temp_true, temp_pred)
            
            tp += temp_tp
            fp += temp_fp
        
        precision = tp / (tp + fp)
        return precision
    
    def weighted_precision(self, y_true, y_pred):
        num_classes = len(pd.unique(y_true))
        class_counts = Counter(y_true)
        
        precision = 0
        
        for class_ in range(num_classes):
            temp_true = [1 if p == class_ else 0 for p in y_true]
            temp_pred = [1 if p == class_ else 0 for p in y_pred]
            
            tp = self.true_positive(temp_true, temp_pred)
            fp = self.false_positive(temp_true, temp_pred)
            
            temp_precision = tp / (tp + fp)
            weighted_precision = class_counts[class_] * temp_precision
            precision += weighted_precision
        
        return precision/(len(y_true))
    
    def pk(self, y_true, y_pred, k):
        if k == 0:
            return 0
        
        y_pred = y_pred[:k]
        pred_set = set(y_pred)
        true_set = set(y_true)
        common_values = pred_set.intersection(true_set)
        
        return len(common_values) / len(y_pred[:k])
    
    def apk(self, y_true, y_pred, k):
        pk_values = []
        for i in range(1, k+1):
            pk_values.append(self.pk(y_true, y_pred, i))
        
        if len(pk_values) == 0:
            return 0
        
        return sum(pk_values) / len(pk_values)
    
    def mapk(self, y_true, y_pred, k):
        apk_values = []
        for i in range(len(y_true)):
            apk_values.append(self.apk(y_true[i], y_pred[i], k = k))
        
        return sum(apk_values) / len(apk_values)
        
class TestClassificationMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.l1 = [0, 1, 1, 1, 0, 0, 0, 1]
        self.l2 = [0, 1, 0, 1, 0, 1, 0, 0]
        self.l3 = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1] 
        self.l4 =  [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99] 
        self.l5 =  [ [1, 2, 3], [0, 2], [1], [2, 3], [1, 0], []] 
        self.l6 = [ [0, 1, 2], [1], [0, 2, 3], [2, 3, 4, 0], [0, 1, 2], [0]] 
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
        
    
    def test_log_loss(self):
        log_loss = metrics.log_loss(self.l3, self.l4)
        self.assertEqual(self.classification_metrics.log_loss(self.l3, self.l4),
                         log_loss,
                         "Incorrect log loss")
        
    
    def test_macro_precision(self):
        macro_precision = metrics.precision_score(self.l1, self.l2, average = 'macro')
        self.assertEqual(self.classification_metrics.macro_precision(self.l1, self.l2),
                         macro_precision,
                         "Incorrect macro precision")
        
    def test_micro_precision(self):
        micro_precision = metrics.precision_score(self.l1, self.l2, average = 'micro')
        self.assertEqual(self.classification_metrics.micro_precision(self.l1, self.l2),
                         micro_precision,
                         "Incorrect micro precision")
        
    def test_weighted_precision(self):
        weighted_precision = metrics.precision_score(self.l1, self.l2, average = 'weighted')
        self.assertEqual(self.classification_metrics.weighted_precision(self.l1, self.l2),
                         weighted_precision,
                         "Incorrect weighted precision")
        
    def test_mapk(self):
        for i in range(len(self.l5)):
            for j in range(1, 4):
                print(f"""
                    y_true = {self.l5[i]},
                    y_pred = {self.l6[i]},
                    AP@{j} = {self.classification_metrics.apk(self.l5[i], self.l6[i], k = j)}
                    """
                     )
        self.assertEqual(self.classification_metrics.mapk(self.l5, self.l6, k = 2),
                         0.375,
                         "Incorrect mean average precision @ 2")


        
if __name__ == '__main__':
    precisions, recalls = ClassificationMetrics().plot_precision_recall()
    
    #plt.figure(figsize = (8, 8))
    #plt.plot(recalls, precisions)
    #plt.xlabel('Recalls')
    #plt.ylabel('Precisions')
    #plt.show()
    
    roc_df = ClassificationMetrics().plot_roc()
    print(roc_df)
    #plt.figure(figsize = (8, 8))
    #plt.plot(roc_df['fpr'], roc_df['tpr'])
    #plt.fill_between(roc_df['fpr'], roc_df['tpr'], alpha = 0.4)
    #plt.xlabel('FPR')
    #plt.ylabel('TPR')
    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    #plt.show();
    
    
    unittest.main()