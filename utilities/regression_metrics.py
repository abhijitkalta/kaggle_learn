import unittest
import numpy as np
import pandas as pd
from sklearn import metrics

class RegressionMetrics:
    def mean_absolute_error(self, y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            absolute_error = np.absolute(yt - yp)
            error += absolute_error
        
        return error / len(y_true)
    
    def mean_squared_error(self, y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            squared_error = (yt - yp) ** 2
            error += squared_error
        
        return error / (len(y_true))
    
    def root_mean_sqaured_error(self, y_true, y_pred):
        return np.sqrt(self.mean_squared_error(y_true, y_pred))
                        
    def mean_squared_logarithmic_error(self, y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            msle = (np.log(yt + 1) - np.log(yp + 1)) ** 2
            error += msle
        
        return error / len(y_true)
    
    def mean_percentage_error(self, y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            mpe = (yt - yp) / yt
            error += mpe
        
        return error / len(y_true)
    
    def mean_abs_percentage_error(self, y_true, y_pred):
        error = 0
        for yt, yp in zip(y_true, y_pred):
            mape = np.abs(yt - yp)/ yt
            error += mape
        
        return error / len(y_true)
    
    def r2(self, y_true, y_pred):
        y_mean = np.mean(y_true)
        
        numerator = 0
        denominator = 0
        
        for yt, yp in zip(y_true, y_pred):
            numerator += (yt - yp) ** 2
            denominator += (yt - y_mean) ** 2
        
        ratio = numerator / denominator
        return 1 - ratio
                        
class TestRegressionMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.l1 = np.random.randint(0, 100, 50)
        self.l2 = self.l1 + np.random.randint(0, 100, 50)
        self.regression_metrics = RegressionMetrics()
        
    def test_mean_absolute_error(self):
        mae = np.round(metrics.mean_absolute_error(self.l1, self.l2), 6)
        self.assertEqual(np.round(self.regression_metrics.mean_absolute_error(self.l1, self.l2), 6),
                         mae,
                         "Incorrect mean absolute error"
                        )
    def test_mean_sqaured_error(self):
        mse = np.round(metrics.mean_squared_error(self.l1, self.l2), 6)
        self.assertEqual(np.round(self.regression_metrics.mean_squared_error(self.l1, self.l2), 6),
                         mse,
                         "Incorrect mean sqaured error"
                        )
    
    def test_mean_squared_log_error(self):
        msle = np.round(metrics.mean_squared_log_error(self.l1, self.l2), 6)
        self.assertEqual(np.round(self.regression_metrics.mean_squared_logarithmic_error(self.l1,                                self.l2), 6),
                         msle,
                         "Incorrect mean sqaured log error"
                        )
    def test_r2_score(self):
        r2_score = np.round(metrics.r2_score(self.l1, self.l2), 6)
        self.assertEqual(np.round(self.regression_metrics.r2(self.l1, self.l2), 6),
                         r2_score,
                         "Incorrect r2 score"
                        )
         

if __name__ == '__main__':
    unittest.main()