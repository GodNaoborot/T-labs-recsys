"""Проверки целочисленной формулы восстановления начала."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_start_time_formula import exact_start

class FormulaAuditTest(unittest.TestCase):
    def test_matches_floor_and_phase(self):
        length=np.repeat(np.arange(5,1005,5),151)
        ratio=np.tile(np.arange(151),200)
        end=np.full(len(length),10000)
        expected=np.floor(end-length*ratio/100-5).astype('int64')
        actual=exact_start(end,length,ratio)
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual%5,(-((length*ratio+99)//100))%5)
    def test_exact_recovery_cancels_outcome(self):
        np.testing.assert_array_equal(exact_start([120,180],[100,100],[20,80]),[95,95])
    def test_wrong_endpoint_can_encode_outcome(self):
        np.testing.assert_array_equal(exact_start([1000,1000],[100,100],[20,100]),[975,895])

if __name__=='__main__':
    unittest.main()
