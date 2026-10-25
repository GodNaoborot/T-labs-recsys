"""Свойства условной перестановки без реальных данных и моделей."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audit_start_time_phase_permutation import permute_phase

class PhasePermutationTest(unittest.TestCase):
    def test_strata_and_marginals_preserved(self):
        fresh=np.array([1,2,3,4,6,7,8,9,11,12,13,14,np.nan],dtype='float32')
        length=np.array([100]*4+[200]*4+[100]*4+[100])
        changed=permute_phase(fresh,length)
        for value,bucket in [(100,0),(200,1),(100,2)]:
            rows=(length==value)&(np.floor(fresh/5)==bucket)
            np.testing.assert_array_equal(np.sort(changed[rows]),np.sort(fresh[rows]))
        self.assertTrue(np.isnan(changed[-1]))
        np.testing.assert_array_equal(np.floor(changed[:-1]/5),np.floor(fresh[:-1]/5))
        np.testing.assert_array_equal(changed,permute_phase(fresh,length))
    def test_singletons_are_unchanged(self):
        fresh=np.array([1.,6.,11.])
        np.testing.assert_array_equal(permute_phase(fresh,np.array([1,2,3])),fresh)

if __name__=='__main__':
    unittest.main()
