import unittest
import numpy as np
from original_quota_fix import repair_integer_counts

class QuotaTests(unittest.TestCase):
    def test_regression(self):
        np.testing.assert_array_equal(repair_integer_counts([-1,1,1,1,1],3),[0,0,1,1,1])
    def test_unchanged_and_no_mutation(self):
        for a in [[0,0],[3,0,2],[0,7]]:
            original=np.array(a)
            np.testing.assert_array_equal(repair_integer_counts(original,sum(a)),original)
            np.testing.assert_array_equal(original,a)
    def test_multiple_negative(self):
        a=repair_integer_counts([-2,-1,3,3],3)
        self.assertTrue((a>=0).all()); self.assertEqual(a.sum(),3)
    def test_invalid(self):
        for a,n in [([1,2],2),([.5,.5],1),([np.nan],0),([],0),([-1],-1)]:
            with self.assertRaises(ValueError): repair_integer_counts(a,n)
    def test_random_original_rounding(self):
        rng=np.random.default_rng(42)
        for _ in range(10000):
            p=rng.dirichlet(np.ones(int(rng.integers(2,30))))
            n=int(rng.integers(0,100))
            a=np.round(p*n).astype(int); a[a.argmax()]+=n-a.sum()
            b=repair_integer_counts(a,n)
            self.assertEqual(b.sum(),n); self.assertTrue((b>=0).all())
            if (a>=0).all(): np.testing.assert_array_equal(a,b)

if __name__=='__main__': unittest.main()
