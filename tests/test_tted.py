import math
import os
import unittest
import torch
import torch.nn.functional as F
from ted import logmeanexp_support, text_axis, tted_score


class TTEDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.p = torch.randn(17, 8)
        self.d = F.normalize(torch.randn(9, 8), dim=-1)
        self.h = F.normalize(torch.randn(11, 8), dim=-1)
        self.u = text_axis(torch.randn(8), torch.randn(8))

    def test_dense_reference(self):
        c = F.normalize(self.p.float(), dim=-1) @ self.u
        def support(b):
            return torch.logsumexp(-(c[:, None] - (b @ self.u)[None, :])**2 / .05, -1) - math.log(len(b))
        expected = support(self.d) - .7 * support(self.h)
        for chunk in (1, 4, 1024):
            torch.testing.assert_close(tted_score(self.p,self.d,self.h,self.u,tau=.05,fp_weight=.7,chunk_size=chunk),expected)

    def test_role_swap(self):
        a=tted_score(self.p,self.d,self.h,self.u,tau=.05)
        b=tted_score(self.p,self.h,self.d,self.u,tau=.05)
        torch.testing.assert_close(a,-b)

    def test_identical_banks(self):
        torch.testing.assert_close(tted_score(self.p,self.d,self.d,self.u,tau=.05),torch.zeros(17))

    def test_empty_inputs(self):
        self.assertEqual(tted_score(self.p[:0],self.d,self.h,self.u,tau=.05).shape,(0,))
        torch.testing.assert_close(logmeanexp_support(torch.ones(3),torch.empty(0),tau=.05),torch.zeros(3))

    def test_bank_permutation(self):
        torch.testing.assert_close(tted_score(self.p,self.d,self.h,self.u,tau=.05),tted_score(self.p,self.d.flip(0),self.h.flip(0),self.u,tau=.05))

    def test_axis_reprojection(self):
        v=text_axis(torch.randn(8),torch.randn(8))
        a=tted_score(self.p,self.d,self.h,self.u,tau=.05)
        b=tted_score(self.p,self.d,self.h,v,tau=.05)
        self.assertFalse(torch.allclose(a,b))

    def test_invalid_inputs(self):
        for tau in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):
                tted_score(self.p,self.d,self.h,self.u,tau=tau)
        with self.assertRaises(ValueError):
            text_axis(torch.ones(8),torch.ones(8))
        with self.assertRaises(ValueError):
            tted_score(self.p,self.d,self.h,self.u*2,tau=.05)
        with self.assertRaises(ValueError):
            tted_score(self.p,self.d[:,:7],self.h,self.u,tau=.05)

    @unittest.skipUnless(os.environ.get("TED_TEST_CUDA") == "1" and torch.cuda.is_available(),"Set TED_TEST_CUDA=1 to opt into GPU validation")
    def test_cuda(self):
        cpu=tted_score(self.p,self.d,self.h,self.u,tau=.05)
        gpu=tted_score(self.p.cuda(),self.d,self.h,self.u.cuda(),tau=.05).cpu()
        torch.testing.assert_close(cpu,gpu,atol=2e-5,rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
