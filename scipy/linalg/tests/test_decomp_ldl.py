
from __future__ import division, print_function, absolute_import

from numpy.testing import TestCase, assert_array_almost_equal

from numpy import array, transpose, dot, conjugate, zeros_like, fill_diagonal
from numpy.random import rand, randn
from scipy.linalg import ldl_factor, ldl_solve
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg import cho_factor, cho_solve


def normal(size):
    return randn(*size)


class TestLDL(TestCase):

    def test_versus_lu(self):
        n, nrhs = 5, 3
        a = normal((n, n))
        b = normal((n, nrhs))
        a = a + a.T

        c_ipiv_lower = ldl_factor(a)
        x_ldl = ldl_solve(c_ipiv_lower, b)
        x_lu = lu_solve(lu_factor(a), b)
        assert_array_almost_equal(x_ldl, x_lu)

    def test_speed_versus_cho(self):
        from time import time

        n, nrhs = 5000, 1
        sigma = 1
        a = normal((n, n))
        b = normal((n, nrhs))
        a = dot(a.T, a)
        fill_diagonal(a, a.diagonal() + n * sigma**2)

        t_cho = time()
        f_cho = cho_factor(a)
        t_cho = time() - t_cho
        x_cho = cho_solve(f_cho, b)

        t_ldl = time()
        f_ldl = ldl_factor(a)
        t_ldl = time() - t_ldl
        x_ldl = ldl_solve(f_ldl, b)

        print("TIMES:", t_cho, t_ldl)

        assert_array_almost_equal(x_cho, x_ldl)
