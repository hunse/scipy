"""LDL decomposition functions."""

from __future__ import division, print_function, absolute_import

from numpy import asarray_chkfinite, asarray

# Local imports
from .misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs

__all__ = ['ldl_factor', 'ldl_solve']


def ldl_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    TODO
    """

    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')

    overwrite_a = overwrite_a or _datacopied(a1, a)
    sytrf, = get_lapack_funcs(('sytrf',), (a1,))
    c, ipiv, info = sytrf(a1, lower=lower, overwrite_a=overwrite_a)
    if info > 0:
        raise LinAlgError("%d-th leading minor not positive definite" % info)
    if info < 0:
        raise ValueError(
            'illegal value in %d-th argument of internal sytrf' % -info)
    return c, ipiv, lower


def ldl_solve(c_ipiv_lower, b, overwrite_b=False, check_finite=True):
    """
    TODO
    """

    (c, ipiv, lower) = c_ipiv_lower
    if check_finite:
        b1 = asarray_chkfinite(b)
        c = asarray_chkfinite(c)
    else:
        b1 = asarray(b)
        c = asarray(c)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("The factored matrix c is not square.")
    if c.shape[1] != b1.shape[0] or c.shape[0] != ipiv.size:
        raise ValueError("incompatible dimensions.")

    overwrite_b = overwrite_b or _datacopied(b1, b)

    sytrs, = get_lapack_funcs(('sytrs',), (c, b1))
    x, info = sytrs(c, ipiv, b1, lower=lower, overwrite_b=overwrite_b)
    if info != 0:
        raise ValueError(
            'illegal value in %d-th argument of internal sytrs' % -info)
    return x

