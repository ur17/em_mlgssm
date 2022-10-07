from numpy.core import asarray
from numpy.core import newaxis
from numpy.core import sqrt
from numpy.core import multiply
from numpy.core import amax
from numpy.core import matmul
from numpy.core import divide
from numpy.core import dot

from numpy.lib import diag
from numpy.linalg import svd
from numpy.linalg import eigh

from numpy.linalg.linalg import _makearray
from numpy.linalg.linalg import transpose



def diagonalization(a):
    w, v = eigh(dot(a.T, a))

    w = w[::-1]; v = v[:,::-1]
    s = sqrt(w)
    u = dot(a, v); u = dot(u, diag(s**(-1)))
    vt = v.T

    return u, s, vt


def pseudo_inverse(a, rcond=1e-15, hermitian=False):
    a, wrap = _makearray(a)
    rcond = asarray(rcond)
    a = a.conjugate()

    try:
        u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)
        cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)
        large = s > cutoff
        s = divide(1, s, where=large, out=s)
        s[~large] = 0
        res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
        return wrap(res)

    except:
        u, s, vt = diagonalization(a)
        cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)
        large = s > cutoff
        s = divide(1, s, where=large, out=s)
        s[~large] = 0
        res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
        return wrap(res)