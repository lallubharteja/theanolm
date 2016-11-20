#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import theano

def random_weight(shape, scale=None):
    """Generates a weight matrix from “standard normal” distribution.

    :type shape: tuple of ints
    :param shape: size of each dimension (typically there are two dimensions,
                  input and output)

    :type scale: float
    :param scale: if other than None, the random numbers will be scaled by this
                  factor

    :rtype: numpy.ndarray
    :returns: the generated weight matrix
    """

    result = numpy.random.randn(*shape)
    if scale is not None:
        result = scale * result
    return result.astype(theano.config.floatX)

def orthogonal_weight(in_size, out_size, scale=None):
    """Generates a weight matrix from “standard normal” distribution. If in_size
    matches out_size, generates an orthogonal matrix.

    :type in_size: int
    :param in_size: size of the input dimension of the weight

    :type out_size: int
    :param out_size: size of the output dimension of the weight

    :type scale: float
    :param scale: if other than None, the matrix will be scaled by this factor,
                  unless an orthogonal matrix is created
    """

    if in_size != out_size:
        return random_weight((in_size, out_size), scale)

    nonorthogonal_matrix = numpy.random.randn(in_size, out_size)
    result, _, _ = numpy.linalg.svd(nonorthogonal_matrix)
    return result.astype(theano.config.floatX)

def test_value(size, high):
    """Creates a matrix of random numbers that can be used as a test value for a
    parameter to enable debugging Theano errors.

    The type of ``high`` defines the type of the returned array. For integers,
    the range does not include the maximum value. If ``high`` is a boolean,
    returns an int8 array, as Theano uses int8 to represent a boolean.

    :type size: int or tuple of ints
    :param size: dimensions of the matrix

    :type high: int, float, or bool
    :param high: maximum value for the generated random numbers

    :rtype: numpy.ndarray
    :returns: a matrix or vector containing the generated values
    """

    if isinstance(high, bool):
        return numpy.random.randint(0, int(high), size=size).astype('int8')
    elif isinstance(high, (int, numpy.int32, numpy.int64)):
        return numpy.random.randint(0, high, size=size).astype('int64')
    elif isinstance(high, (float, numpy.float32, numpy.float64)):
        return high * numpy.random.rand(*size).astype(theano.config.floatX)
    else:
        raise TypeError("High value should be int, float, or bool.")

def get_submatrix(matrices, index, size, end_index=None):
    """Returns a submatrix of a concatenation of 2 or 3 dimensional
    matrices.

    :type matrices: TensorVariable
    :param matrices: symbolic 2 or 3 dimensional matrix formed by
                     concatenating matrices of length size

    :type index: int
    :param index: index of the matrix to be returned

    :type size: TensorVariable
    :param size: size of the last dimension of one submatrix

    :type end_index: int
    :param end_index: if set to other than None, returns a concatenation of all
                      the submatrices from ``index`` to ``end_index``
    """

    if end_index is None:
        end_index = index
    start = index * size
    end = (end_index + 1) * size
    if matrices.ndim == 3:
        return matrices[:, :, start:end]
    elif matrices.ndim == 2:
        return matrices[:, start:end]
    else:
        raise ValueError("get_submatrix() requires a 2 or 3 dimensional matrix.")
