

from numpy import *

import theano
import theano.tensor as Th
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, ElemwiseSumLayer
from lasagne.layers import SliceLayer, get_output, get_all_params
from lasagne.layers import LSTMLayer
import downhill

eps = finfo(float32).eps;

def lasagne_separate(M, P, FE, W1, W2, z1, z2, hh=.0001, ep=5000, d=0, wsp=.0001, plt=True):
    # Gt dictionary shapes
    K = [W1.shape[0], W2.shape[0]]

    # GPU cached data
    _M = theano.shared(M.astype(float32))

    # Input is the learned dictionary set
    lW = hstack((W1.T, W2.T)).astype(float32)
    _lW = Th.matrix('_lW');
    fI = InputLayer(shape=lW.shape, input_var=_lW)

    # Split in two paths
    fW1 = SliceLayer(fI, indices=slice(0, K[0]), axis=1)
    fW2 = SliceLayer(fI, indices=slice(K[0], K[0] + K[1]), axis=1)

    # Dropout?
    dfW1 = DropoutLayer(fW1, d)
    dfW2 = DropoutLayer(fW2, d)

    N_sequence = 10
    # # Compute source modulators
    # R1 = LSTMLayer(dfW1, N_sequence)
    # R2 = LSTMLayer(dfW2, N_sequence)
    
    # Bring to standard orientation
    R = ElemwiseSumLayer([R1, R2])

    # Cost function
    cost = (_M * (Th.log(_M + eps) - Th.log(get_output(R) + eps)) - _M +
            get_output(R)).mean() + wsp * (Th.mean(abs(R1.W)) + Th.mean(abs(R2.W)))

    # Train it using Lasagne
    opt = downhill.build('rprop', loss=cost, inputs=[_lW], params=get_all_params(R))
    train = downhill.Dataset(lW, batch_size=0)
    er = downhill_train(opt, train, hh, ep, None)[-1]

    # Get outputs
    _r = nget(R, _lW, lW) + eps
    _r1 = nget(R1, _lW, lW)
    _r2 = nget(R2, _lW, lW)
    o1 = FE.ife(_r1 * (M / _r), P)
    o2 = FE.ife(_r2 * (M / _r), P)
    sxr = bss_eval(o1, 0, vstack((z1, z2))) + bss_eval(o2, 1, vstack((z1, z2)))

    return o1, o2, (array(sxr[:3]) + array(sxr[3:])) / 2.