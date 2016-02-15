import time
from numpy import *

eps = finfo( float32).eps;


#
# Load a set of files
#

def sound_set( tp):
    import scipy.io.wavfile as wavfile
    import paris.speech

    # Two sinusoids signal
    if tp == 1:
        l = 8*1024
        sr = 8000
        def clip0( x):
            return x * (x>0)
        z1 = clip0( sin( 3*linspace( 0, 2*pi, l))) * sin( 1099*linspace( 0, 2*pi, l))
        z2 = clip0( sin( 2*linspace( 0, 2*pi, l))) * sin( 3222*linspace( 0, 2*pi, l))
        z3 = clip0( sin( 5*linspace( 0, 2*pi, l))) * sin( 1099*linspace( 0, 2*pi, l))
        z4 = clip0( sin( 3*linspace( 0, 2*pi, l))) * sin( 3222*linspace( 0, 2*pi, l))

        z1 = hstack( (zeros(l/8),z1))
        z2 = hstack( (zeros(l/8),z2))
        z3 = hstack( (zeros(l/8),z3))
        z4 = hstack( (zeros(l/8),z4))

    # Small TIMIT/chimes set
    elif tp == 2:
        sr,z1 = wavfile.read('/usr/local/timit/timit-wav/train/dr1/mdac0/sa1.wav')
        sr,z2 = wavfile.read('/Users/paris/Desktop/Dropbox/chimes.wav')
        sr,z3 = wavfile.read('/usr/local/timit/timit-wav/train/dr1/mdac0/sa2.wav')
        z4 = z2[z1.shape[0]:]

        l = min( [z1.shape[0], z2.shape[0], z3.shape[0], z4.shape[0]])
        z1 = z1[:int(2048*floor(l/2048))]
        z2 = z2[:z1.shape[0]]
        z3 = z3[:z1.shape[0]]
        z4 = z4[:z1.shape[0]]
        z1 = z1 / std( z1)
        z2 = z2 / std( z2)
        z3 = z3 / std( z3)
        z4 = z4 / std( z4)

    # Larger TIMIT/chimes data set
    elif tp == 3:
        ts,tr = paris.speech.tset()
        sr,z = wavfile.read('/Users/paris/Desktop/Dropbox/chimes.wav')

        ts[0] = z[:ts[0].shape[0]]
        tr[0] = z[ts[0].shape[0]:]

        m = min( tr[0].shape[0], tr[1].shape[0])
        tr[0] = tr[0][:m]
        tr[1] = tr[1][:m]

        z1 = tr[1] / std( tr[1])
        z2 = tr[0] / std( tr[0])
        z3 = ts[1] / std( ts[1])
        z4 = ts[0] / std( ts[0])

    # TIMIT male/female set
    elif tp == 4:
        #ts,tr = tset( 'fpad0', 'mbma1', 6)
        ts,tr = paris.speech.tset()
        sr = 16000

        tr[0] = tr[0][:min(tr[0].shape[0],tr[1].shape[0])]
        tr[1] = tr[1][:min(tr[0].shape[0],tr[1].shape[0])]

        z1 = tr[1] / std( tr[1])
        z2 = tr[0] / std( tr[0])
        z3 = ts[1] / std( ts[1])
        z4 = ts[0] / std( ts[0])

    # Pad them
    sz = 1024

    def zp( x):
        return hstack( (zeros(sz),x[:int(sz*floor(x.shape[0]/sz))],zeros(sz)))

    tr1 = zp( z1[:int(sz*floor(z1.shape[0]/sz))])
    tr2 = zp( z2[:int(sz*floor(z2.shape[0]/sz))])
    ts1 = zp( z3[:int(sz*floor(z3.shape[0]/sz))])
    ts2 = zp( z4[:int(sz*floor(z4.shape[0]/sz))])

    # Show me
    #soundsc( ts1+ts2, sr)

    return tr1,tr2,ts1,ts2


#
# Sound feature class
#

# Sound feature class
class sound_feats:

    # Initializer
    def __init__(self, sz, hp, wn):
        import scipy.fftpack

        self.sz = sz
        self.hp = hp
        self.wn = wn

        # Forward transform definition
        self.F = scipy.fftpack.fft( identity( self.sz))

        # Inverse transform with a window
        self.iF = conj( self.wn * self.F.T)

    # Modulator definition
    def md( self, x):
        return abs( x)+eps

    # Buffer with overlap
    def buffer( self, s):
        return array( [s[i:i+self.sz] for i in arange( 0, len(s)-self.sz+1, self.hp)]).T

    # Define overlap add matrix
    def oam( self, n):
        import scipy.sparse
        ii = array( [i*self.hp+arange( self.sz) for i in arange( n)]).flatten()
        jj = array( [i*self.sz+arange( self.sz) for i in arange( n)]).flatten()
        return scipy.sparse.coo_matrix( (ones( len( ii)), (ii,jj)) ).tocsr()

    # Front end
    def fe( self, s):
        C = self.F.dot( self.wn*self.buffer( s))[:self.sz/2+1,:]
        M = self.md( C)
        P = C / M
        return (M,P)

    # Inverse transform
    def ife( self, M, P):
        oa = self.oam( M.shape[1])
        f = vstack( (M*P,conj(M*P)[-2:0:-1,:]))
        return oa.dot( reshape( real( self.iF.dot( f)), (-1,1), order='F')).flatten()


#
# NMF separation
#

# Define Polya Urn function
def pu( x, r, ep, b, sp):

    # Constants
    m = x.shape[0]
    n = x.shape[1]

    # Normalize input
    g = sum( x, axis=0)+eps
    x /= g

    # Learn or fit?
    if isscalar( r):
        w = random.rand( m, r)+10
        w /= sum( w, axis=0)
        lw = True
    else:
        w = hstack( r)
        r = w.shape[1]
        lw = False

    # Init activations
    h = random.rand( r, n)+10
    h /= sum( h, axis=0)+eps

    # Start churning
    for e in arange( ep):
        # Get tentative estimate
        v = x / (w.dot( h)+eps)
        if lw:
            nw = w * v.dot( h.T)
        nh = h * w.T.dot( v)
        
        # Sparsity
        if sp > 0:
            nh = nh + b*nh**(1+sp)

        # Get estimate and normalize
        if lw:
            w = nw / (sum( nw, axis=0) + eps)
        h = nh / (sum( nh, axis=0) + eps)
        
    h *= g
    return (w,h)



# Learn models and separate
def nmf_sep( Z, FE, K, s = None):
    from paris.signal import bss_eval
  
    if s is not None:
        random.seed( s)

    # Get features
    M1,P1 = FE.fe( Z[0])
    M2,P2 = FE.fe( Z[1])
    MT,PT = FE.fe( Z[2]+Z[3])

    # Overcomplete or not?
    t0 = time.time()
    if 1:
        w1,_ = pu( copy(M1), K[0], 300, 0, 0)
        w2,_ = pu( copy(M2), K[1], 300, 0, 0)
        w1 /= sum( w1, axis=0, keepdims=True)
        w2 /= sum( w2, axis=0, keepdims=True)
        w = (w1,w2)
        sp = [0,0]
    else:
        # Get overcomplete bases
        w = [M1 / sum( M1, axis=0), M2 / sum( M2, axis=0)]
        sp = [.5,1]

    # Fit 'em on mixture
    t1 = time.time()
    _,h = pu( copy( MT), w, 300, sp[0], sp[1])
    print 'Done in', time.time()-t0, time.time()-t1, 'sec'

    # Get modulator estimates
    q = cumsum( [0, w[0].shape[1], w[1].shape[1]])
    fr = [w[i].dot( h[q[i]:q[i+1],:]) for i in arange( 2)]
    fr0 = hstack(w).dot(h)+eps

    # Resynth with Wiener filtering
    r = [FE.ife( fr[0] * (MT/fr0), PT),
         FE.ife( fr[1] * (MT/fr0), PT)]
    #r = [FE.ife( fr[0], PT),
    #     FE.ife( fr[1], PT)]

    # Get results
    sxr = array( [bss_eval( r[i], i, vstack((Z[2],Z[3]))) for i in arange( len( r))])

    return mean( sxr, axis=0),r


#
# All-in-ont NMF separation
#

def nmf_run( K, sz = 1024, hp = None, s = 0):
    
    # Load sound set
    random.seed( s)
    Z = sound_set(4)

    # Front-end details
    if hp is None:
      hp = sz/4
    wn = reshape( hanning(sz+1)[:-1], (sz,1))**.5

    # Make feature class
    FE = sound_feats( sz, hp, wn)

    # Separate
    sxr,r = nmf_sep( Z, FE, K)

    return sxr


#
# Learn NN model of a sound using lasagne
#

import theano
import theano.tensor as Th

import downhill

# Training loop
def downhill_train( opt, train, hh, ep, pl):
    cst = []
    st = time.time()
    lt = st
    try:
        for tm,_ in opt.iterate( train, learning_rate=hh, max_updates=ep, patience=ep, min_improvement=0):
            cst.append( tm['loss'])
            if time.time() - lt > 4 and pl is not None:
                nt = time.time()
                epc = len( cst)
                clf()
                pl()
                semilogy( cst), grid( 'on')
                title( 'Cost: %.1e  Speed: %.2f ep/s  Time: %.1f/%.1f' %
                  (cst[-1], epc/(nt-st), nt-st, ep/(epc/(nt-st))) )
                ylabel( 'Cost')
                drawnow()
                lt = time.time()
    except KeyboardInterrupt:
        pass

    if pl is not None:
        clf()
        nt = time.time()
        epc = len( cst)
        pl()
        semilogy( cst), grid( 'on')
        title( 'Cost: %.1e  Speed: %.2f ep/s  Time: %.1f/%.1f' %
            (cst[-1], epc/(nt-st), nt-st, ep/(epc/(nt-st))) )
        ylabel( 'Cost')
        drawnow()
    return cst


# Parameterized softplus
def psoftplus( x, p = 1.):
    return Th.switch( x < -30./p, 0., Th.switch( x > 30./p, x, Th.log1p( Th.exp( p*x))/p))

from lasagne.layers  import InputLayer, DenseLayer, DropoutLayer, ElemwiseSumLayer
from lasagne.layers import SliceLayer, get_output, get_all_params
from lasagne.updates import adam

# Get a Lasagne layer output
def nget( x, s, y):
    return theano.function( [s], squeeze( get_output( x, deterministic=True)))( y)

# Learn models using a Lasagne network
def lasagne_models( M, P, FE, z, K = 20, hh = .0001, ep = 5000, d = 0, wsp = 0.0001, plt = True):
    from paris.signal import bss_eval
    
    # Copy key variables to GPU
    _M = Th.matrix( '_M')

    # Input and forward transform
    I = InputLayer( shape=M.T.shape, input_var=_M)

    # First layer is the transform to a non-negative subspace
    H0  = DenseLayer( I, num_units=K, nonlinearity=lambda x: psoftplus( x, 3.), b=None)

    # Optional dropout
    H = DropoutLayer( H0, d)

    # Compute source modulator
    R  = DenseLayer( H, num_units=M.T.shape[1], nonlinearity=lambda x: psoftplus( x, 3.), b=None)

    # Cost function
    cost = (_M*(Th.log(_M+eps) - Th.log( get_output( R)+eps)) - _M + get_output( R)).mean() \
       + wsp*Th.mean( abs( R.W))

    # Train it using Lasagne
    opt = downhill.build( 'rprop', loss=cost, inputs=[_M], params=get_all_params( R))
    train = downhill.Dataset( M.T.astype(float32), batch_size=0)
    er = downhill_train( opt, train, hh, ep, None)[-1]

    # Get approximation
    _r = nget( R, _M, M.T.astype( float32)).T
    _h = nget( H, _M, M.T.astype( float32)).T
    o = FE.ife( _r, P)
    sxr = bss_eval( o, 0, array([z]))

    return R,sxr

#
# Separate mixture given NN models
#

# Lasagne separate
def lasagne_separate( M, P, FE, W1, W2, z1, z2, hh = .0001, ep = 5000, d = 0, wsp =.0001, plt = True):
    from paris.signal import bss_eval

    # Gt dictionary shapes
    K = [W1.shape[0],W2.shape[0]]

    # GPU cached data
    _M = theano.shared( M.astype( float32))

    # Input is the learned dictionary set
    lW = hstack( (W1.T,W2.T)).astype( float32)
    _lW  = Th.matrix( '_lW');
    fI = InputLayer( shape=lW.shape, input_var=_lW)

    # Split in two paths
    fW1 = SliceLayer( fI, indices=slice(0,K[0]), axis=1)
    fW2 = SliceLayer( fI, indices=slice(K[0],K[0]+K[1]), axis=1)

    # Dropout?
    dfW1 = DropoutLayer( fW1, d)
    dfW2 = DropoutLayer( fW2, d)

    # Compute source modulators
    R1  = DenseLayer( dfW1, num_units=M.shape[1], nonlinearity=lambda x: psoftplus( x, 3.), b=None)
    R2  = DenseLayer( dfW2, num_units=M.shape[1], nonlinearity=lambda x: psoftplus( x, 3.), b=None)

    # Bring to standard orientation
    R = ElemwiseSumLayer( [R1, R2])

    # Cost function
    cost = (_M*(Th.log(_M+eps) - Th.log( get_output( R)+eps)) - _M + get_output( R)).mean() \
       + wsp*(Th.mean( abs( R1.W))+Th.mean( abs( R2.W)))

    # Train it using Lasagne
    opt = downhill.build( 'rprop', loss=cost, inputs=[_lW], params=get_all_params( R))
    train = downhill.Dataset( lW, batch_size=0)
    er = downhill_train( opt, train, hh, ep, None)[-1]

    # Get outputs
    _r  = nget( R,  _lW, lW) + eps
    _r1 = nget( R1, _lW, lW)
    _r2 = nget( R2, _lW, lW)
    o1 = FE.ife( _r1 * (M/_r), P)
    o2 = FE.ife( _r2 * (M/_r), P)
    sxr = bss_eval( o1, 0, vstack( (z1,z2))) + bss_eval( o2, 1, vstack( (z1,z2)))

    return o1,o2,(array(sxr[:3]) + array(sxr[3:]))/2.

# Lasagne separate
def lasagne_separate2( M, P, FE, W1, W2, z1, z2, hh = .0001, ep = 5000, d = 0, wsp =.0001, plt = True):
    from paris.signal import bss_eval

    # Gt dictionary shapes
    K = [W1.shape[0],W2.shape[0]]

    # GPU cached data
    _M = theano.shared( M.T.astype( float32))
    dum = Th.vector( 'dum')

    # We have weights to discover
    H = theano.shared( random.rand( M.T.shape[0],K[0]+K[1]).astype( float32))
    fI = InputLayer( shape=(M.T.shape[0],K[0]+K[1]), input_var=H)

    # Split in two pathways
    fW1 = SliceLayer( fI, indices=slice(0,K[0]), axis=1)
    fW2 = SliceLayer( fI, indices=slice(K[0],K[0]+K[1]), axis=1)

    # Dropout?
    dfW1 = DropoutLayer( fW1, d)
    dfW2 = DropoutLayer( fW2, d)

    # Compute source modulators using previously learned dictionaries
    R1  = DenseLayer( dfW1, num_units=M.shape[0], W=W1.astype( float32),
      nonlinearity=lambda x: psoftplus( x, 3.), b=None)
    R2  = DenseLayer( dfW2, num_units=M.shape[0], W=W2.astype( float32),
      nonlinearity=lambda x: psoftplus( x, 3.), b=None)

    # Add the two approximations
    R = ElemwiseSumLayer( [R1, R2])

    # Cost function
    cost = (_M*(Th.log(_M+eps) - Th.log( get_output( R)+eps)) - _M + get_output( R)).mean() \
       + wsp*Th.mean( H) + 0*Th.mean( dum)

    # Train it using Lasagne
    opt = downhill.build( 'rprop', loss=cost, inputs=[dum], params=[H])
    train = downhill.Dataset( array( [0]).astype(float32), batch_size=0)
    er = downhill_train( opt, train, hh, ep, None)[-1]

    # Get outputs
    _r  = nget( R,  dum, array( [0]).astype(float32)) + eps
    _r1 = nget( R1, dum, array( [0]).astype(float32))
    _r2 = nget( R2, dum, array( [0]).astype(float32))
    o1 = FE.ife( _r1 * (M/_r), P)
    o2 = FE.ife( _r2 * (M/_r), P)
    sxr = bss_eval( o1, 0, vstack( (z1,z2))) + bss_eval( o2, 1, vstack( (z1,z2)))

    return o1,o2,(array(sxr[:3]) + array(sxr[3:]))/2.


#
# Learn NN models using Theano directly
#

# Learn models using a Theano network
def downhill_models( M, P, FE, z, K = 20, hh = .001, ep = 5000, dp = 0, wsp = .001, plt = False):
    from paris.signal import bss_eval

    rng = theano.tensor.shared_randomstreams.RandomStreams(0)

    # Shared variables to use
    x = Th.matrix('x')
    y = theano.shared( M.astype( theano.config.floatX))
    d = theano.shared( float32( dp))

    # Network weights
    W0 = theano.shared( sqrt( 2./(K+M.shape[0]))*random.randn( K, M.shape[0]).astype( theano.config.floatX))
    W1 = theano.shared( sqrt( 2./(K+M.shape[0]))*random.randn( M.shape[0], K).astype( theano.config.floatX))

    # First layer is the transform to a non-negative subspace
    h = psoftplus( W0.dot( x), 3.)

    # Dropout
    if dp > 0:
        h *= (1. / (1. - d) * (rng.uniform(size=h.shape) > d).astype( theano.config.floatX)).astype( theano.config.floatX)

    # Second layer reconstructs the input
    r = psoftplus( W1.dot( h), 3.)

    # Approximate input using KL-like distance
    cost = Th.mean( y * (Th.log( y+eps) - Th.log( r+eps)) - y + r) + wsp*Th.mean( abs( W1))

    # Make an optimizer and define the training input
    opt = downhill.build( 'rprop', loss=cost, inputs=[x], params=[W0,W1])
    train = downhill.Dataset( M.astype( theano.config.floatX), batch_size=0)

    # Train it
    downhill_train( opt, train, hh, ep, None)

    # Get approximation
    d = 0
    _,_r = theano.function( inputs = [x], outputs = [h,r], updates = [])( M.astype( theano.config.floatX))
    o = FE.ife( _r, P)
    sxr = bss_eval( o, 0, array([z]))

    return W1.get_value(),sxr


# Separate using downhill
def downhill_separate( M, P, FE, W1, W2, z1, z2, hh = .001, ep = 5000, d = 0, wsp =.0001, plt = True):
    from paris.signal import bss_eval

    # Get dictionary sizes
    K = [W1.shape[1],W2.shape[1]]

    # Cache some things
    y = Th.matrix('y')
    w1 = theano.shared( W1.astype( theano.config.floatX), 'w1')
    w2 = theano.shared( W2.astype( theano.config.floatX), 'w2')

    # Activations to learn
    h1 = theano.shared( sqrt( 2./(K[0]+M.shape[1]))*random.randn( K[0], M.shape[1]).astype( theano.config.floatX))
    h2 = theano.shared( sqrt( 2./(K[1]+M.shape[1]))*random.randn( K[1], M.shape[1]).astype( theano.config.floatX))

    # Dropout
    if d > 0:
        dw1 = w1 * 1./(1.-d) * (rng.uniform(size=w1.shape) > d).astype( theano.config.floatX)
        dw2 = w2 * 1./(1.-d) * (rng.uniform(size=w2.shape) > d).astype( theano.config.floatX)
    else:
        dw1 = w1
        dw2 = w2

    # Approximate input
    r1 = psoftplus( dw1.dot( h1), 3.)
    r2 = psoftplus( dw2.dot( h2), 3.)
    r = r1 + r2

    # KL-distance to input
    cost = Th.mean( y * (Th.log( y+eps) - Th.log( r+eps)) - y + r) \
       + wsp*(Th.mean( abs( h1)) + Th.mean( abs( h2)))

    # Make it callable and derive updates
    ffwd_f = theano.function( inputs = [], outputs = [r1,r2,h1,h2], updates = [])

    # Make an optimizer and define the inputs
    opt = downhill.build( 'rprop', loss=cost, inputs=[y], params=[h1,h2])
    train = downhill.Dataset( M.astype( theano.config.floatX), batch_size=0)

    # Train it
    cst = downhill_train( opt, train, hh, ep, None)

    # So what happened?
    d = 0
    _r1,_r2,_h1,_h2 = ffwd_f()
    _r = _r1 + _r2 + eps
    o1 = FE.ife( _r1 * (M/_r), P)
    o2 = FE.ife( _r2 * (M/_r), P)
    sxr = bss_eval( o1, 0, vstack( (z1,z2))) + bss_eval( o2, 1, vstack( (z1,z2)))

    # Return things of note
    return o1,o2,(array(sxr[:3]) + array(sxr[3:]))/2.


#
# All in one NN separation
#

def nn_run( K, sz = 1024, hp = None, s = 0):
    from deep_sep_expr import sound_set, sound_feats, lasagne_models, lasagne_separate
    l = True

    # Load sound set
    random.seed( s)
    Z = sound_set(4)

    # Front-end details
    if hp is None:
      hp = sz/4
    wn = reshape( hanning(sz+1)[:-1], (sz,1))**.5

    # Make feature class
    FE = sound_feats( sz, hp, wn)

    # Learn models
    M,P = FE.fe( Z[0])
    if l:
      n1,c1 = lasagne_models( M, P, FE, Z[0], K[0], .01, 1000, 0.25, .0001, False)
    else:
      n1,c1 = downhill_models( M, P, FE, Z[0], K[0], .01, 1000, 0.25, .0001, False)
    print array( c1)

    M,P = FE.fe( Z[1])
    if l:
      print 'Using Lasagne'
      n2,c2 = lasagne_models( M, P, FE, Z[1], K[1], .01, 1000, 0.25, .0001, False)
    else:
      print 'Using Theano'
      n2,c2 = downhill_models( M, P, FE, Z[1], K[1], .01, 1000, 0.25, .0001, False)
    print array( c2)

    # Separate
    M,P = FE.fe( Z[2]+Z[3])
    if l:
      o1,o2,sxr = lasagne_separate( M, P, FE, n1.W.get_value(), n2.W.get_value(), \
                                 Z[2], Z[3], .0001, 2000, 0, 0.01, False)
    else:
      o1,o2,sxr = downhill_separate( M, P, FE, n1, n2, \
                                 Z[2], Z[3], .0001, 2000, 0, 0.01, False)

    return sxr
