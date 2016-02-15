import os
import scipy.io.wavfile as wavfile
from numpy import *

# Load a TIMIT data set
def tset( mf = None, ff = None, dr = None):
   # Where the files are
   p = '/usr/local/timit/timit-wav/train/';

   # Pick a speaker directory
   if dr is None:
     dr = random.randint( 1, 8)
   p += 'dr%d/' % dr

   # Get two random speakers
   if mf is None:
     mf = [name for name in os.listdir( p) if name[0] == 'm']
     mf = random.choice( mf)
   if ff is None:
     ff = [name for name in os.listdir( p) if name[0] == 'f']
     ff = random.choice( ff)
   print ('dr%d/' % dr), mf, ff

   # Load all the wav files
   ms = [wavfile.read(p+mf+'/'+n)[1] for n in os.listdir( p+mf) if 'wav' in n]
   fs = [wavfile.read(p+ff+'/'+n)[1] for n in os.listdir( p+ff) if 'wav' in n]

   # Find suitable test file pair
   l1 = map( lambda x : x.shape[0], ms)
   l2 = map( lambda x : x.shape[0], fs)
   d = array( [[abs(t1-t2) for t1 in l1] for t2 in l2])
   i = argmin( d)
   l = max( [l1[i%10], l2[i/10]])
   ts = [pad( ms[i%10], (0,l-l1[i%10]), 'constant'), pad( fs[i/10], (0,l-l2[i/10]), 'constant')]

   # Get training data
   ms.pop( i%10)
   fs.pop( i/10)
   tr = [concatenate(ms), concatenate(fs)]

   return map( lambda x : (x-mean(x))/std(x), ts), map( lambda x : (x-mean(x))/std(x), tr)