import cmath
import numpy
import scikits.audiolab as audio
import matplotlib.pyplot as plt
import random

def memoize(f):
   cache = {}

   def memoizedFunction(*args):
      if args not in cache:
         cache[args] = f(*args)
      return cache[args]

   memoizedFunction.cache = cache
   return memoizedFunction

@memoize
def omega(p, q):
   return cmath.exp((2.0 * cmath.pi * 1j * q) / p)

def pad(inputList):
   k = 0
   while 2**k < len(inputList):
      k += 1
   return numpy.concatenate((inputList, ([0] * (2**k - len(inputList)))))

def fft(signal):
   n = len(signal)
   if n == 1:
      return signal
   else:
      Feven = fft([signal[i] for i in xrange(0, n, 2)])
      Fodd = fft([signal[i] for i in xrange(1, n, 2)])

      combined = [0] * n
      for m in xrange(n/2):
         combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
         combined[m + n/2] = Feven[m] - omega(n, -m) * Fodd[m]

      return combined

def ifft(signal):
   timeSignal = fft([x.conjugate() for x in signal])
   return [x.conjugate()/len(signal) for x in timeSignal]

# delta = [1, 0, 0, 0, 0, 0, 0, 0] # unshifted delta of length 8
# deltaShift = [0, 1, 0, 0, 0, 0, 0, 0] # unshifted delta of length 8


norm = lambda x: cmath.polar(x)[0]

# messing with audio

def frequencyFilter(signal):
   for i in range(20000, len(signal)-20000):
      signal[i] = 0


def processWithNumpy(signal):
   transformedSignal = numpy.fft.fft(signal)
   frequencyFilter(transformedSignal)

   cleanedSignal = numpy.fft.ifft(transformedSignal)
   return numpy.array(cleanedSignal, dtype=numpy.float64)

def processWithOurFFT(signal):
   # this one is significantly slower, but not unreasonable
   transformedSignal = numpy.array(fft(pad(signal)))
   frequencyFilter(transformedSignal)

   cleanedSignal = ifft(transformedSignal)
   return numpy.array(cleanedSignal, dtype=numpy.float64)

# put this code in one of the two functions above to graph the transformed signal
# plt.plot([norm(x) for x in transformedSignal])
# plt.show() # blocks the program execution until the window is closed


(inputSignal, samplingRate, bits) = audio.wavread('no_tree_ent.wav')
inputSignal = numpy.array([x/2.0 + random.random()*0.1 for x in inputSignal])

noisyFile = audio.Sndfile('no_tree_noisy.wav', 'w', audio.Format('wav'), 1, samplingRate)
noisyFile.write_frames(inputSignal)
noisyFile.close()

outputSignal = processWithNumpy(inputSignal)

outputFile = audio.Sndfile('no_tree_transformed.wav', 'w', audio.Format('wav'), 1, samplingRate)
outputFile.write_frames(outputSignal)
outputFile.close()

