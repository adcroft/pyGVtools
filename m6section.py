import numpy as np
import numpy.matlib

def m6section(x, z, q, representation='pcm'):
  """
  Creates the appropriate quadmesh coordinates to plot a scalar q(1:nk,1:ni) at
  horizontal positions x(1:ni+1) and between interfaces at z(nk+1,ni).

  Returns X(2*ni+1, Z(nk+1,2*ni+1) and Q(nk,2*ni) to be passed to pcolormesh.

  TBD: Optionally, x can be dimensioned as x(ni) in which case it will be extraplated as if it had 
  had dimensions x(ni+1).
  
  Optional argument:
  
  representation='pcm' (default) yields a step-wise visualization, appropriate for
           z-coordinate models.
  representation='plm' yields a piecewise-linear visualization more representative
           of general-coordinate (and isopycnal) models.
  representation='linear' is the aesthetically most pleasing but does not
           represent the data conservatively.

  """

  if x.ndim!=1: raise Exception('The x argument must be a vector')
  if z.ndim!=2: raise Exception('The z argument should be a 2D array')
  if q.ndim!=2: raise Exception('The z argument should be a 2D array')
  qnk, qni = q.shape
  znk, zni = z.shape
  xni = x.size
  if zni!=qni: raise Exception('The last dimension of z and q must be equal in length')
  if znk!=qnk+1: raise Exception('The first dimension of z must 1 longer than that of q')
  if xni!=qni+1: raise Exception('The length of x must 1 longer than the last dimension of q')

  if type( z ) == np.ma.core.MaskedArray: z[z.mask] = 0
  if type( q ) == np.ma.core.MaskedArray: q[q.mask] = 0

  if representation=='pcm':
    X = np.zeros((2*qni))
    X[::2] = x[:-1]
    X[1::2] = x[1:]
    Z = np.zeros((qnk+1,2*qni))
    Z[:,::2] = z
    Z[:,1::2] = z
    Q = np.zeros((qnk,2*qni-1))
    Q[:,::2] = q
    Q[:,1::2] = ( q[:,:-1] + q[:,1:] )/2.
  elif representation=='linear':
    X = np.zeros((2*qni+1))
    X[::2] = x
    X[1::2] = ( x[0:-1] + x[1:] )/2.
    Z = np.zeros((qnk+1,2*qni+1))
    Z[:,1::2] = z
    Z[:,2:-1:2] = ( z[:,0:-1] + z[:,1:] )/2.
    Z[:,0] = z[:,0]
    Z[:,-1] = z[:,-1]
    Q = np.zeros((qnk,2*qni))
    Q[:,::2] = q
    Q[:,1::2] = q
  elif representation=='plm':
    X = np.zeros((2*qni))
    X[::2] = x[:-1]
    X[1::2] = x[1:]
    # PLM reconstruction for Z
    d2 = 0.*z ; d2[:,1:-1] = ( z[:,2:] - z[:,:-2] )/2. # Centered slope
    s = 0.*z ; s[d2>0] = 1. ; s[d2<0] = -1. # Sign of centered slope
    dz = z[:,1:] - z[:,0:-1] # Difference on edges
    dzz = 0 *z; dzz[:,1:-1] = dz[:,1:] * dz[:,0:-1]
    s[dzz<=0] = 0 # Flatten extrema
    S = 0.*z
    S[:,:-1] = np.minimum( np.abs(d2[:,:-1]), np.abs(dz) )
    S[:,1:] = np.minimum( S[:,1:], np.abs(dz) )
    Z = np.zeros((qnk+1,2*qni))
    Z[:,::2] = z - S*s/2.
    Z[:,1::2] = z + S*s/2.
    Q = np.zeros((qnk,2*qni-1))
    Q[:,::2] = q
    Q[:,1::2] = ( q[:,:-1] + q[:,1:] )/2.
  else: raise Exception('Unknown representation!')

  return X, Z, Q


import matplotlib.pyplot as plt

# Test data
x=np.arange(5)
z=np.array([[0,0.2,0,-.1],[1,1.5,.5,.4],[2,2,1.5,2],[3,2.3,1.5,2.1]])*-1
q=np.matlib.rand(3,4)
print 'x=',x
print 'z=',z
print 'q=',q

X, Z, Q = m6section(x, z, q)
print 'X=',X
print 'Z=',Z
print 'Q=',Q
plt.subplot(3,1,1)
plt.pcolormesh(X, Z, Q)

X, Z, Q = m6section(x, z, q, representation='linear')
print 'X=',X
print 'Z=',Z
print 'Q=',Q
plt.subplot(3,1,2)
plt.pcolormesh(X, Z, Q)

X, Z, Q = m6section(x, z, q, representation='plm')
print 'X=',X
print 'Z=',Z
print 'Q=',Q
plt.subplot(3,1,3)
plt.pcolormesh(X, Z, Q)

plt.show()
