import numpy as np
import numpy.matlib


def axisAverage(q, z=None, h=None, area=None, mask=None, axis=-1):
  """
  Calculates the average of scalar q along an axis with appropriate volume
  weighting using either interface positions or level thickness. If provided,
  column surface area and a mask can be included in the volume weighting.

  z and h are exlcusive arguments but one or the other must be provided.

  The default axis to average is the last axis.
  """

  # Reconcile absence of either z or h since we will need both
  if not z==None:
    if not h==None: raise Exception('Both z and h have been provided. Only one of h or z is required.')
    if len(z.shape)<3: raise Exception('The interface heights, z, must be at leeast 3-dimensional')
    h = - np.diff(z, n=1, axis=-3)
    if q.shape != h.shape: raise Exception('z must have one extra level than q but otherwise have the same shape')
  else:
    if h==None: raise Exception('Neither z nor h have been provided. One of h or z is required.')
    if len(h.shape)<3: raise Exception('The level thicknesses, h, must be at leeast 3-dimensional')
    if q.shape != h.shape: raise Exception('q and h must have the same shape')
    z = - np.cumsum(h, axis=-3)

  #if len(z.shape)==3: zCenter = z[:-1,:,:] - z[1:,:,:]
  #elif len(z.shape)==4: zCenter = z[:,:-1,:,:] - z[:,1:,:,:]
  #else: raise('Higher than 4-dimensions not handled yet')

  qShape = list(q.shape)

  if area==None: weight2d = np.ones((qShape[-2:]))
  else: weight2d = area
  if not mask==None: weight2d = weight2d * mask
  nk = qShape[-3]
  hShape = list(h.shape)
  zShape = list(z.shape)

  zc = ( z[:-1,:] + z[1:,:] ) / 2.

  del hShape[axis]
  del qShape[axis]
  del zShape[axis]
  hOut = np.zeros( hShape )
  qOut = np.zeros( qShape )
  zcOut = np.zeros( hShape )
  zOut = np.zeros( zShape )
  #zOut[0,:] = np.sum(weight2d*z[0,:,:], axis=axis) / np.sum(weight2d, axis=axis)
  for k in range(nk):
    sumW = np.sum(weight2d*h[k,:,:], axis=axis)
    sumQ = np.sum(weight2d*h[k,:,:]*q[k,:,:], axis=axis)
    sumH = np.sum(weight2d*h[k,:,:]*h[k,:,:], axis=axis)
    sumZ = np.sum(weight2d*h[k,:,:]*zc[k,:,:], axis=axis)
    hOut[k,:] = sumH/sumW
    qOut[k,:] = sumQ/sumW
    zcOut[k,:] = sumZ/sumW
    zOut[k+1,:] = zOut[k] - hOut[k,:]

  return qOut, zOut, hOut

# Tests
if __name__ == '__main__':

  # Test data
  z=np.array([[0,0.2,0.3,-.1],[1,1.5,.7,.4],[2,2,1.5,2],[3,2.3,1.5,2.1]])*-1
  z=z.reshape(4,4,1)
  q=np.array(np.matlib.rand(3,4))
  q=q.reshape(3,4,1)
  print('z.shape=',z.shape,type(z))
  print('q.shape=',q.shape,type(q))
  print('z=',z)
  print('q=',q)

  qOut, zOut, hOut = axisAverage(q,z,axis=-2)
  print('qOut=',qOut)
  print('hOut=',hOut)
  print('zOut=',zOut)
