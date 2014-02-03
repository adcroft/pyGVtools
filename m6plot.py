"""
A method for producing a standardized pseudo-colot plot of 2D data
"""

import numpy, numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import math


def xyplot(field, x=None, y=None, area=None,
  xLabel=None, xUnits=None, yLabel=None, yUnits=None,
  title='', suptitle='', nBins=None, cLim=None, landColor=[.5,.5,.5], colormap=None,
  aspect=[16,9], resolution=576,
  ignore=None, save=None, debug=False, show=False):
  """
  Renders plot of scalar field, field(x,y).

  Arguments:
  field       Scalar 2D array to be plotted.
  x           x coordinate (1D or 2D array). If x is the same size as field then x is treated as
              the cell center coordinates.
  y           y coordinate (1D or 2D array). If x is the same size as field then y is treated as
              the cell center coordinates.
  area        2D array of cell areas (used for statistics). Default None.
  xLabel      The label for the x axis. Default 'Longitude'.
  xUnits      The units for the x axis. Default 'degrees E'.
  yLabel      The label for the x axis. Default 'Latitude'.
  yUnits      The units for the x axis. Default 'degrees N'.
  title       The title to place at the top of the panel. Default ''.
  suptitle    The super-title to place at the top of the figure. Default ''.
  nBins       The number of colors levels (used is cLim is missing or only specifies the color range).
  cLim        A tuple of (min,max) color range OR a list of contour levels. Default None.
  landColor   An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  colormap    The name of the colormap to use. Default None.
  aspect      The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution  The vertical rseolutin of the figure given in pixels. Default 720.
  ignore      A value to use as no-data (NaN). Default None.
  save        Name of file to save figure in. Default None.
  debug       If true, report sutff for debugging. Default False.
  show        If true, causes the figure to appear on screen. Used for testing. Default False.
  """

  # Create coordinates if not provided
  xLabel, xUnits, yLabel, yUnits = createLabels(x, y, xLabel, xUnits, yLabel, yUnits)
  if debug: print 'x,y label/units=',xLabel,xUnits,yLabel,yUnits
  xCoord, yCoord = createCoords(field, x, y)

  # Diagnose statistics
  if ignore!=None: maskedField = numpy.ma.masked_array(field, mask=[field==ignore])
  else: maskedField = field.copy()
  sMin, sMax, sMean, sStd, sRMS = myStats(maskedField, area, debug=debug)
  xLims = boundaryStats(xCoord)
  yLims = boundaryStats(yCoord)

  # Choose colormap
  if nBins==None and (cLim==None or len(cLim)==2): nBins=35
  if colormap==None: colormap = chooseColorMap(sMin, sMax)
  cmap, norm, extend = chooseColorLevels(sMin, sMax, colormap, cLim=cLim, nBins=nBins)

  setFigureSize(aspect, resolution, debug=debug)
  plt.gcf().subplots_adjust(left=.08, right=.99, wspace=0, bottom=.09, top=.9, hspace=0)
  axis = plt.gca()
  plt.pcolormesh(xCoord, yCoord, maskedField, cmap=cmap, norm=norm)
  plt.colorbar(fraction=.08, pad=0.02, extend=extend)
  plt.gca().set_axis_bgcolor(landColor)
  plt.xlim( xLims )
  plt.ylim( yLims )
  axis.annotate('min=%.6g\nmax=%.6g'%(sMin,sMax), xy=(0.0,1.01), xycoords='axes fraction', verticalalignment='bottom', fontsize=10)
  if area!=None:
    axis.annotate('mean=%.6g\nrms=%.6g'%(sMean,sRMS), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
    axis.annotate(' std=%.6g\n'%(sStd), xy=(1.0,1.01), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='left', fontsize=10)
  if len(xLabel+xUnits)>0: plt.xlabel(label(xLabel, xUnits))
  if len(yLabel+yUnits)>0: plt.ylabel(label(yLabel, yUnits))
  if len(title)>0: plt.title(title)
  if len(suptitle)>0: plt.suptitle(suptitle)
  if show: plt.show(block=False)
  if save!=None: plt.savefig(save)


def xycompare(field1, field2, x=None, y=None, area=None,
  xLabel=None, xUnits=None, yLabel=None, yUnits=None,
  title1='', title2='', title3='A - B', addPlabel=True, suptitle='',
  nBins=None, cLim=None, dLim=None, landColor=[.5,.5,.5], colormap=None, dcolormap=None,
  aspect=[6,10], resolution=1200,
  ignore=None, save=None, debug=False, show=False):
  """
  Renders n-panel plot of two scalar fields, field1(x,y) and field2(x,y).

  Arguments:
  field1      Scalar 2D array to be plotted and compared to field2.
  field2      Scalar 2D array to be plotted and compared to field1.
  x           x coordinate (1D or 2D array). If x is the same size as field then x is treated as
              the cell center coordinates.
  y           y coordinate (1D or 2D array). If x is the same size as field then y is treated as
              the cell center coordinates.
  area        2D array of cell areas (used for statistics). Default None.
  xLabel      The label for the x axis. Default 'Longitude'.
  xUnits      The units for the x axis. Default 'degrees E'.
  yLabel      The label for the x axis. Default 'Latitude'.
  yUnits      The units for the x axis. Default 'degrees N'.
  title1      The title to place at the top of panel 1. Default ''.
  title2      The title to place at the top of panel 1. Default ''.
  title3      The title to place at the top of panel 1. Default 'A-B'.
  addPlabel   Adds a 'A:' or 'B:' to the title1 and title2. Default True.
  suptitle    The super-title to place at the top of the figure. Default ''.
  nBins       The number of colors levels (used is cLim is missing or only specifies the color range).
  cLim        A tuple of (min,max) color range OR a list of contour levels for the field plots. Default None.
  dLim        A tuple of (min,max) color range OR a list of contour levels for the difference plot. Default None.
  landColor   An rgb tuple to use for the color of land (no data). Default [.5,.5,.5].
  colormap    The name of the colormap to use for the field plots. Default None.
  dcolormap   The name of the colormap to use for the differece plot. Default None.
  aspect      The aspect ratio of the figure, given as a tuple (W,H). Default [16,9].
  resolution  The vertical rseolutin of the figure given in pixels. Default 1280.
  ignore      A value to use as no-data (NaN). Default None.
  save        Name of file to save figure in. Default None.
  debug       If true, report sutff for debugging. Default False.
  show        If true, causes the figure to appear on screen. Used for testing. Default False.
  """

  if (field1.shape)!=(field2.shape): raise Exception('field1 and field2 must be the same shape')

  # Create coordinates if not provided
  xLabel, xUnits, yLabel, yUnits = createLabels(x, y, xLabel, xUnits, yLabel, yUnits)
  if debug: print 'x,y label/units=',xLabel,xUnits,yLabel,yUnits
  xCoord, yCoord = createCoords(field1, x, y)

  # Diagnose statistics
  if ignore!=None: maskedField1 = numpy.ma.masked_array(field1, mask=[field1==ignore])
  else: maskedField1 = field1.copy()
  s1Min, s1Max, s1Mean, s1Std, s1RMS = myStats(maskedField1, area, debug=debug)
  if ignore!=None: maskedField2 = numpy.ma.masked_array(field2, mask=[field2==ignore])
  else: maskedField2 = field2.copy()
  s2Min, s2Max, s2Mean, s2Std, s2RMS = myStats(maskedField2, area, debug=debug)
  dMin, dMax, dMean, dStd, dRMS = myStats(maskedField1 - maskedField2, area, debug=debug)
  dRxy = corr(maskedField1 - s1Mean, maskedField2 - s2Mean, area)
  print dRxy
  s12Min = min(s1Min, s2Min); s12Max = max(s1Max, s2Max)
  xLims = boundaryStats(xCoord); yLims = boundaryStats(yCoord)
  if debug:
    print 's1: min, max, mean =', s1Min, s1Max, s1Mean
    print 's2: min, max, mean =', s2Min, s2Max, s2Mean
    print 's12: min, max =', s12Min, s12Max

  # Choose colormap
  if nBins==None and (cLim==None or len(cLim)==2): cBins=35
  else: cBins=nBins
  if colormap==None: colormap = chooseColorMap(s12Min, s12Max)
  cmap, norm, extend = chooseColorLevels(s12Min, s12Max, colormap, cLim=cLim, nBins=cBins)

  def annotateStats(axis, sMin, sMax, sMean, sStd, sRMS):
    axis.annotate('min=%.6g\nmax=%.6g'%(sMin,sMax), xy=(0.0,1.025), xycoords='axes fraction', verticalalignment='bottom', fontsize=10)
    if sMean!=None:
      axis.annotate('mean=%.6g\nrms=%.6g'%(sMean,sRMS), xy=(1.0,1.025), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
      axis.annotate(' std=%.6g\n'%(sStd), xy=(1.0,1.025), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='left', fontsize=10)

  if addPlabel: preTitleA = 'A: '; preTitleB = 'B: '
  else: preTitleA = ''; preTitleB = ''

  setFigureSize(aspect, resolution, debug=debug)
  plt.gcf().subplots_adjust(left=.11, right=.94, wspace=0, bottom=.05, top=.94, hspace=0.15)

  plt.subplot(3,1,1)
  plt.pcolormesh(xCoord, yCoord, maskedField1, cmap=cmap, norm=norm)
  plt.colorbar(fraction=.08, pad=0.02, extend=extend)
  plt.gca().set_axis_bgcolor(landColor)
  plt.xlim( xLims ); plt.ylim( yLims )
  annotateStats(plt.gca(), s1Min, s1Max, s1Mean, s1Std, s1RMS)
  plt.gca().set_xticklabels([''])
  if len(yLabel+yUnits)>0: plt.ylabel(label(yLabel, yUnits))
  if len(title1)>0: plt.title(preTitleA+title1)

  plt.subplot(3,1,2)
  plt.pcolormesh(xCoord, yCoord, maskedField2, cmap=cmap, norm=norm)
  plt.colorbar(fraction=.08, pad=0.02, extend=extend)
  plt.gca().set_axis_bgcolor(landColor)
  plt.xlim( xLims ); plt.ylim( yLims )
  annotateStats(plt.gca(), s2Min, s2Max, s2Mean, s2Std, s2RMS)
  plt.gca().set_xticklabels([''])
  if len(yLabel+yUnits)>0: plt.ylabel(label(yLabel, yUnits))
  if len(title2)>0: plt.title(preTitleB+title2)

  plt.subplot(3,1,3)
  if dcolormap==None: dcolormap = chooseColorMap(dMin, dMax)
  cmap, norm, extend = chooseColorLevels(dMin, dMax, dcolormap, cLim=dLim, nBins=nBins)
  plt.pcolormesh(xCoord, yCoord, maskedField1 - maskedField2, cmap=cmap, norm=norm)
  plt.colorbar(fraction=.08, pad=0.02, extend=extend)
  plt.gca().set_axis_bgcolor(landColor)
  plt.xlim( xLims ); plt.ylim( yLims )
  annotateStats(plt.gca(), dMin, dMax, dMean, dStd, dRMS)
  plt.gca().annotate(' r(A,B)=%.6g\n'%(dRxy), xy=(1.0,-0.14), xycoords='axes fraction', verticalalignment='bottom', horizontalalignment='right', fontsize=10)
  if len(xLabel+xUnits)>0: plt.xlabel(label(xLabel, xUnits))
  if len(yLabel+yUnits)>0: plt.ylabel(label(yLabel, yUnits))
  if len(title3)>0: plt.title(title3)
  if len(suptitle)>0: plt.suptitle(suptitle)


  if show: plt.show(block=False)
  if save!=None: plt.savefig(save)


def chooseColorMap(sMin, sMax):
  """
  Based on the min/max extremes of the data, choose a colormap that fits the data.
  """
  if sMin<0 and sMax>0: return 'seismic'
  elif sMax>0 and sMin<0.1*sMax: return 'hot'
  elif sMin<0 and sMax>0.1*sMin: return 'hot_r'
  else: return 'spectral'


def chooseColorLevels(sMin, sMax, colorMapName, cLim=None, nBins=None, steps=[1,2,2.5,5,10]):
  """
  If nBins is a positive integer, choose sensible color levels with nBins colors.
  If cLim is a 2-element tuple, create color levels within the cLim range
  or if cLim is a vector, use cLim as contour levels.
  If cLim provides more than 2 color interfaces, nBins must be absent.
  If cLim is absent, the sMin,sMax are used as the color range bounds.
  
  Returns cmap, norm and extend.
  """
  if nBins==None and cLim==None: raise Exception('At least one of cLim or nBins is required.')
  if cLim!=None:
    if len(cLim)<2: raise Exception('cLim must be at least 2 values long.')
    if nBins==None and len(cLim)==2: raise Exception('nBins must be provided when cLims specifies a color range.')
    if nBins!=None and len(cLim)>2: raise Exception('nBins cannot be provided when cLims specifies color levels.')
  if cLim==None: levels = MaxNLocator(nbins=nBins, steps=steps).tick_values(sMin, sMax)
  elif len(cLim)==2: levels = MaxNLocator(nbins=nBins, steps=steps).tick_values(cLim[0], cLim[1])
  else: levels = cLim

  nColors = len(levels)-1
  if sMin<levels[0] and sMax>levels[-1]: extend = 'both'; eColors=[1,1]
  elif sMin<levels[0] and sMax<=levels[-1]: extend = 'min'; eColors=[1,0]
  elif sMin>=levels[0] and sMax>levels[-1]: extend = 'max'; eColors=[0,1]
  else: extend = 'neither'; eColors=[0,0]

  cmap = plt.cm.get_cmap(colorMapName,lut=nColors+eColors[0]+eColors[1])
  cmap0 = cmap(0.)
  cmap1 = cmap(1.)
  cmap = ListedColormap(cmap(range(eColors[0],nColors+1-eColors[1]+eColors[0])))#, N=nColors)
  if eColors[0]>0: cmap.set_under(cmap0)
  if eColors[1]>0: cmap.set_over(cmap1)
  norm = BoundaryNorm(levels, ncolors=cmap.N)
  return cmap, norm, extend


def myStats(s, area, s2=None, debug=False):
  """
  Calculates mean, standard deviation and root-mean-square of s.
  """
  sMin = numpy.ma.min(s); sMax = numpy.ma.max(s)
  if area==None: return sMin, sMax, None, None, None
  weight = area.copy()
  if debug: print 'myStats: sum(area) =',numpy.ma.sum(weight)
  if not numpy.ma.getmask(s).any()==numpy.ma.nomask: weight[s.mask] = 0.
  sumArea = numpy.ma.sum(weight)
  if debug: print 'myStats: sum(area) =',sumArea,'after masking'
  if debug: print 'myStats: sum(s) =',numpy.ma.sum(s)
  if debug: print 'myStats: sum(area*s) =',numpy.ma.sum(weight*s)
  mean = numpy.ma.sum(weight*s)/sumArea
  std = math.sqrt( numpy.ma.sum( weight*((s-mean)**2) )/sumArea )
  rms = math.sqrt( numpy.ma.sum( weight*(s**2) )/sumArea )
  if debug: print 'myStats: mean(s) =',mean
  if debug: print 'myStats: std(s) =',std
  if debug: print 'myStats: rms(s) =',rms
  return sMin, sMax, mean, std, rms


def corr(s1, s2, area):
  """
  Calculates the correlation coefficient between s1 and s2, assuming s1 and s2 have
  not mean. That is s1 = S - mean(S), etc.
  """
  weight = area.copy()
  if not numpy.ma.getmask(s1).any()==numpy.ma.nomask: weight[s1.mask] = 0.
  sumArea = numpy.ma.sum(weight)
  v1 = numpy.ma.sum( weight*(s1**2) )/sumArea
  v2 = numpy.ma.sum( weight*(s2**2) )/sumArea
  if v1==0 or v2==0: return numpy.NaN
  rxy = numpy.ma.sum( weight*(s1*s2) )/sumArea / math.sqrt( v1*v2 )
  return rxy


def createCoords(s, x, y):
  """
  Checks that x and y are appropriate 2D corner coordinates
  and tries to make some if they are not.
  """
  nj, ni = s.shape
  if x==None: xCoord = numpy.arange(0., ni+1)
  else: xCoord = numpy.ma.filled(x, 0.)
  if y==None: yCoord = numpy.arange(0., nj+1)
  else: yCoord = numpy.ma.filled(y, 0.)

  # Turn coordinates into 2D arrays if 1D arrays were provided
  if len(xCoord.shape)==1:
    nxy = yCoord.shape
    xCoord = numpy.matlib.repmat(xCoord, nxy[0], 1)
  nxy = xCoord.shape
  if len(yCoord.shape)==1: yCoord = numpy.matlib.repmat(yCoord.T, nxy[-1], 1).T
  if xCoord.shape!=yCoord.shape: raise Exception('The shape of coordinates are mismatched!')

  # Create corner coordinates from center coordinates is center coordinates were provided
  if xCoord.shape!=yCoord.shape: raise Exception('The shape of coordinates are mismatched!')
  if s.shape==xCoord.shape:
    xCoord = expandJ( expandI( xCoord ) )
    yCoord = expandJ( expandI( yCoord ) )
  return xCoord, yCoord


def expandI(a):
  """
  Expands an array by one column, averaging the data to the middle columns and
  extrapolating for the first and last columns. Need for shifting coordinates
  from centers to corners.
  """
  nj, ni = a.shape
  b = numpy.zeros((nj, ni+1))
  b[:,1:-1] = 0.5*( a[:,:-1] + a[:,1:] )
  b[:,0] = a[:,0] + 0.5*( a[:,0] - a[:,1] )
  b[:,-1] = a[:,-1] + 0.5*( a[:,-1] - a[:,-2] )
  return b


def expandJ(a):
  """
  Expands an array by one row, averaging the data to the middle columns and
  extrapolating for the first and last rows. Need for shifting coordinates
  from centers to corners.
  """
  nj, ni = a.shape
  b = numpy.zeros((nj+1, ni))
  b[1:-1,:] = 0.5*( a[:-1,:] + a[1:,:] )
  b[0,:] = a[0,:] + 0.5*( a[0,:] - a[1,:] )
  b[-1,:] = a[-1,:] + 0.5*( a[-1,:] - a[-2,:] )
  return b


def boundaryStats(a):
  """
  Returns the minimum and maximum values of a only on the boundaries of the array.
  """
  amin = numpy.amin(a[0,:])
  amin = min(amin, numpy.amin(a[1:,-1]))
  amin = min(amin, numpy.amin(a[-1,:-1]))
  amin = min(amin, numpy.amin(a[1:-1,0]))
  amax = numpy.amax(a[0,:])
  amax = max(amax, numpy.amax(a[1:,-1]))
  amax = max(amax, numpy.amax(a[-1,:-1]))
  amax = max(amax, numpy.amax(a[1:-1,0]))
  return amin, amax


def setFigureSize(aspect, verticalResolution, debug=False):
  """
  Set the figure size based on vertical resolution and aspect ratio (tuple of W,H).
  """
  width = int(1.*aspect[0]/aspect[1] * verticalResolution) # First guess
  if debug: print 'setFigureSize: first guess width =',width
  width = width + ( width % 2 ) # Make even
  if debug: print 'setFigureSize: corrected width =',width
  if debug: print 'setFigureSize: height =',verticalResolution
  plt.figure(figsize=(width/100., verticalResolution/100.)) # 100 dpi always?


def label(label, units):
  """
  Combines a label string and units string together in the form 'label [units]'
  unless one of the other is empty.
  """
  string = unicode(label)
  if len(units)>0: string = string + ' [' + unicode(units) + ']'
  return string


def createLabels(x, y, xLabel, xUnits, yLabel, yUnits):
  """
  Checks that x and y are appropriate 2D corner coordinates
  and tries to make some if they are not.
  """
  if x==None:
    if xLabel==None: xLabel='i'
    if xUnits==None: xUnits=''
  else:
    if xLabel==None: xLabel=u'Longitude'
    if xUnits==None: xUnits=u'\u00B0E'
  if y==None:
    if yLabel==None: yLabel='j'
    if yUnits==None: yUnits=''
  else:
    if yLabel==None: yLabel=u'Latitude'
    if yUnits==None: yUnits=u'\u00B0N'
  return xLabel, xUnits, yLabel, yUnits


# Test
if __name__ == '__main__':
  import nccf
  file = 'baseline/19000101.ocean_static.nc'
  D,(y,x),_ = nccf.readVar(file,'depth_ocean')
  y,_,_ = nccf.readVar(file,'geolat')
  x,_,_ = nccf.readVar(file,'geolon')
  area,_,_ = nccf.readVar(file,'area_t')
  xyplot(D, x, y, title='Depth', ignore=0, suptitle='Testing', area=area, cLim=[0, 5500], nBins=12, debug=True)#, save='fig_test.png')
  xycompare(D, .9*D, x, y, title1='Depth', ignore=0, suptitle='Testing', area=area, nBins=12, debug=True)#, save='fig_test2.png')
  plt.show()
