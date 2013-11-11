#!/usr/bin/env python

class MyError(Exception):
  """
  Class for error handling
  """
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


# Try to import required packages/modules
import re
import os
try: import argparse
except: raise MyError('This version of python is not new enough. python 2.7 or newer is required.')
try: from netCDF4 import Dataset
except: raise MyError('Unable to import netCDF4 module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_netcdf4')
try: import numpy as np
except: raise MyError('Unable to import numpy module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_numpy')
try: import matplotlib.pyplot as plt
except: raise MyError('Unable to import matplotlib.pyplot module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_matplotlib')
import warnings

debug = False # Global debugging
warnings.simplefilter('error', UserWarning)


def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  This is the highest level procedure invoked from the very end of the script.
  """
  global debug # Declared global in order to set it
  global optCmdLineArgs # For optional argument handling within routines

  # Arguments
  parser = argparse.ArgumentParser(description=
       'Yada yada yada',
       epilog='Written by A.Adcroft, 2013.')
  parser.add_argument('file_var_slice', type=str,
		  help='File/variable/slice specification in the form file,variable,slice1,slice2,slice3,...')
  parser.add_argument('-cm','--colormap', type=str, default='',
                  help='Specify the colormap.')
  parser.add_argument('--clim', type=float, nargs=2,
                  help='Specify the lower/upper color range.')
  parser.add_argument('-i','--ignore', type=float, nargs=1,
                  help='Mask out the specified value.')
  parser.add_argument('-o','--output', type=str, default='',
                  help='Name of image file to create.')
  parser.add_argument('--stats', action='store_true',
                  help='Print the statistics of viewed data.')
  parser.add_argument('--list', action='store_true',
                  help='Print selected data to terminal.')
  parser.add_argument('-d','--debug', action='store_true',
                  help='Turn on debugging information.')
  optCmdLineArgs = parser.parse_args()

  if optCmdLineArgs.debug: debug = True

  processSimplePlot(optCmdLineArgs.file_var_slice, optCmdLineArgs)


def processSimplePlot(fileVarSlice, args):
  """
  Generates a plot based on the file/variable/slice specified
  """

  # Extract file, variable and slice specs from fileVarSlice
  if debug: print 'processSimplePlot: fileVarSlice=',fileVarSlice
  (fileName, variableName, sliceSpecs) = splitFileVarPos(fileVarSlice)
  if debug: print 'processSimplePlot: fileName=',fileName,'variableName=',variableName,'sliceSpecs=',sliceSpecs

  # Open netcdf file
  try: rg=Dataset(fileName, 'r');
  except:
    if os.path.isfile(fileName): raise MyError('There was a problem opening "'+fileName+'".')
    raise MyError('Could not find file "'+fileName+'".')

  # If no variable is specified, summarize the file contents and exit
  if not variableName:
    print 'No variable name specified! Specify a varible from the following summary of "'\
          +fileName+'":\n'
    summarizeFile(rg)
    exit(0)

  # Check that the variable is in the file (allowing for case mismatch)
  for v in rg.variables:
    if variableName.lower() == v.lower(): variableName=v ; break
  if not variableName in rg.variables:
    print 'Known variables in file: '+''.join( (str(v)+', ' for v in rg.variables) )
    raise MyError('Did not find "'+variableName+'" in file "'+fileName+'".')

  # Obtain data along with 1D coordinates, labels and limits
  var1 = NetcdfSlice(rg, variableName, sliceSpecs)

  if var1.rank>2: # Intercept requests for ranks >2
    print 'Variable name "%s" has resolved rank %i.\nI can only plot 1D and 2D data.\n\nFile summary is:\n'%(variableName, var1.rank)
    summarizeFile(rg); print
    raise MyError('Rank of requested data is too large to plot')

  # Optionally mask out a specific value
  if optCmdLineArgs.ignore:
    var1.data = np.ma.masked_array(var1.data, mask=[var1.data==optCmdLineArgs.ignore])

  if optCmdLineArgs.list: print 'processSimplePlot: Data =\n',var1.data
  if optCmdLineArgs.stats:
    dMin = np.min(var1.data); dMax = np.max(var1.data)
    print 'Mininum=',dMin,'Maximum=',dMax
    #dMin = np.min(var1.data[var1.data!=0]); dMax = np.max(var1.data[var1.data!=0])
    #print 'Mininum=',dMin,'Maximum=',dMax,'(ignoring zeros)'
  # Now plot
  if var1.rank==0: print var1.data
  elif var1.rank==1: # Line plot
    if isAttrEqualTo(var1.coordObjs[0],'cartesian_axis','z'): # Transpose 1d plot
      xCoord = var1.data ; yData = var1.coordData[0]
      plt.plot(var1.data, var1.coordData[0])
      plt.xlabel(var1.label)
      plt.ylabel(var1.coordLabels[0]); plt.ylim(var1.coordLimits[0][0], var1.coordLimits[0][1])
      if var1.coordData[0][0]>var1.coordData[0][-1]: plt.gca().invert_yaxis()
      if isAttrEqualTo(var1.coordObjs[0],'positive','down'): plt.gca().invert_yaxis()
    else: # Normal 1d plot
      xCoord = var1.coordData[0]; yData = var1.data
      plt.plot(var1.coordData[0], var1.data)
      plt.xlabel(var1.coordLabels[0]); plt.xlim(var1.coordLimits[0][0], var1.coordLimits[0][-1])
      plt.ylabel(var1.label)
  elif var1.rank==2: # Pseudo color plot
    if debug: print 'processSimplePlot: coordData[1]=',var1.coordData[1]
    if debug: print 'processSimplePlot: coordData[0]=',var1.coordData[0]
    # Add an extra element to coordinate to force pcolormesh to draw all cells
    coordData = []
    coordData.append( np.append(var1.coordData[0],2*var1.coordData[0][-1]-var1.coordData[0][-2]) )
    coordData.append( np.append(var1.coordData[1],2*var1.coordData[1][-1]-var1.coordData[1][-2]) )
    if isAttrEqualTo(var1.coordObjs[1],'cartesian_axis','z'): # Happens for S(t,z)
      xCoord = coordData[0]; yCoord = coordData[1]; zData = np.transpose(var1.data)
      xLabel = var1.coordLabels[0]; xLims = var1.coordLimits[0]
      yLabel = var1.coordLabels[1]; yLims = var1.coordLimits[1]
      yObj = var1.coordObjs[1]
    else:
      xCoord = coordData[1]; yCoord = coordData[0]; zData = var1.data
      xLabel = var1.coordLabels[1]; xLims = var1.coordLimits[1]
      yLabel = var1.coordLabels[0]; yLims = var1.coordLimits[0]
      yObj = var1.coordObjs[0]
    plt.pcolormesh(xCoord,yCoord,zData)
    if isAttrEqualTo(yObj,'cartesian_axis','z'): # Z on y axis ?
      if yCoord[0]>yCoord[-1]: plt.gca().invert_yaxis(); yLims = reversed(yLims)
      if isAttrEqualTo(yObj,'positive','down'): plt.gca().invert_yaxis(); yLims = reversed(yLims)
    plt.title(var1.label)
    plt.xlim(xLims); plt.ylim(yLims)
    makeGuessAboutCmap()
    plt.tight_layout()
    plt.colorbar()
  if optCmdLineArgs.output:
    plt.savefig(optCmdLineArgs.output,pad_inches=0.)
  else: # Interactive
    def keyPress(event):
      if event.key=='q': exit(0)
    if var1.rank==1:
      def statusMesg(x,y):
        # -1 needed because of extension for pcolormesh
        i = min(range(len(xCoord)-1), key=lambda l: abs(xCoord[l]-x))
        if not i==None:
          val = yData[i]
          if val is np.ma.masked: return 'x=%.3f  %s(%i)=NaN'%(x,variableName,i+1)
          else: return 'x=%.3f  %s(%i)=%g'%(x,variableName,i+1,val)
        else: return 'x=%.3f y=%.3f'%(x,y)
    elif var1.rank==2:
      def statusMesg(x,y):
        # -1 needed because of extension for pcolormesh
        i = min(range(len(xCoord)-1), key=lambda l: abs(xCoord[l]-x))
        j = min(range(len(yCoord)-1), key=lambda l: abs(yCoord[l]-y))
        if not i==None:
          val = zData[j,i]
          if val is np.ma.masked: return 'x,y=%.3f,%.3f  %s(%i,%i)=NaN'%(x,y,variableName,i+1,j+1)
          else: return 'x,y=%.3f,%.3f  %s(%i,%i)=%g'%(x,y,variableName,i+1,j+1,val)
        else: return 'x,y=%.3f,%.3f'%(x,y)
      #xmin,xmax=plt.xlim(); ymin,ymax=plt.ylim();
      axis=plt.gca()
      xmin,xmax=axis.get_xlim(); ymin,ymax=axis.get_ylim();
      def zoom(event): # Scroll wheel up/down
        if event.button == 'up': scaleFactor = 1/1.5 # deal with zoom in
        elif event.button == 'down': scaleFactor = 1.5 # deal with zoom out
        elif event.button == 2: scaleFactor = 1.0
        else: return
        axmin,axmax=axis.get_xlim(); aymin,aymax=axis.get_ylim();
        (axmin,axmax),(aymin,aymax) = newLims(
          (axmin,axmax), (aymin,aymax), (event.xdata, event.ydata),
          (xmin,xmax), (ymin,ymax), scaleFactor)
        if axmin==None: return
        axis.set_xlim(axmin, axmax); axis.set_ylim(aymin, aymax)
        plt.draw() # force re-draw
      plt.gcf().canvas.mpl_connect('scroll_event', zoom)
      def zoom2(event): zoom(event)
      plt.gcf().canvas.mpl_connect('button_press_event', zoom2)
    plt.gca().format_coord = statusMesg
    plt.gcf().canvas.mpl_connect('key_press_event', keyPress)
    plt.show()


class NetcdfSlice:
  """
  Class for reading a slice of data from a netcdf using convenient index or coordinate ranges.
  """
  def __init__(self, rootGroup, variableName, sliceSpecs):
    """
    Match each slice listed in sliceSpecs with a dimension of variableName in rootGroup and read
    on that corresponding subset of data
    """
    variableHandle = rootGroup.variables[variableName]
    if debug: print 'NetcdfSlice: variableHandle=',variableHandle
    variableDims = variableHandle.dimensions
    if debug: print 'NetcdfSlice: variableDims=',variableDims
    if not (sliceSpecs==None) and len(sliceSpecs)>len(variableDims):
      raise MyError('Too many coordinate slices specified! Variable "'+variableName+
          '" has %i dimensions but you specified %i.'
          % ( len(variableDims), len(sliceSpecs) ) )
  
    # Separate provided slices into named and general
    namedSlices = []; generalSlices = []
    reGen = re.compile('[a-zA-Z_]')
    if not sliceSpecs==None:
      for s in sliceSpecs:
        if reGen.match(s): namedSlices.append(s)
        else: generalSlices.append(s)
    if debug:
      print 'NetcdfSlice: generalSlices=',generalSlices
      print 'NetcdfSlice: namedSlices=',namedSlices
  
    # Rebuild sliceSpecs by matching each of the variables actual dimensions to a named or general slice
    sliceSpecs = []
    for d in variableDims:
      thisSlice = None
      for s in namedSlices: # First look through the named slices
        n,v = s.split('=')
        if n==''+d:
          thisSlice = s
          namedSlices.remove(s)
          break
      if (not thisSlice) and len(generalSlices)>0: # Now try a general slice
        thisSlice = generalSlices[0]
        del generalSlices[0]
      if not thisSlice: thisSlice = ':' # If we ran out of general slices use a default "all" slice
      sliceSpecs.append(thisSlice)
    if debug: print 'NetcdfSlice: sliceSpecs=',sliceSpecs
    if len(namedSlices): raise MyError('The named dimension in "%s" is not a dimension of the variable "%s".'
                                % (namedSlices[0], variableName) )
    if len(generalSlices): raise MyError('There is an impossible problem. I should probably be debugged.')
  
    # Now interpret the slice specification for each dimensions
    slices1 = []; slices2 = []; labels = []; limits = []; coordData = []; coordObjs = []
    for d,s in zip(variableDims, sliceSpecs):
      equalsSplit = re.match("""
          (                          # A super group of the next two groups
          (?P<lhs>[A-Za-z0-9_]*?)    # An optional dimension name
          (?P<equals>=)              # Equals
          )?                         # Both the dimension name and equals are optional
          (?P<rhs>                   # Super group of everything on the RHS
          (?P<low>[0-9Ee\.\\+\\-]*)  # Valid number
          (?P<colon>:)?              # Colon separates low:high parts of range
          (?P<high>[0-9Ee\.\\+\\-]*) # Valid number
          )                          # Any of the three previous groups is optional but at least one is needed
          (?P<excess>.*)             # Nothing else is allowed but this will catch anything else
          """, s, re.VERBOSE)
      if debug: print 'NetcdfSlice: Interpretting "%s", groups='%(s),equalsSplit.groups()
      lhsEquals, lhs, equals, rhs, low, colon, high, excess = equalsSplit.groups()
      if len(excess)>0: raise MyError('Syntax error: could not interpret "'+s+'".')
      if len(rhs)==0: raise MyError('Syntax error: could not find range on RHS of "'+s+'".')
      if debug:
        print 'NetcdfSlice: Interpretting "%s", name = "%s"' % (s, lhs)
        print 'NetcdfSlice: Interpretting "%s", equals provided "%s"' % (s, equals)
        print 'NetcdfSlice: Interpretting "%s", ranges provided "%s"' % (s, colon)
        print 'NetcdfSlice: Interpretting "%s", low range "%s"' % (s, low)
        print 'NetcdfSlice: Interpretting "%s", high range "%s"' % (s, high)
  
      # Read the entire coordinate for this dimension
      dimensionHandle = rootGroup.dimensions[d]
      if d in rootGroup.variables:
        dimensionVariableHandle = rootGroup.variables[d]
        dimensionValues = dimensionVariableHandle[:]
        labels.append( constructLabel(dimensionVariableHandle, d) )
      else:
        dimensionVariableHandle = None
        dimensionValues = np.arange( len(dimensionHandle) ) + 1
        labels.append(d+' (index)')
      coordObjs.append( dimensionVariableHandle )
      if equals==None: # Handle case where index space was specified
        # Check that only integers were provided
        def stringIsInt(s):
          if len(s)==0: return True
          try: f=float(s)
          except ValueError: return False
          try: i=int(s)
          except ValueError: return False
          return f==float(i)
        if not stringIsInt(low):
          raise MyError('The lower end of the range "%s" must be an integer'%(s))
        if not stringIsInt(high):
          raise MyError('The upper end of the range "%s" must be an integer'%(s))
        if low=='': indexBegin = 0
        else: indexBegin = int(low) - 1 # Convert from Fortran indexing
        if colon==None: indexEnd = indexBegin
        else:
          if high=='': indexEnd = len(dimensionHandle) - 1
          else: indexEnd = int(high) - 1 # Convert from Fortran indexing
      else: # An equals was specified so the RHS low:high is in coordinate space
        if low=='': indexBegin = 0
        else: indexBegin = min(range(len(dimensionValues)), key=lambda i: abs(dimensionValues[i]-float(low)))
        if colon==None: indexEnd = indexBegin
        else:
          if high=='': indexEnd = len(dimensionHandle) - 1
          else: indexEnd = min(range(len(dimensionValues)), key=lambda i: abs(dimensionValues[i]-float(high)))
      if debug: print 'NetcdfSlice: Interpretting %s, begin:end = %i,%i' % (s,indexBegin,indexEnd)
      if indexEnd<indexBegin:
        if dimensionValues[1]<dimensionValues[0]: indexBegin, indexEnd = indexEnd, indexBegin
        elif isAttrEqualTo( dimensionVariableHandle, 'cartesian_axis', 'x'):
          print 'Note: Assuming modulo behavior for specification "%s" on dimension "%s"' %(s, d)
        else: raise MyError('Index ranges are inverted for %s'%(s))
      # Extrapolate for coordinate bounds
      if len(dimensionValues)>1:
        cMin = 1.5*dimensionValues[0] - 0.5*dimensionValues[1]
        cMax = 1.5*dimensionValues[-1] - 0.5*dimensionValues[-2]
        cRange = cMax - cMin
      else: cMin = dimensionValues[0]; cMax = cMin; cRange = 0.
      # Assign coordinateData (currently assume variable corresponding to dimension is 1D)
      if indexEnd>=indexBegin:
        indices1=slice(indexBegin, indexEnd+1); indices2 = None
        slices1.append( indices1 ); slices2.append( indices1 )
        coordinateData = dimensionValues[indices1]
      else:
        indices1 = slice(indexBegin, -1); indices2 = slice(0, indexEnd+1)
        slices1.append( indices1 ); slices2.append( indices2 )
        coordinateData = np.append(dimensionValues[indices1], dimensionValues[indices2]+cRange)
      if debug: print d,'=',coordinateData
      # Now record the actual bounds of coordinateData
      if len(coordinateData)>1:
        cMin = 1.5*coordinateData[0] - 0.5*coordinateData[1]
        cMax = 1.5*coordinateData[-1] - 0.5*coordinateData[-2]
        cRange = cMax - cMin
      else: cMin = coordinateData[0]; cMax = cMin; cRange = 0.
      limits.append((cMin, cMax))
      coordData.append( coordinateData )
    if slices1==slices2: variableData = variableHandle[slices1]
    else: variableData = np.append(variableHandle[slices1], variableHandle[slices2], axis=len(slices2)-1)

    # Remove singleton dimensions, recording values
    singletons = []; idel = []
    for i, cd in enumerate(coordData):
      if len(cd)==1: idel.insert(0, i)
    for i in idel: del coordData[i]; del coordObjs[i]; del labels[i]; del limits[i]
    if debug: print 'singletons=',singletons
    variableData = np.squeeze(variableData)

    # Attributes of class
    self.slices1 = slices1
    self.slices2 = slices2
    self.coordData = coordData
    self.coordLabels = labels
    self.coordObjs = coordObjs
    self.coordLimits = limits
    self.data = variableData
    self.label = constructLabel(variableHandle, variableName)
    self.apparentRank = len(slices1)
    self.singletons = singletons
    self.naturalShape = variableHandle.shape
    self.rank = len(variableData.shape)


def splitFileVarPos(string):
  """
  Split a string in form of "file,variable[...]" into three string parts
  Valid forms are "file", "file,variable" or "file,variable,3,j=,=2.,z=3.1:5.4,..."
  """
  m = re.match(r'([\w\.~/]+)[,:]?(.*)',string)
  fName = m.group(1)
  (vName, pSpecs) = splitVarPos(m.group(2))
  if debug: print 'splitFileVarPos: fName=',fName,'vName=',vName,'pSpecs=',pSpecs
  return fName, vName, pSpecs


def splitVarPos(string):
  """
  Split a string in form of "variable[...]" into two string parts
  Valid forms are "variable" or "variable[3,j=:,=2.,z=3.1:5.4,...,]"
  """
  vName = None; pSpecs = None
  if string:
    cSplit = string.split(',')
    if cSplit: vName = cSplit[0]
    if len(cSplit)>1: pSpecs = cSplit[1:]
    #m = re.match('(\w+)(\[([\w,:=\.\\+\\-EeNnDd]*?)\])?(.*)',string)
    #if m:
    #  if len(m.groups())>3 and len(m.group(4))>0: raise MyError('Syntax error "'+m.group(4)+'"?')
    #  vName = m.group(1)
    #  if m.group(3): pSpecs = m.group(3)
    #else: raise MyError('Could not decipher "'+string+'" for variable name.')
  #if pSpecs: pSpecs = re.split(',',pSpecs)
  if debug: print 'splitVarPos: vName=',vName,'pSpecs=',pSpecs
  return vName, pSpecs


# Interpret strSpec and return list of indices
def iRange(ncDim, strSpec, ncVar):
  equalParts = strSpec.split('='); dLen = len(ncDim)
  if debug: print '    = split',equalParts
  if len(equalParts)>2: raise MyError('Syntax error in "'+strSpec+'".')
  elif len(equalParts)==0: raise MyError('Impossible!')
  elif len(equalParts)==1: # Format : is: :ie is:ie
    colonParts = equalParts[0].split(':')
    if debug: print '    : split',colonParts
    if len(colonParts)>2: raise MyError('Too many :\'s in "'+strSpec+'".')
    # Interpret left of : to set indS
    if colonParts[0]=='': indS = 0
    else: indS = int(colonParts[0])-1 # Convert fortran/matlab index to python
    # Interpret right of : to set indE
    if len(colonParts)==1: indE = indS+1
    elif colonParts[1]=='': indE = dLen
    else: indE = int(colonParts[1]) # No conversion necessary because of python ranges
    if indS>=indE: raise MyError('In '+strSpec+' the index range is reversed.')
    if indS<0: raise MyError('In '+strSpec+' the start of the index range must >=1.')
    if indE>dLen: raise MyError('In '+strSpec+' the end of the index range is out of bounds.\n'
        +'The corresponding dimension, '+str(ncDim._name)+', has length '+str(dLen)+'.')
  elif len(equalParts)==2: # Format =: =xs: =:xe =xs:xe
    coordVals = ncVar[:]; coordSign = 1
    if coordVals[-1] < coordVals[0]: coordSign = -1 # In lieu of a sign() function
    colonParts = equalParts[1].split(':')
    if debug: print '    : split',colonParts
    if len(colonParts)==1 and colonParts[0]=='': raise MyError('Nothing on the r.h.s. of =\'s!')
    if len(colonParts)>2: raise MyError('Too many :\'s in "'+strSpec+'".')
    # Interpret left of : to set coordS
    if colonParts[0]=='': coordS = coordVals[0]
    else: coordS = float(colonParts[0])
    # Interpret right of : to set coordE
    if len(colonParts)==1: coordE = coordS
    elif colonParts[1]=='': coordE = coordVals[-1]
    else: coordE = float(colonParts[1])
    if debug: print '    coord range=',coordS,coordE
    if coordSign*(coordE-coordS) < 0: raise MyError('The coordinate range "'+strSpec+'" is inverted!')
    indS = coord2index(coordVals, coordS, roundUp=True)
    if coordE==coordS: indE = indS + 1
    else: indE = coord2index(coordVals, coordE)+1 # Python range requires +1
  if debug: print '    is,ie=',indS,indE
  if indE-indS==1: return [indS]
  else: return range(indS,indE)


# Returns index of element of coordList with nearest value to coordVal
def coord2index(coordList, coordVal, roundUp=False):
  ind = min(range(len(coordList)), key=lambda i:abs(coordList[i]-coordVal))
  if roundUp and (ind+1<len(coordList)):
    # Correct for rounding 0.5 down to 0.
    if abs(coordList[ind+1]-coordVal)<=abs(coordList[ind]-coordVal): ind=ind+1
  if debug: print '      coord(',ind,')=',coordList[ind],' matches coord=',coordVal
  return ind


def constructLabel(ncObj, default=''):
  """
  Returns a string combining CF attiributes "long_name" and "units"
  """
  if debug: print 'constructLabel: ncObj=',ncObj
  label = ''
  if 'long_name' in ncObj.ncattrs():
    label += str(ncObj.getncattr('long_name'))+' '
  else: label += ncObj._name+' '
  if 'units' in ncObj.ncattrs():
    label += '('+str(ncObj.getncattr('units'))+')'
  if len(label)==0: label = default+' (index)'
  return label


def isAttrEqualTo(ncObj, name, value):
  """
  Returns True if ncObj has attribute "name" that matches "value"
  """
  if not ncObj: return False
  if name in ncObj.ncattrs():
    if value.lower() in str(ncObj.getncattr(name)).lower():
      return True
  return False


# Make an intelligent choice about which colormap to use
def makeGuessAboutCmap():
  vmin, vmax = plt.gci().get_clim()
  if vmin==vmax:
    if debug: print 'vmin,vmax=',vmin,vmax
    vmin = vmin - 1; vmax = vmax + 1
  if optCmdLineArgs.colormap:
    plt.set_cmap(optCmdLineArgs.colormap)
  else:
    if vmin*vmax>=0: # Single signed data
      if max(vmin,vmax)>0: plt.set_cmap('hot')
      else: plt.set_cmap('hot_r')
    else: # Multi-signed data
      cutOffFrac = 0.3
      if -vmin<vmax and -vmin/vmax>cutOffFrac:
        plt.clim(-vmax,vmax)
        plt.set_cmap('seismic')
      elif -vmin>vmax and -vmax/vmin>cutOffFrac:
        plt.clim(vmin,-vmin)
        plt.set_cmap('seismic')
      else: plt.set_cmap('spectral')
  if optCmdLineArgs.clim:
    plt.clim(optCmdLineArgs.clim[0],optCmdLineArgs.clim[1])


# Generate a succinct summary of the netcdf file contents
def summarizeFile(rg):
  dims = rg.dimensions; vars = rg.variables
  print 'Dimensions:'
  for dim in dims:
    oString = ' '+dim+' ['+str(len( dims[dim] ))+']'
    if dim in vars:
      n = len( dims[dim] ); obj = rg.variables[dim]
      if n>5: oString += ' = '+str(obj[0])+'...'+str(obj[n-1])
      else: oString += ' = '+str(obj[:])
      if 'long_name' in obj.ncattrs(): oString += ' "'+obj.getncattr('long_name')+'"'
      if 'units' in obj.ncattrs(): oString += ' ('+obj.getncattr('units')+')'
    print oString
  print; print 'Variables:'
  for var in vars:
    if var in dims: continue # skip listing dimensions as variables
    oString = ' '+var+' [ '; dString = ''
    obj = vars[var]; varDims = obj.dimensions
    for dim in varDims:
      if len(dString)>0: dString += ', '
      dString += dim+'['+str(len( dims[dim] ))+']'
    oString += dString+' ]'
    if 'long_name' in obj.ncattrs(): oString += ' "'+obj.getncattr('long_name')+'"'
    if 'units' in obj.ncattrs(): oString += ' ('+obj.getncattr('units')+')'
    print oString


# Calculate a new window by scaling the current window, centering
# on the cursor if possible.
def newLims(cur_xlim, cur_ylim, cursor, xlim, ylim, scale_factor):
  cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
  cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
  xdata = cursor[0]; ydata = cursor[1]
  new_xrange = cur_xrange*scale_factor; new_yrange = cur_yrange*scale_factor
  xdata = min( max( xdata, xlim[0]+new_xrange ), xlim[1]-new_xrange )
  xL = max( xlim[0], xdata - new_xrange ); xR = min( xlim[1], xdata + new_xrange )
  if ylim[1]>ylim[0]:
    ydata = min( max( ydata, ylim[0]+new_yrange ), ylim[1]-new_yrange )
    yL = max( ylim[0], ydata - new_yrange ); yR = min( ylim[1], ydata + new_yrange )
  else:
    ydata = min( max( ydata, ylim[1]-new_yrange ), ylim[0]+new_yrange )
    yR = max( ylim[1], ydata + new_yrange ); yL = min( ylim[0], ydata - new_yrange )
  if xL==cur_xlim[0] and xR==cur_xlim[1] and \
     yL==cur_ylim[0] and yR==cur_ylim[1]: return (None, None), (None, None)
  return (xL, xR), (yL, yR)


# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()
