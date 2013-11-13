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
  parser.add_argument('-sg','--supergrid', type=str, default=None,
                  help='The supergrid to use for horizontal coordinates.')
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

  createUI(optCmdLineArgs.file_var_slice, optCmdLineArgs)


def createUI(fileVarSlice, args):
  """
  Generates a plot based on the file/variable/slice specified
  """

  # Extract file, variable and slice specs from fileVarSlice
  if debug: print 'createUI: fileVarSlice=',fileVarSlice
  (fileName, variableName, sliceSpecs) = splitFileVarPos(fileVarSlice)
  if debug: print 'createUI: fileName=',fileName,'variableName=',variableName,'sliceSpecs=',sliceSpecs

  # Open netcdf file
  try: rg = Dataset(fileName, 'r');
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

  # Obtain meta data along with 1D coordinates, labels and limits
  var1 = NetcdfSlice(rg, variableName, sliceSpecs)
  if var1.rank>2: # Intercept requests for rank >2
    summarizeFile(rg); print
    raise MyError( 'Variable name "%s" has resolved rank %i. Only 1D and 2D data can be plotted until you buy a holgraphic display.'%(variableName, var1.rank))
  var1.getData() # Actually read data from file

  # Optionally mask out a specific value
  if optCmdLineArgs.ignore:
    var1.data = np.ma.masked_array(var1.data, mask=[var1.data==optCmdLineArgs.ignore])

  if optCmdLineArgs.list: print 'createUI: Data =\n',var1.data
  if optCmdLineArgs.stats:
    dMin = np.min(var1.data); dMax = np.max(var1.data)
    print 'Mininum=',dMin,'Maximum=',dMax
    #dMin = np.min(var1.data[var1.data!=0]); dMax = np.max(var1.data[var1.data!=0])
    #print 'Mininum=',dMin,'Maximum=',dMax,'(ignoring zeros)'
  # Now plot
  if var1.rank==0:
    for d in var1.allDims:
      print '%s = %g %s'%(d.name,d.values[0],d.units)
    print var1.name+' = '+repr(var1.data)+'   '+var1.units
    exit(0)
  elif var1.rank==1: # Line plot
    if var1.dims[0].isZaxis: # Transpose 1d plot
      xCoord = var1.data ; yData = var1.dims[0].values
      plt.plot(xCoord, yData)
      plt.xlabel(var1.label)
      plt.ylabel(var1.dims[0].label);
      if var1.dims[0].values[0]>var1.dims[0].values[-1]: plt.gca().invert_yaxis()
      if var1.dims[0].positiveDown: plt.gca().invert_yaxis()
    else: # Normal 1d plot
      xCoord = var1.dims[0].values; yData = var1.data
      plt.plot(xCoord, yData)
      plt.xlabel(var1.dims[0].label); plt.xlim(var1.dims[0].limits[0], var1.dims[0].limits[-1])
      plt.ylabel(var1.label)
  elif var1.rank==2: # Pseudo color plot
    # Add an extra element to coordinate to force pcolormesh to draw all cells
    coordData = []
    #coordData.append( np.append(var1.dims[0].values,2*var1.dims[0].values[-1]-var1.dims[0].values[-2]) )
    #coordData.append( np.append(var1.dims[1].values,2*var1.dims[1].values[-1]-var1.dims[1].values[-2]) )
    coordData.append( extrapCoord( var1.dims[0].values ) )
    coordData.append( extrapCoord( var1.dims[1].values ) )
    if var1.dims[1].isZaxis: # Happens for S(t,z)
      xCoord = coordData[0]; yCoord = coordData[1]; zData = np.transpose(var1.data)
      xLabel = var1.dims[0].label; xLims = var1.dims[0].limits
      yLabel = var1.dims[1].label; yLims = var1.dims[1].limits
      yDim = var1.dims[1]
    else:
      xLabel = var1.dims[1].label; xLims = var1.dims[1].limits
      yLabel = var1.dims[0].label; yLims = var1.dims[0].limits
      if args.supergrid==None:
        xCoord = coordData[1]; yCoord = coordData[0]
      else:
        xCoord, xLims = readSGvar(args.supergrid, 'x', var1.dims)
        yCoord, yLims = readSGvar(args.supergrid, 'y', var1.dims)
      zData = var1.data
      yDim = var1.dims[0]
    plt.pcolormesh(xCoord,yCoord,zData)
    if yDim.isZaxis: # Z on y axis ?
      if yCoord[0]>yCoord[-1]: plt.gca().invert_yaxis(); yLims = reversed(yLims)
      if yDim.positiveDown: plt.gca().invert_yaxis(); yLims = reversed(yLims)
    plt.title(var1.label)
    plt.xlim(xLims); plt.ylim(yLims)
    makeGuessAboutCmap()
    plt.tight_layout()
    plt.colorbar()
  axis=plt.gca()
  if var1.singleDims:
    text = ''
    for d in var1.singleDims:
      if len(text): text = text+'   '
      text = text + d.name + ' = ' + str(d.values[0])
      if d.units: text = text + ' (' + d.units + ')'
    axis.annotate(text, xy=(0.005,.995), xycoords='figure fraction', verticalalignment='top', fontsize=8)
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
        if len(xCoord.shape)==1:
          # -2 needed because of coords are for vertices and need to be averaged to centers
          i = min(range(len(xCoord)-2), key=lambda l: abs((xCoord[l]+xCoord[l+1])/2.-x))
          j = min(range(len(yCoord)-2), key=lambda l: abs((yCoord[l]+yCoord[l+1])/2.-y))
        else:
          idx = np.abs( np.fabs( xCoord[0:-1,0:-1]+xCoord[1:,1:]+xCoord[0:-1,1:]+xCoord[1:,0:-1]-4*x)
              +np.fabs( yCoord[0:-1,0:-1]+yCoord[1:,1:]+yCoord[0:-1,1:]+yCoord[1:,0:-1]-4*y) ).argmin()
          j,i = np.unravel_index(idx,zData.shape)
        if not i==None:
          val = zData[j,i]
          if val is np.ma.masked: return 'x,y=%.3f,%.3f  %s(%i,%i)=NaN'%(x,y,variableName,i+1,j+1)
          else: return 'x,y=%.3f,%.3f  %s(%i,%i)=%g'%(x,y,variableName,i+1,j+1,val)
        else: return 'x,y=%.3f,%.3f'%(x,y)
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


class NetcdfDim:
  """
  Class for describing a dimension in a netcdf file
  """
  def __init__(self, rootGroup, dimensionName, sliceSpec):
    """
    Initialize a dimension by interpretting a sliceSpec
    """
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
          """, sliceSpec, re.VERBOSE)
    if debug: print 'NetcdfSlice: Interpretting "%s", groups='%(sliceSpec),equalsSplit.groups()
    lhsEquals, lhs, equals, rhs, low, colon, high, excess = equalsSplit.groups()
    if len(excess)>0: raise MyError('Syntax error: could not interpret "'+sliceSpec+'".')
    if len(rhs)==0: raise MyError('Syntax error: could not find range on RHS of "'+sliceSpec+'".')
    if debug:
      print 'NetcdfDim: Interpretting "%s", name = "%s"' % (sliceSpec, lhs)
      print 'NetcdfDim: Interpretting "%s", equals provided "%s"' % (sliceSpec, equals)
      print 'NetcdfDim: Interpretting "%s", ranges provided "%s"' % (sliceSpec, colon)
      print 'NetcdfDim: Interpretting "%s", low range "%s"' % (sliceSpec, low)
      print 'NetcdfDim: Interpretting "%s", high range "%s"' % (sliceSpec, high)
    dimensionHandle = rootGroup.dimensions[dimensionName]
    self.isZaxis = False
    self.positiveDown = None
    if dimensionName in rootGroup.variables:
      dimensionVariableHandle = rootGroup.variables[dimensionName]
      dimensionValues = None
      self.label, self.name, self.units = constructLabel(dimensionVariableHandle, dimensionName)
      if isAttrEqualTo(dimensionVariableHandle,'cartesian_axis','z'):
        self.isZaxis = True
        if isAttrEqualTo(dimensionVariableHandle,'positive','down'): self.positiveDown = True
        else: self.positiveDown = False
    else:
      dimensionVariableHandle = None
      dimensionValues = np.arange( len(dimensionHandle) ) + 1
      self.label = dimensionName
      self.name = dimensionName
      self.units = ''
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
        raise MyError('The lower end of the range "%s" must be an integer'%(sliceSpec))
      if not stringIsInt(high):
        raise MyError('The upper end of the range "%s" must be an integer'%(sliceSpec))
      if low=='': indexBegin = 0
      else: indexBegin = int(low) - 1 # Convert from Fortran indexing
      if colon==None: indexEnd = indexBegin
      else:
        if high=='': indexEnd = len(dimensionHandle) - 1
        else: indexEnd = int(high) - 1 # Convert from Fortran indexing
    else: # An equals was specified so the RHS low:high is in coordinate space
      if dimensionVariableHandle: dimensionValues = dimensionVariableHandle[:] # Read global coordinate data
      cMin = 1.5*dimensionValues[0] - 0.5*dimensionValues[1]
      cMax = 1.5*dimensionValues[-1] - 0.5*dimensionValues[-2]
      isLongitude = int(0.5+cMax-cMin)==360
      if low=='': indexBegin = 0; fLow = cMin
      else:
        fLow = float(low)
        indexBegin = min(range(len(dimensionValues)), key=lambda i: abs(dimensionValues[i]-fLow))
        if indexBegin==0 and isLongitude and float(low)<cMin:
          indexBegin = min(range(len(dimensionValues)), key=lambda i: abs(dimensionValues[i]-fLow-360.))
      if colon==None: indexEnd = indexBegin
      else:
        if high=='': indexEnd = len(dimensionHandle) - 1
        else:
          indexEnd = min(range(len(dimensionValues)), key=lambda i: abs(dimensionValues[i]-float(high)))
          if indexEnd==len(dimensionValues)-1 and isLongitude and float(high)>cMax:
            indexEnd = min(range(len(dimensionValues)), key=lambda i: abs(dimensionValues[i]-float(high)+360.))
          if indexEnd==indexBegin and abs(float(high)-fLow)>0:
            if indexEnd>0: indexEnd = indexEnd - 1
            else: indexBegin = indexBegin + 1
    # Convert index bounds to index lists
    if indexEnd>=indexBegin:
      self.slice1 = slice(indexBegin, indexEnd+1)
      self.slice2 = None
      self.len = indexEnd - indexBegin + 1
    else:
      self.slice1 = slice(indexBegin, -1)
      self.slice2 = slice(0, indexEnd+1)
      self.len = len(dimensionHandle) - indexBegin + 1 + indexEnd
    self.lenInFile = len(dimensionHandle)
    self.limits = (None, None)
    self.dimensionVariableHandle = dimensionVariableHandle
    self.values = None
  def getData(self):
    """
    Read dimension variable data if it has not been read
    """
    if self.dimensionVariableHandle: # If the handle is None then the values were created already
      if self.slice2:
        cMin = 1.5*self.dimensionVariableHandle[0] - 0.5*self.dimensionVariableHandle[1]
        cMax = 1.5*self.dimensionVariableHandle[-1] - 0.5*self.dimensionVariableHandle[-2]
        self.values = np.append(self.dimensionVariableHandle[self.slice1], self.dimensionVariableHandle[self.slice2]+(cMax-cMin))
      else: self.values = self.dimensionVariableHandle[self.slice1]
    if self.len>1:
      cMin = 1.5*self.values[0] - 0.5*self.values[1]
      cMax = 1.5*self.values[-1] - 0.5*self.values[-2]
    else: cMin = self.values[0]; cMax = cMin
    self.limits = (cMin, cMax)
    if debug: print self
  def __repr__(self):
    return 'len=%i, name="%s", units=%s, label="%s"'%(self.len, self.name, self.units, self.label)+' min/max='+repr(self.limits)+' slice1='+repr(self.slice1)+' slice2='+repr(self.slice2) #+' values='+repr(self.values)


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
    dims=[]
    for d,s in zip(variableDims, sliceSpecs):
      dims.append( NetcdfDim(rootGroup, d, s) )

    # Group singleton dimensions and active dimensions
    activeDims = []; singleDims = []
    for d in dims:
      if d.len==1: singleDims.append(d)
      else: activeDims.append(d)

    # Attributes of class
    self.variableHandle = variableHandle
    self.allDims = dims
    self.dims = activeDims
    self.singleDims = singleDims
    self.data = None
    self.label, self.name, self.units = constructLabel(variableHandle, variableName)
    self.rank = len(self.dims)
  def getData(self):
    """
    Popolate NetcdfSlice.data with data from file
    """
    slices1 = []; slices2 = []
    for d in self.allDims:
      d.getData()
      if d.slice2: slices1.append( d.slice1 ); slices2.append( d.slice2 )
      else: slices1.append( d.slice1 ); slices2.append( d.slice1 )
    if slices1==slices2:
      self.data = np.squeeze( self.variableHandle[slices1] )
    else:
      self.data = np.ma.concatenate(
        ( np.squeeze( self.variableHandle[slices1] ),
          np.squeeze( self.variableHandle[slices2] ) ), axis=1)


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


def constructLabel(ncObj, default=''):
  """
  Returns a string combining CF attiributes "long_name" and "units"
  """
  if debug: print 'constructLabel: ncObj=',ncObj
  label = ''; name = None
  if 'long_name' in ncObj.ncattrs():
    label += str(ncObj.long_name)
  else: label += ncObj._name
  name = label; units = None
  if 'units' in ncObj.ncattrs():
    units = str(ncObj.units)
    label += ' ('+units+')'
  if len(label)==0: label = default+' (index)'
  return label, name ,units


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
      if 'long_name' in obj.ncattrs(): oString += ' "'+obj.long_name+'"'
      if 'units' in obj.ncattrs(): oString += ' ('+obj.units+')'
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
    if 'long_name' in obj.ncattrs(): oString += ' "'+obj.long_name+'"'
    if 'units' in obj.ncattrs(): oString += ' ('+obj.units+')'
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


def extrapCoord(xCell):
  """
  Returns the (extrapolated/interpolated) positions of vertices, derived from cell center positions
  """
  newCoord = 0.5*( xCell[0:-1] + xCell[1:] )
  newCoord = np.insert(newCoord, 0, 1.5*xCell[0] - 0.5*xCell[1])
  newCoord = np.append(newCoord, [1.5*xCell[-1] - 0.5*xCell[-2]])
  return newCoord


def readSGvar(fileName, varName, varDims):
  try: rg = Dataset(fileName,'r')
  except:
    if os.path.isfile(fileName): raise MyError('There was a problem opening "'+fileName+'".')
    raise MyError('Could not find file "'+fileName+'".')

  if not varName in rg.variables:
    raise MyError('Could not find %s in %s'%(varName,fileName))

  dims = rg.dimensions
  xVarDim = None; yVarDim = None
  for d in varDims:
    if 2*d.lenInFile==len(dims['nx']):
      if xVarDim: raise MyError('Too many dimensions matches for nx')
      else: xVarDim = d
    if 2*d.lenInFile==len(dims['ny']):
      if yVarDim: raise MyError('Too many dimensions matches for nx')
      else: yVarDim = d
  xSlice1 = slice(xVarDim.slice1.start*2, xVarDim.slice1.stop*2+1, 2)
  ySlice1 = slice(yVarDim.slice1.start*2, yVarDim.slice1.stop*2+1, 2)
  if xVarDim.slice2==None:
    cData = rg.variables[varName][ySlice1,xSlice1]
  else:
    xSlice2 = slice(xVarDim.slice2.start*2+1, xVarDim.slice2.stop*2+1, 2)
    ySlice2 = slice(yVarDim.slice1.start*2, yVarDim.slice1.stop*2+1, 2)
    cData1 = rg.variables[varName][ySlice1,xSlice1]
    cData2 = rg.variables[varName][ySlice2,xSlice2]
    if varName=='x': cData2 = cData2 + 361.
    cData = np.append( cData1, cData2, axis=1)
  cMin = np.min( cData[:,0] ); cMin = min( cMin, np.min( cData[:,-1] ) )
  cMin = min( cMin, np.min( cData[0,:] ) ); cMin = min( cMin, np.min( cData[-1,:] ) )
  cMax = np.max( cData[:,0] ); cMax = max( cMax, np.max( cData[:,-1] ) )
  cMax = max( cMax, np.max( cData[0,:] ) ); cMax = max( cMax, np.max( cData[-1,:] ) )
  return cData, (cMin, cMax)


# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()
