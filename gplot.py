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
import time
try: import argparse
except: raise MyError('This version of python is not new enough. python 2.7 or newer is required.')
try: from netCDF4 import MFDataset, Dataset
except: raise MyError('Unable to import netCDF4 module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_netcdf4')
try: import numpy as np
except: raise MyError('Unable to import numpy module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_numpy')
try: import matplotlib.pyplot as plt
except: raise MyError('Unable to import matplotlib.pyplot module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_matplotlib')
import warnings

# Import stand alone (static) functions
import m6toolbox

debug = False # Global debugging
warnings.simplefilter('error', UserWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')
global_eVar = None # Global for averaging from within FnSlice


def parseCommandLine():
  """
  Parse the command line positional and optional arguments.
  This is the highest level procedure invoked from the very end of the script.
  """
  global debug # Declared global in order to set it
  global start_time
  start_time = time.time()

  # Arguments
  parser = argparse.ArgumentParser(description=
      '''
      gplot.py can plot 1- and 2-dimensional data, which itself can be extracted from
      multiple-dimensional data.
      ''',
      epilog='Written by A.Adcroft, 2013.')
  parser.add_argument('file_var_slice', type=str,
      metavar='FILE[,VARIABLE[,SLICE1[,SLICE2[...]]]]',
      help='''File, variable and slice specification. Valid forms include filename.nc ;
      filename.nc,variable ; filename.nc,variable,slice ; filename.nc,variable,slice1,slice2 ; etc.
      Each slice takes the form of single VALUE or [START]:[STOP] range.
      When VALUE, START or STOP are positive integers, they refer to indices (START at 1).
      In the range form (with a colon), a missing START or STOP will indicate the beginning or
      end of the dimension.
      An inverted range (START>STOP) indicates to wrap around a periodic dimension. 
      Any of VALUE, START or STOP can take the form '=POS' or '=POS1:POS2' in which case POS, POS1 and POS2 
      are coordinate values or ranges.
      ''')
  parser.add_argument('-cm','--colormap', type=str, default='',
      help=''' Specify the colormap. The default colormap is determined by the data.
      For single-signed data with some values near zero, the 'hot' colormap is used.
      For multi-signed data with near symmetric ranges, the 'seismic' colormap is used, and the color
      range automatically centered on zero.
      Otherwise, the 'spectral' colormap is used.
      See http://matplotlib.org/examples/color/colormaps_reference.html for a list of colormaps.
      ''')
  parser.add_argument('--clim', type=float, nargs=2, metavar=('MIN','MAX'),
      help='''Specify the minimum/maximum color range.
      Values outside this range will be clipped to the minimum or maximum.''')
  parser.add_argument('--ignore', type=float, nargs=1,
      help='Mask out the values equal to IGNORE.')
  parser.add_argument('--ignorelt', type=float, nargs=1,
      help='Mask out values less than IGNORELT.')
  parser.add_argument('--ignoregt', type=float, nargs=1,
      help='Mask out values less than IGNOREGT.')
  parser.add_argument('--scale', type=float, nargs=1,
      help='The factor to multiply by before plotting.')
  parser.add_argument('--offset', type=float, nargs=1,
      help='An offset to add before plotting.')
  parser.add_argument('--log10', action='store_true',
      help='Take the logarithm (base 10) of data before plotting.')
  parser.add_argument('-sg','--supergrid', type=str, default=None,
      help='The super-grid to use for horizontal coordinates.')
  parser.add_argument('-os','--oceanstatic', type=str, default=None,
      help='The ocean_static file to use for horizontal coordinates.')
  parser.add_argument('-IJ','--indices', action='store_true',
      help='Use memory indices for coordinates.')
  parser.add_argument('-e','--elevation', type=str, default=None,
      help='The file[,variable] from which to read elevation for vertical section plots.')
  parser.add_argument('--animate', action='store_true',
      help='Animate over the unlimited dimension.')
  parser.add_argument('-o','--output', type=str, default='',
      help='Name of image file to create.')
  parser.add_argument('-r','--resolution', type=int, default=600,
      help='Vertial resolution (in pixels) for image, e.g. 720 would give 720p video resolution. Default is 600 pixels.')
  parser.add_argument('-ar','--aspect', type=float, nargs=2, default=[16., 9.], metavar=('WIDTH','HEIGHT'),
      help='An aspect ratio for image such as 16 9 (widescreen) or 4 3. Default is 16 9.')
  parser.add_argument('--stats', action='store_true',
      help='Print the statistics of viewed data.')
  parser.add_argument('--list', action='store_true',
      help='Print selected data to terminal.')
  parser.add_argument('-d','--debug', action='store_true',
      help='Turn on debugging information.')
  parser.add_argument('--unittests', action='store_true', help=argparse.SUPPRESS)
  optCmdLineArgs = parser.parse_args()

  if optCmdLineArgs.debug: enableDebugging()
  if optCmdLineArgs.unittests: unittests(optCmdLineArgs); return

  createUI(optCmdLineArgs.file_var_slice, optCmdLineArgs)


def createUI(fileVarSlice, args):
  """
  Generates a plot based on the file/variable/slice specified
  """

  # Extract file, variable and slice specs from fileVarSlice
  if debug: print('createUI: fileVarSlice=',fileVarSlice)
  (fileName, variableName, sliceSpecs) = splitFileVarPos(fileVarSlice)
  if debug: print('createUI: fileName=',fileName,'variableName=',variableName,'sliceSpecs=',sliceSpecs)

  # Read the meta-data for elevation, if asked for (needed for section plots)
  if args.elevation:
    if args.elevation == 'same': (elevFileName, elevVariableName, elevSliceSpecs) = splitFileVarPos(fileName)
    else: (elevFileName, elevVariableName, elevSliceSpecs) = splitFileVarPos(args.elevation)
    if elevSliceSpecs==None: elevSliceSpecs = sliceSpecs
    if elevVariableName==None: elevVariableName='elevation'
    if debug: print('elevFileName=',elevFileName,'eName=',elevVariableName,'eSlice=',elevSliceSpecs)
    eRg, eVar = readVariableFromFile(elevFileName, elevVariableName, elevSliceSpecs,
        ignoreCoords=args.indices, alternativeNames=['elev', 'e', 'h'])
    global global_eVar
    global_eVar = eVar
  else: eVar = None

  # Read the meta-data for the variable to be plotted
  rg, var = readVariableFromFile(fileName, variableName, sliceSpecs, ignoreCoords=args.indices)

  # Set figure shape
  setFigureSize(args.aspect[0]/args.aspect[1], args.resolution)

  # Based on rank, either create interactive plot, animate or intercept requests for rank >2
  if var.rank==3 and args.animate and not var.unlimitedDim==None:
    n0 = var.unlimitedDim.slice1.start; n1 = var.unlimitedDim.slice1.stop
    var.rank = 2; var.unlimitedDim.len = 1
    var.singleDims.insert(0, var.unlimitedDim)
    var.dims.remove(var.unlimitedDim)
    if not eVar == None:
      eVar.rank = 2; eVar.unlimitedDim.len = 1
      eVar.singleDims.insert(0, eVar.unlimitedDim)
      eVar.dims.remove(eVar.unlimitedDim)
    for n in range(n0,n1):
      var.singleDims[0].slice1 = slice(n,n+1)
      var.singleDims[0].getData(forceRead=True)
      if not eVar == None:
        eVar.singleDims[0].slice1 = slice(n,n+1)
        eVar.singleDims[0].getData(forceRead=True)
      if n>0:
        if args.output:
          plt.close(); setFigureSize(args.aspect[0]/args.aspect[1], args.resolution)
        else: plt.clf()
      render(var, args, frame=n+1, elevation=eVar)
      if not args.output:
        if n==n0: plt.show(block=False)
        else: plt.draw()
  elif var.rank>2:
    summarizeFile(rg); print()
    raise MyError( 'Variable name "%s" has resolved rank %i. Only 1D and 2D data can be plotted until you buy a holographic display.'%(variableName, var.rank))
  else:
    render(var, args, elevation=eVar)
    if not args.output: plt.show()
  

def render(var, args, elevation=None, frame=0):
  var.getData() # Actually read data from file
  # Optionally mask out a specific value
  if args.ignore:
    var.data = np.ma.masked_array(var.data, mask=[var.data==args.ignore])
  if args.ignorelt:
    var.data = np.ma.masked_array(var.data, mask=[var.data<=args.ignorelt])
  if args.ignoregt:
    var.data = np.ma.masked_array(var.data, mask=[var.data>=args.ignoregt])
  if args.scale:
    var.data = args.scale * var.data

  if args.list: print('createUI: Data =\n',var.data)
  if args.stats:
    dMin = np.min(var.data); dMax = np.max(var.data)
    if dMin==0 and dMax>0:
      dMin = np.min(var.data[var.data!=0])
      print('Mininum=',dMin,'(ignoring zeros) Maximum=',dMax)
    elif dMax==0 and dMin<0:
      dMax = np.max(var.data[var.data!=0])
      print('Mininum=',dMin,'Maximum=',dMax,'(ignoring zeros)')
    else: print('Mininum=',dMin,'Maximum=',dMax)

  if args.offset: var.data = var.data + args.offset
  if args.log10: var.data = np.ma.log10(var.data)

  # Now plot
  if var.rank==0:
    for d in var.allDims:
      print('%s = %g %s'%(d.name,d.values[0],d.units))
    print(var.vname+' = ',var.data,'   '+var.units)
    exit(0)
  elif var.rank==1: # Line plot
    if var.dims[0].isZaxis: # Transpose 1d plot
      xCoord = var.data ; yData = var.dims[0].values
      plt.plot(xCoord, yData)
      plt.xlabel(var.label)
      plt.ylabel(var.dims[0].label);
      if var.dims[0].values[0]>var.dims[0].values[-1]: plt.gca().invert_yaxis()
      if var.dims[0].positiveDown: plt.gca().invert_yaxis()
    else: # Normal 1d plot
      xCoord = var.dims[0].values; yData = var.data
      plt.plot(xCoord, yData)
      plt.xlabel(var.dims[0].label); plt.xlim(var.dims[0].limits[0], var.dims[0].limits[-1])
      plt.ylabel(var.label)
  elif var.rank==2: # Pseudo color plot
    # Add an extra element to coordinate to force pcolormesh to draw all cells
    if var.dims[1].isZaxis: # Happens for S(t,z)
      xCoord = extrapCoord( var.dims[0].values); yCoord = extrapCoord( var.dims[1].values)
      zData = np.transpose(var.data)
      xLabel = var.dims[0].label; xLims = var.dims[0].limits
      yLabel = var.dims[1].label; yLims = var.dims[1].limits
      yDim = var.dims[1]
    else:
      xLabel = var.dims[1].label; xLims = var.dims[1].limits
      yLabel = var.dims[0].label; yLims = var.dims[0].limits
      if args.supergrid==None:
        if args.oceanstatic==None:
          xCoord = extrapCoord( var.dims[1].values); yCoord = extrapCoord( var.dims[0].values)
        else:
          xCoord, xLims = readOSvar(args.oceanstatic, 'geolon_c', var.dims)
          yCoord, yLims = readOSvar(args.oceanstatic, 'geolat_c', var.dims)
          xLabel = 'Longitude (\u00B0E)' ; yLabel = 'Latitude (\u00B0N)'
      else:
        xCoord, xLims = readSGvar(args.supergrid, 'x', var.dims)
        yCoord, yLims = readSGvar(args.supergrid, 'y', var.dims)
        xLabel = 'Longitude (\u00B0E)' ; yLabel = 'Latitude (\u00B0N)'
      zData = var.data
      yDim = var.dims[0]
    if yDim.isZaxis and not elevation==None: # Z on y axis ?
      if elevation.refreshable: elevation.getData()
      #yCoord = elevation.data
      xCoord, yCoord, zData = m6toolbox.section2quadmesh(xCoord, elevation.data, zData, representation='pcm')
      yLims = (np.amin(yCoord[-1,:]), np.amax(yCoord[0,:]))
      #yCoord = extrapElevation( yCoord )
      yLabel = 'Elevation (m)'
    plt.pcolormesh(xCoord,yCoord,zData)
    if yDim.isZaxis and elevation==None: # Z on y axis ?
      if yCoord[0]>yCoord[-1]: plt.gca().invert_yaxis(); yLims = reversed(yLims)
      if yDim.positiveDown: plt.gca().invert_yaxis(); yLims = reversed(yLims)
    if len(var.label)>50: fontsize=10
    elif len(var.label)>30: fontsize=12
    else: fontsize=14;
    if args.scale: plt.title(var.label+' x%e'%(args.scale[0]),fontsize=fontsize)
    else: plt.title(var.label,fontsize=fontsize)
    plt.xlim(xLims); plt.ylim(yLims)
    plt.xlabel(xLabel) ; plt.ylabel(yLabel)
    makeGuessAboutCmap(clim=args.clim, colormap=args.colormap)
    plt.tight_layout()
    plt.colorbar(fraction=.08)
  axis=plt.gca()
  if var.singleDims:
    text = ''
    for d in var.singleDims:
      if len(text): text = text+'   '
      text = text + d.name + ' = ' + str(d.values[0])
      if d.units: text = text + ' (' + d.units + ')'
    axis.annotate(text, xy=(0.005,.995), xycoords='figure fraction', verticalalignment='top', fontsize=8)
  if args.output:
    if args.animate:
      dt = time.time() - start_time
      nf = var.singleDims[0].initialLen
      print('Writing file "%s" (%i/%i)'%(args.output%(frame),frame,nf), \
            'Elapsed %.1fs, %.2f FPS, total %.1fs, remaining %.1fs'%(dt, frame/dt, 1.*nf/frame*dt, (1.*nf/frame-1.)*dt))
      try: plt.savefig(args.output%(frame),pad_inches=0.)
      except: raise MyError('output filename must contain %D.Di when animating')
    else: plt.savefig(args.output,pad_inches=0.)
  elif not args.animate: # Interactive and static
    def keyPress(event):
      if event.key=='q': exit(0)
    if var.rank==1:
      def statusMesg(x,y):
        # -1 needed because of extension for pcolormesh
        i = min(list(range(len(xCoord)-1)), key=lambda l: abs(xCoord[l]-x))
        if not i==None:
          val = yData[i]
          if val is np.ma.masked: return 'x=%.3f  %s(%i)=NaN'%(x,var.vname,i+1)
          else: return 'x=%.3f  %s(%i)=%g'%(x,var.vname,i+1,val)
        else: return 'x=%.3f y=%.3f'%(x,y)
    elif var.rank==2:
      def statusMesg(x,y):
        if len(xCoord.shape)==1:
          # -2 needed because of coords are for vertices and need to be averaged to centers
          i = min(list(range(len(xCoord)-2)), key=lambda l: abs((xCoord[l]+xCoord[l+1])/2.-x))
          j = min(list(range(len(yCoord)-2)), key=lambda l: abs((yCoord[l]+yCoord[l+1])/2.-y))
        else:
          idx = np.abs( np.fabs( xCoord[0:-1,0:-1]+xCoord[1:,1:]+xCoord[0:-1,1:]+xCoord[1:,0:-1]-4*x)
              +np.fabs( yCoord[0:-1,0:-1]+yCoord[1:,1:]+yCoord[0:-1,1:]+yCoord[1:,0:-1]-4*y) ).argmin()
          j,i = np.unravel_index(idx,zData.shape)
        if not i==None:
          val = zData[j,i]
          if val is np.ma.masked: return 'x,y=%.3f,%.3f  %s(%i,%i)=NaN'%(x,y,var.vname,i+1,j+1)
          else: return 'x,y=%.3f,%.3f  %s(%i,%i)=%g'%(x,y,var.vname,i+1,j+1,val)
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


def readVariableFromFile(fileName, variableName, sliceSpecs, ignoreCoords=False, alternativeNames=None):
  """
  Open netCDF file, find and read the variable meta-information and return both
  the netcdf object and variable object
  """
  # Open netcdf file
  try: rg = MFDataset(fileName, 'r', aggdim='time')
  except:
    if debug: print('Unable to open %s with MFDataset'%(fileName))
    try: rg = Dataset(fileName, 'r')
    except:
      if os.path.isfile(fileName): raise MyError('There was a problem opening "'+fileName+'".')
      raise MyError('Could not find file "'+fileName+'".')

  # If no variable is specified, summarize the file contents and exit
  if not variableName:
    print('No variable name specified! Specify a varible from the following summary of "'\
          +fileName+'":\n')
    summarizeFile(rg)
    exit(0)

  # Intercept the functions of variables
  if isFunction(variableName): return rg, FnSlice(rg, variableName, sliceSpecs, ignoreCoords=ignoreCoords)

  # Check that the variable is in the file (allowing for case mismatch)
  for v in rg.variables:
    if variableName.lower() == v.lower(): variableName=v ; break
  if not variableName in rg.variables:
    if alternativeNames==None:
      print('Known variables in file: '+''.join( (str(v)+', ' for v in rg.variables) ))
      raise MyError('Did not find "'+variableName+'" in file "'+fileName+'".')
    else:
      for v in alternativeNames:
        if v in rg.variables: variableName=v ; break

  # Obtain meta data along with 1D coordinates, labels and limits
  return rg, NetcdfSlice(rg, variableName, sliceSpecs, ignoreCoords=ignoreCoords)


class NetcdfDim:
  """
  Class for describing a dimension in a netcdf file
  """
  def __init__(self, rootGroup, dimensionName, sliceSpec, ignoreCoords=False):
    """
    Initialize a dimension by interpreting a sliceSpec
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
    if debug: print('NetcdfDim: Interpreting "%s", groups='%(sliceSpec),equalsSplit.groups())
    lhsEquals, lhs, equals, rhs, low, colon, high, excess = equalsSplit.groups()
    if len(excess)>0: raise MyError('Syntax error: could not interpret "'+sliceSpec+'".')
    if len(rhs)==0: raise MyError('Syntax error: could not find range on RHS of "'+sliceSpec+'".')
    if debug:
      print('NetcdfDim: Interpreting "%s", name = "%s"' % (sliceSpec, lhs))
      print('NetcdfDim: Interpreting "%s", equals provided "%s"' % (sliceSpec, equals))
      print('NetcdfDim: Interpreting "%s", ranges provided "%s"' % (sliceSpec, colon))
      print('NetcdfDim: Interpreting "%s", low range "%s"' % (sliceSpec, low))
      print('NetcdfDim: Interpreting "%s", high range "%s"' % (sliceSpec, high))
    dimensionHandle = rootGroup.dimensions[dimensionName]
    self.isZaxis = False
    self.positiveDown = None
    if dimensionName in rootGroup.variables and not ignoreCoords:
      dimensionVariableHandle = rootGroup.variables[dimensionName]
      dimensionValues = None
      self.label, self.name, self.units = constructLabel(dimensionVariableHandle, dimensionName)
      if isAttrEqualTo(dimensionVariableHandle,'cartesian_axis','z') or \
        dimensionName.lower() in ('depth','z'):
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
        indexBegin = min(list(range(len(dimensionValues))), key=lambda i: abs(dimensionValues[i]-fLow))
        if indexBegin==0 and isLongitude and float(low)<cMin:
          indexBegin = min(list(range(len(dimensionValues))), key=lambda i: abs(dimensionValues[i]-fLow-360.))
      if colon==None: indexEnd = indexBegin
      else:
        if high=='': indexEnd = len(dimensionHandle) - 1
        else:
          indexEnd = min(list(range(len(dimensionValues))), key=lambda i: abs(dimensionValues[i]-float(high)))
          if indexEnd==len(dimensionValues)-1 and isLongitude and float(high)>cMax:
            indexEnd = min(list(range(len(dimensionValues))), key=lambda i: abs(dimensionValues[i]-float(high)+360.))
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
    self.isUnlimited = dimensionHandle.isunlimited()
    self.initialLen = self.len
  def getData(self, forceRead=False):
    """
    Read dimension variable data if it has not been read
    """
    #if not self.values==None: return # Already read
    if self.values==None or forceRead: # If the handle is None then the values were created already
      if self.dimensionVariableHandle==None:
        self.values = np.array(list(range(self.slice1.start, self.slice1.stop))) + 1
      else:
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
    if debug: print('NetcdfDim.getData: ',self)
  def __repr__(self):
    return 'len=%i, name="%s", units=%s, label="%s"'%(self.len, self.name, self.units, self.label)+' min/max='+repr(self.limits)+' slice1='+repr(self.slice1)+' slice2='+repr(self.slice2) #+' values='+repr(self.values)


class NetcdfSlice:
  """
  Class for reading a slice of data from a netcdf file using convenient index or coordinate ranges.
  """
  def __init__(self, rootGroup, variableName, sliceSpecs, ignoreCoords=False):
    """
    Match each slice listed in sliceSpecs with a dimension of variableName in rootGroup and read
    on that corresponding subset of data
    """
    variableHandle = rootGroup.variables[variableName]
    if debug: print('NetcdfSlice: variableName=',variableName)
    variableDims = variableHandle.dimensions
    if debug: print('NetcdfSlice: variableDims=',variableDims)
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
      print('NetcdfSlice: generalSlices=',generalSlices)
      print('NetcdfSlice: namedSlices=',namedSlices)
  
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
    if debug: print('NetcdfSlice: sliceSpecs=',sliceSpecs)
    if len(namedSlices): raise MyError('The named dimension in "%s" is not a dimension of the variable "%s".'
                                % (namedSlices[0], variableName) )
    if len(generalSlices): raise MyError('There is an impossible problem. I should probably be debugged.')
  
    # Now interpret the slice specification for each dimensions
    dims=[]
    for d,s in zip(variableDims, sliceSpecs):
      dims.append( NetcdfDim(rootGroup, d, s, ignoreCoords=ignoreCoords) )

    # Group singleton dimensions and active dimensions
    activeDims = []; singleDims = []
    self.unlimitedDim = None
    for d in dims:
      if d.len==1: singleDims.append(d)
      else: activeDims.append(d)
      if d.isUnlimited: self.unlimitedDim = d

    # Attributes of class
    self.variableHandle = variableHandle
    self.allDims = dims
    self.dims = activeDims
    self.singleDims = singleDims
    self.data = None
    self.label, self.name, self.units = constructLabel(variableHandle, variableName)
    self.vname = variableName
    self.rank = len(self.dims)
    self.refreshable = True
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


class FnSlice:
  """
  Class for reading a function of variables from a netcdf file.
  """
  def __init__(self, rootGroup, fnString, sliceSpecs, ignoreCoords=False):
    """
    Interpret F(x,y,...), assaciate a NetcdfSlice for each of x,y,... and
    apply F() when getting data.
    """
    m = re.match('(\w+)\(([\w,]+)\)',fnString)
    self.function = m.group(1)
    varNetcdfSlices = []
    for v in m.group(2).split(','):
      varNetcdfSlices.append( NetcdfSlice(rootGroup, v, sliceSpecs, ignoreCoords=ignoreCoords) )
    self.vars = varNetcdfSlices
    self.rank = varNetcdfSlices[0].rank
    self.dims = varNetcdfSlices[0].dims
    self.singleDims = varNetcdfSlices[0].singleDims
    self.label = fnString
    self.vname = fnString
    self.data = None
    if self.function.lower() in ['xave', 'xpsi']:
      self.rank = self.rank - 1
      #del self.dims[-1]
    elif self.function.lower() in ['tave']:
      self.rank = self.rank - 1
      del self.dims[0]
  def getData(self):
    """
    Popolate FnfSlice.data with data from file
    """
    for v in self.vars:
      v.getData()
    if self.function.lower() == 'sigma0':
      self.data = m6toolbox.rho_Wright97(self.vars[0].data, self.vars[1].data, 0)
    elif self.function.lower() == 'sigma2':
      self.data = m6toolbox.rho_Wright97(self.vars[0].data, self.vars[1].data, 2e7)
    elif self.function.lower() == 'sigma4':
      self.data = m6toolbox.rho_Wright97(self.vars[0].data, self.vars[1].data, 4e7)
    elif self.function.lower() == 'xave':
      global global_eVar
      if global_eVar==None: raise MyError('Elevation or thickness is necessary to compute a zonal average.')
      if global_eVar.data==None:
        global_eVar.getData()
        global_eVar.refreshable = False
      self.data, zOut, _ = m6toolbox.axisAverage( self.vars[0].data, z=global_eVar.data )
      global_eVar.data = zOut
    elif self.function.lower() == 'xpsi':
      xSum = np.sum(self.vars[0].data, axis=-1) # Zonal sum
      #psi1 = np.cumsum( xSum[:,::-1], axis=-2)
      #psi2 = np.cumsum( xSum, axis=-2)
      #self.data = psi1[:,::-1] - psi2
      nk, nj = xSum.shape
      #self.data = np.cumsum( xSum[:,::-1], axis=-2)[:,::-1]
      self.data = np.zeros((nk+1,nj))
      for k in range(nk,0,-1):
        self.data[k-1,:] = self.data[k,:] - xSum[k-1,:]
      if global_eVar!=None and global_eVar.data==None:
        global_eVar.getData()
        global_eVar.refreshable = False
        global_eVar.data = np.min(global_eVar.data, axis=-1)
        self.data = self.data[1:,:]
    elif self.function.lower() == 'tave':
      self.data = np.mean( self.vars[0].data, axis=0 )
    else: raise MyError('Unknown function: '+self.function)


def splitFileVarPos(string):
  """
  Split a string in form of "file,variable[...]" into three string parts
  Valid forms are "file", "file,variable" or "file,variable,3,j=,=2.,z=3.1:5.4,..."
  """
  m = re.match('([\w\.~/\*\-\[\]]+)[,:]?(.*)',string)
  fName = m.group(1)
  (vName, pSpecs) = splitVarPos(m.group(2))
  if debug: print('splitFileVarPos: fName=',fName,'vName=',vName,'pSpecs=',pSpecs)
  return fName, vName, pSpecs


def splitVarPos(string):
  """
  Split a string in form of "variable[...]" into two string parts
  Valid forms are "variable" or "variable[3,j=:,=2.,z=3.1:5.4,...,]"
  """
  vName = None; pSpecs = None
  if string:
    #cSplit = string.split(',')
    #if cSplit: vName = cSplit[0]
    #if len(cSplit)>1: pSpecs = cSplit[1:]

    #m = re.match('(\w+),?(.*)',string)
    m = re.match('((\w+)(\([\w,]+\))?),?(.*)',string)
    if m:
      vName = m.group(1)
      if m.group(4): pSpecs = m.group(4).split(',')
  if debug: print('splitVarPos: vName=',vName,'pSpecs=',pSpecs)
  return vName, pSpecs


def isFunction(string):
  """
  Detects whether a string takes the form of a function, F(x,y,...)
  """
  m = re.match('(\w+)\(([\w,]+)\)',string)
  if m: return True
  else: return False


def constructLabel(ncObj, default=''):
  """
  Returns a string combining CF attiributes "long_name" and "units"
  """
  label = ''; name = None
  if 'long_name' in ncObj.ncattrs():
    label += str(ncObj.long_name)
  else: label += ncObj._name
  name = label; units = None
  if 'units' in ncObj.ncattrs():
    units = str(ncObj.units)
    label += ' ('+units+')'
  if len(label)==0: label = default+' (index)'
  if debug: print('constructLabel: label,name,units=', label, name, units)
  return label, name ,units


def isAttrEqualTo(ncObj, name, value):
  """
  Returns True if ncObj has attribute "name" that matches "value"
  """
  if not ncObj: return False
  if name in ncObj.ncattrs():
    # .getncattr() works with Dataset but not MFDataset ???
    #if value.lower() in str(ncObj.getncattr(name)).lower():
    if value.lower() in str(ncObj.__getattribute__(name)).lower():
      return True
  return False


# Make an intelligent choice about which colormap to use
def makeGuessAboutCmap(clim=None, colormap=None):
  if clim:
    vmin, vmax = clim[0], clim[1]
  else:
    vmin, vmax = plt.gci().get_clim()
    cutOffFrac = 0.5
    if -vmin<vmax and -vmin/vmax>cutOffFrac: vmin=-vmax
    elif -vmin>vmax and -vmax/vmin>cutOffFrac: vmax=-vmin
  if vmin==vmax:
    if debug: print('vmin,vmax=',vmin,vmax)
    vmin = vmin - 1; vmax = vmax + 1
  plt.clim(vmin,vmax)
  if colormap: plt.set_cmap(colormap)
  else:
    if vmin*vmax>=0 and vmax>0 and 3*vmin<vmax: plt.set_cmap('hot') # Single signed +ve data
    elif vmin*vmax>=0 and vmin<0 and 3*vmax>vmin: plt.set_cmap('hot_r') # Single signed -ve data
    elif abs((vmax+vmin)/(vmax-vmin))<.01: plt.set_cmap('seismic') # Multi-signed symmetric data
    else: plt.set_cmap('spectral')
  landColor=[.5,.5,.5]
  plt.gca().set_facecolor(landColor)
  return (vmin, vmax)


# Generate a succinct summary of the netcdf file contents
def summarizeFile(rg):
  dims = rg.dimensions; vars = rg.variables
  print('Dimensions:')
  for dim in dims:
    oString = ' '+dim+' ['+str(len( dims[dim] ))+']'
    if dim in vars:
      n = len( dims[dim] ); obj = rg.variables[dim]
      if n>5: oString += ' = '+str(obj[0])+'...'+str(obj[n-1])
      else: oString += ' = '+str(obj[:])
      if 'long_name' in obj.ncattrs(): oString += ' "'+obj.long_name+'"'
      if 'units' in obj.ncattrs(): oString += ' ('+obj.units+')'
    print(oString)
  print(); print('Variables:')
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
    print(oString)


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


def extrapElevation(elev):
  """
  Returns the (extrapolated/interpolated) positions of vertices, derived from cell center positions
  """
  elev[elev.mask] = 0
  newElev = 0.5*( elev[:,0:-1] + elev[:,1:] )
  newElev = np.insert(newElev, 0, 1.5*elev[:,0] - 0.5*elev[:,1], axis=1)
  newElev = np.append(newElev, 1.5*elev[:,-1:] - 0.5*elev[:,-2:-1], axis=1)
  return newElev


def setFigureSize(aspect, verticalResolution):
  """
  Set the figure size based on vertical resolution and aspect ratio
  """
  width = int(aspect * verticalResolution) # First guess
  width = width + ( width % 2 ) # Make even
  plt.figure(figsize=(width/100., verticalResolution/100.)) # 100 dpi always?


def readSGvar(fileName, varName, varDims):
  """
  Read a variable from a super-grid file, which is usually at twice the resolution of
  the model grid.
  """
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


def readOSvar(fileName, varName, varDims):
  """
  Read a variable from an ocean_static file, which migh require extrapolation of corner data.
  """
  try: rg = Dataset(fileName,'r')
  except:
    if os.path.isfile(fileName): raise MyError('There was a problem opening "'+fileName+'".')
    raise MyError('Could not find file "'+fileName+'".')

  if not varName in rg.variables:
    raise MyError('Could not find %s in %s'%(varName,fileName))

  dims = rg.dimensions
  xVarDim = None; yVarDim = None
  for d in varDims:
    if d.lenInFile==len(dims['xq']):
      if xVarDim: raise MyError('Too many dimensions matches for nx')
      else: xVarDim = d
    if d.lenInFile==len(dims['yq']):
      if yVarDim: raise MyError('Too many dimensions matches for nx')
      else: yVarDim = d
  xSlice1 = slice(xVarDim.slice1.start, xVarDim.slice1.stop)
  ySlice1 = slice(yVarDim.slice1.start, yVarDim.slice1.stop)
  if xVarDim.slice2==None:
    cData = rg.variables[varName][ySlice1,xSlice1]
  else:
    xSlice2 = slice(xVarDim.slice2.start, xVarDim.slice2.stop)
    ySlice2 = slice(yVarDim.slice1.start, yVarDim.slice1.stop)
    cData1 = rg.variables[varName][ySlice1,xSlice1]
    cData2 = rg.variables[varName][ySlice2,xSlice2]
    if varName=='geolon_c': cData2 = cData2 + 361.
    cData = np.append( cData1, cData2, axis=1)
  if varName=='geolon_c':
    cMin = cData.min(); cMax = cData.max()
    if cMax-cMin>=360.: # Periodic and global
      for (j,i), value in np.ndenumerate(cData):
        if i>0 and value < cData[j,i-1]:
          if value+360.<=cMax: cData[j,i] = cData[j,i]+360.
          else: cData[j,i] = cMax
  cData = np.insert(cData, 0, 2.*cData[:,0]-cData[:,1], axis=1)
  cData = np.insert(cData, 0, 2.*cData[0,:]-cData[1,:], axis=0)
  cMin = np.min( cData[:,0] ); cMin = min( cMin, np.min( cData[:,-1] ) )
  cMin = min( cMin, np.min( cData[0,:] ) ); cMin = min( cMin, np.min( cData[-1,:] ) )
  cMax = np.max( cData[:,0] ); cMax = max( cMax, np.max( cData[:,-1] ) )
  cMax = max( cMax, np.max( cData[0,:] ) ); cMax = max( cMax, np.max( cData[-1,:] ) )
  return cData, (cMin, cMax)


def enableDebugging(newValue=True):
  """
  Sets the global parameter "debug" to control debugging information. This function is needed for
  controlling debugging of routine imported from gplot.py in other scripts.
  """
  global debug
  debug = newValue


def unittests(args):
  print(splitFileVarPos('file.nc'))
  print(splitFileVarPos('file.nc,variable'))
  print(splitFileVarPos('file.nc,variable,:'))
  print(splitFileVarPos('file.nc,variable,=-1:4,:'))
  print(splitFileVarPos('file.nc,variable(S,T),=-1:4,:'))
  print(splitFileVarPos(args.file_var_slice))


# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()
