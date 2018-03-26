#!/usr/bin/env python

# Import functions from gplot.py
from gplot import *

# Try to import required packages/modules
import re
import os
import time
import copy
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

debug = False # Global debugging
warnings.simplefilter('error', UserWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')


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
      gcompare.py visualizes a 1- or 2-D field from two datasets and shows the difference,
      all in a 3-panel plot.
      ''',
      epilog='Written by A.Adcroft, 2013.')
  parser.add_argument('file_var_slice1', type=str,
      metavar='FILE1[,VARIABLE1[,SLICE1[,SLICE2[...]]]]',
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
  parser.add_argument('file_var_slice2', type=str,
      metavar='FILE2[,VARIABLE2[,SLICE1[,SLICE2[...]]]]',
      help='''
      As for argument 1. If the SLICE specifications are missing, those from argument 1 will be assumed.
      If VARIABLE2 is missing the the same variable name from argument 1 will be assumed.
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
  parser.add_argument('--dlim', type=float, nargs=2, metavar=('MIN','MAX'),
      help='''Specify the minimum/maximum color range for the difference plot.
      Values outside this range will be clipped to the minimum or maximum.''')
  parser.add_argument('--panels', type=int, choices=list(range(1,4)), default=3,
      help='''Number of panels to show. 3 panels shows A, B and A-B; 2 panels
      displays A and B; 1 panel shows just A-B.''')
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
  parser.add_argument('--static2', action='store_true',
      help='Hold constant the unlimited dimension index of second field when animating.')
  parser.add_argument('-o','--output', type=str, default='',
      help='Name of image file to create.')
  parser.add_argument('-r','--resolution', type=int,
      help='Vertial resolution (in pixels) for image, e.g. 720 would give 720p video resolution. Default is 1024 pixels.')
  parser.add_argument('-ar','--aspect', type=float, nargs=2, metavar=('WIDTH','HEIGHT'),
      help='An aspect ratio for image such as 16 9 (widescreen) or 4 3. Default is 3 4.')
  parser.add_argument('--stats', action='store_true',
      help='Print the statistics of viewed data.')
  parser.add_argument('--list', action='store_true',
      help='Print selected data to terminal.')
  parser.add_argument('-d','--debug', action='store_true',
      help='Turn on debugging information.')
  optCmdLineArgs = parser.parse_args()

  if optCmdLineArgs.aspect==None:
    if optCmdLineArgs.panels==1: optCmdLineArgs.aspect=[16., 9.]
    elif optCmdLineArgs.panels==2: optCmdLineArgs.aspect=[4., 3.]
    elif optCmdLineArgs.panels==3: optCmdLineArgs.aspect=[3., 4.]
  if optCmdLineArgs.resolution==None:
    if optCmdLineArgs.panels==1: optCmdLineArgs.resolution=600
    elif optCmdLineArgs.panels==2: optCmdLineArgs.resolution=800
    elif optCmdLineArgs.panels==3: optCmdLineArgs.resolution=1024

  if optCmdLineArgs.debug: debug = True ; enableDebugging()

  createUI(optCmdLineArgs.file_var_slice1, optCmdLineArgs.file_var_slice2, optCmdLineArgs)


def createUI(fileVarSlice1, fileVarSlice2, args):
  """
  Generates a  3-panel plot based on the files/variables/slices specified
  """

  # Extract file, variable and slice specs from fileVarSlice1
  if debug: print('createUI: fileVarSlice1=',fileVarSlice1)
  (fileName1, variableName1, sliceSpecs1) = splitFileVarPos(fileVarSlice1)
  if debug: print('createUI: fileName1=',fileName1,'variableName1=',variableName1,'sliceSpecs1=',sliceSpecs1)

  # Extract file, variable and slice specs from fileVarSlice2
  if debug: print('createUI: fileVarSlice2=',fileVarSlice2)
  (fileName2, variableName2, sliceSpecs2) = splitFileVarPos(fileVarSlice2)
  if os.path.isdir(fileName2):
    fileName2=os.path.join(fileName2,os.path.basename(fileName1))
  if sliceSpecs2==None: sliceSpecs2 = sliceSpecs1
  if variableName2==None: variableName2 = variableName1
  if debug: print('createUI: fileName2=',fileName2,'variableName2=',variableName2,'sliceSpecs2=',sliceSpecs2)

  # Read the meta-data for elevation, if asked for (needed for section plots)
  if args.elevation:
    (elevFileName, elevVariableName, elevSliceSpecs) = splitFileVarPos(args.elevation)
    if elevSliceSpecs==None: elevSliceSpecs = sliceSpecs1
    if elevVariableName==None: elevVariableName='elevation'
    if debug: print('elevFileName=',elevFileName,'eName=',elevVariableName,'eSlice=',elevSliceSpecs)
    eRg, eVar = readVariableFromFile(elevFileName, elevVariableName, elevSliceSpecs,
        ignoreCoords=args.indices, alternativeNames=['elev', 'e', 'h'])
  else: eVar = None

  # Read the meta-data for the variable to be plotted
  rg1, var1 = readVariableFromFile(fileName1, variableName1, sliceSpecs1, ignoreCoords=args.indices)
  rg2, var2 = readVariableFromFile(fileName2, variableName2, sliceSpecs2, ignoreCoords=args.indices)

  if not args.static2 and var1.rank!=var2.rank:
    raise MyError('%s and %s have different ranks'%(variableName1,variableName2))

  # Set figure shape
  setFigureSize(args.aspect[0]/args.aspect[1], args.resolution)

  # Based on rank, either create interactive plot, animate or intercept requests for rank >2
  if var1.rank==3 and args.animate and not var1.unlimitedDim==None:
    n0 = var1.unlimitedDim.slice1.start; n1 = var1.unlimitedDim.slice1.stop
    var1.rank = 2; var1.unlimitedDim.len = 1
    var1.singleDims.insert(0, var1.unlimitedDim)
    var1.dims.remove(var1.unlimitedDim)
    if not args.static2:
      var2.rank = 2; var2.unlimitedDim.len = 1
      var2.singleDims.insert(0, var2.unlimitedDim)
      var2.dims.remove(var2.unlimitedDim)
    for n in range(n0,n1):
      var1.singleDims[0].slice1 = slice(n,n+1)
      var1.singleDims[0].getData(forceRead=True)
      if not args.static2:
        var2.singleDims[0].slice1 = slice(n,n+1)
        var2.singleDims[0].getData(forceRead=True)
      if n>0:
        if args.output:
          plt.close(); setFigureSize(args.aspect[0]/args.aspect[1], args.resolution)
        else: plt.clf()
      #render(var1, args, frame=n+1)
      render3panels(fileName1, var1, fileName2, var2, eVar, args, frame=n+1)
      if not args.output:
        if n==n0: plt.show(block=False)
        else: plt.draw()
  elif var1.rank>2:
    summarizeFile(rg1); print()
    summarizeFile(rg2); print()
    raise MyError( 'Variable name "%s" has resolved rank %i. Only 1D and 2D data can be plotted until you buy a holographic display.'%(variableName1, var1.rank))
  else:
    render3panels(fileName1, var1, fileName2, var2, eVar, args, frame=0)
    if not args.output: plt.show()

def render3panels(fileName1, var1, fileName2, var2, eVar, args, frame):
  nPanels = args.panels
  if nPanels==3: plt.gcf().subplots_adjust(left=.10, right=.97, wspace=0, bottom=.05, top=.9, hspace=.2)
  else: plt.gcf().subplots_adjust(left=.10, right=.97, wspace=0, bottom=.09, top=.9, hspace=.2)
  var1.getData() # Actually read data from file
  var2.getData() # Actually read data from file
  if nPanels>1:
    plt.subplot(nPanels,1,1)
    clim = render(var1, args, elevation=eVar, frame=frame)
    plt.title('A:  %s'%fileName1)
    plt.subplot(nPanels,1,2)
    args.clim = clim
    render(var2, args, elevation=eVar, frame=frame, skipXlabel=(nPanels!=2))
    plt.title('B:  %s'%fileName2)
  if nPanels==3:
    plt.subplot(nPanels,1,3)
    plt.title('A - B')
  elif nPanels==1:
    plt.title('%s - %s'%(fileName1,fileName2))
  plt.suptitle(var1.label, fontsize=18)
  if nPanels in [1,3]:
    varDiff = copy.copy(var1)
    varDiff.data = var1.data - var2.data
    render(varDiff, args, elevation=eVar, skipXlabel=False, ignoreClim=True, frame=frame)


def render(var, args, elevation=None, frame=0, skipXlabel=True, skipTitle=True, ignoreClim=False):
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
  if args.log10: var.data = np.log10(var.data)

  # Now plot
  clim = None
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
      elevation.getData()
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
    if not skipTitle:
      if args.scale: plt.title(var.label+' x%e'%(args.scale[0]),fontsize=fontsize)
      else: plt.title(var.label,fontsize=fontsize)
    plt.xlim(xLims); plt.ylim(yLims)
    if not skipXlabel: plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if ignoreClim: makeGuessAboutCmap(clim=args.dlim, colormap=args.colormap)
    else: clim = makeGuessAboutCmap(clim=args.clim, colormap=args.colormap)
    plt.colorbar(fraction=.08)
  axis=plt.gca()
  if var.singleDims:
    text = ''
    for d in var.singleDims:
      if len(text): text = text+'   '
      text = text + d.name + ' = ' + str(d.values[0])
      if d.units: text = text + ' (' + d.units + ')'
    if not skipXlabel: axis.annotate(text, xy=(0.005,.995), xycoords='figure fraction', verticalalignment='top', fontsize=8)
  if args.output:
    if args.animate:
      if not skipXlabel:
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
  return clim

# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()
