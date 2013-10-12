#!/usr/bin/env python

# A simple error message generator with optional error code to return to shell
def error(msg, code=9):
  print 'Error: ' + msg
  exit(code)


# Try to import required packages/modules
import re
try: import argparse
except: error('This version of python is not new enough. python 2.7 or newer is required.')
try: from netCDF4 import Dataset
except: error('Unable to import netCDF4 module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_netcdf4')
try: import numpy as np
except: error('Unable to import numpy module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_numpy')
try: import matplotlib.pyplot as plt
except: error('Unable to import matplotlib.pyplot module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_matplotlib')

debug = False # Global debugging


# Parse the command line positional and optional arguments. This is the
# highest level procedure invoked from the very end of the script.
def parseCommandLine():
  global debug # Declared global in order to set it
  global optCmdLineArgs # For optional argument handling within routines

  # Arguments
  parser = argparse.ArgumentParser(description=
       'Yada yada yada',
       epilog='Written by A.Adcroft, 2013.')
  parser.add_argument('filename', type=str,
                      help='netcdf file to read.')
  parser.add_argument('variable', type=str,
                      nargs='?', default='',
                      help='Name of variable to plot. If absent a summary of file contents will be issued.')
  parser.add_argument('pos', type=str,
                      nargs='*', default='',
                      help='Indices or location specification.')
  parser.add_argument('-cm','--colormap', type=str, default='',
                      help='Specify the colormap.')
  parser.add_argument('--clim', type=float, nargs=2,
                      help='Specify the lower/upper color range.')
  parser.add_argument('-i','--ignore', type=float, nargs=1,
                      help='Mask out the given value.')
  parser.add_argument('-d','--debug', action='store_true',
                      help='Turn on debugging information.')
  parser.add_argument('-o','--output', type=str, default='',
                      help='Name of image file to create.')
  parser.add_argument('--stats', action='store_true',
                      help='Calculate statistics of viewed data.')
  parser.add_argument('--list', action='store_true',
                      help='Print selected data to screen.')
  optCmdLineArgs = parser.parse_args()

  if optCmdLineArgs.debug: debug = True

  processCommands(optCmdLineArgs.filename, optCmdLineArgs.variable, optCmdLineArgs.pos)


# processCommands() figures out the logic of what to actually do
def processCommands(fileName, variableName, sliceSpecs):
  # Open netcdf file
  try: rg=Dataset( fileName, 'r' );
  except: error('There was a problem opening "'+fileName+'".')

  # If no variable is specified, summarize the file contents
  if len(variableName)==0:
    print 'No variable name specified! Specify a varible from the following summary of "'\
          +fileName+'":'
    print
    summarizeFile(rg)
    exit(0)

  # Get the variable
  var = ''
  for v in rg.variables:
    if variableName.lower() == v.lower(): var=v
  if len(var)==0:
    print 'Known variables in file: '+''.join( (str(var)+', ' for var in rg.variables) )
    error('Did not find "'+variableName+'" in file "'+fileName+'".')
  var = rg.variables[var] # Hereafter, var is the selected variable netcdf object

  # Process each dimension of the variable. For each dimension, check the first
  # dimension spec for relevence. If any named spec matches the dim, apply it. If not,
  # apply the first non-specific spec. If there are no general spec's left, use the full
  # dimension.
  if debug: print 'All slice specifications=',sliceSpecs
  reGen = re.compile('[a-zA-Z_]')
  generalSpecs = [s for s in sliceSpecs for m in [reGen.match(s)] if not m]
  if debug: print 'All general specifications=',generalSpecs
  slices = []
  dims = var.dimensions; vars = rg.variables;
  for dim in dims:
    if debug: print 'Processing',dim,' slice specs=',sliceSpecs
    if len(sliceSpecs)>0: # Look for user provided spec
      reDim = re.compile(dim+'=', re.IGNORECASE)
      matchingSliceSpec = [s for s in sliceSpecs for m in [reDim.match(s)] if m]
      if debug: print '  Matching spec=',matchingSliceSpec
      # More than one match is an error
      if len(matchingSliceSpec)>1: error('Only one specification per dimension is allowed.\n'
          'The following specs appear to be for the same dimension: '+str(matchingSliceSpec))
      elif len(matchingSliceSpec)==0: # No spec is specific to this dimesion
        if len(generalSpecs): # Use up a general spec if any
          matchingSliceSpec = generalSpecs[0]
          generalSpecs.remove(matchingSliceSpec)
          if debug: print '  New general specs=',generalSpecs
          sliceSpecs.remove(matchingSliceSpec) # Pop this spec from the stack
        else: matchingSliceSpec = ':' # if no general specs left
      elif len(matchingSliceSpec)==1: # There is one matching specific spec
        matchingSliceSpec=matchingSliceSpec[0]
        sliceSpecs.remove(matchingSliceSpec) # Pop this spec from the stack
    else: matchingSliceSpec = ':' # Stack was empty
    if dim in vars: dVar = rg.variables[dim]
    else: dVar = None
    if debug: slices += [ iRange(rg.dimensions[dim], str(matchingSliceSpec), dVar) ]
    else:
      try: slices += [ iRange(rg.dimensions[dim], str(matchingSliceSpec), dVar) ]
      except: error('Unable to interpret dimension specificion "'+str(matchingSliceSpec)+'".')
  # Check that all user provided specs were used
  if len(sliceSpecs)>0: error('Some dimension specifications were not used.\n'
    +'Specifically, specifications '+''.join(str(s)+' ' for s in sliceSpecs)+' were unusable.\n'
    +'Variable "'+variableName+'" has dimensions: '+''.join(str(d)+' ' for d in dims)+'\n')
  if debug: print 'slices=',slices

  # Determine rank of data
  shape = []
  for s in slices:
    shape += [len(s)]
  rank = len(shape)-shape.count(1)
  if debug: print 'Requested data shape=',shape,'rank=',rank
  if rank>2: error('The requested data was multi-dimensional with rank '+str(rank)+'.\n'
           +'Only 0-, 1- and 2-dimensional data can be processed.')
  # Now read the coordinate data and variable data
  coordData=[]; axisLabel=[]; coordObj=[]
  for i,dim in enumerate(dims):
    if len(slices[i])>1:
      if dim in vars:
        coordData += [rg.variables[dim][slices[i]]]
        axisLabel += [constructLabel(rg.variables[dim],dim)]
        coordObj += [rg.variables[dim]]
      else:
        coordData += [np.array(slices[i])]
        axisLabel += [dim+' (index)']
        coordObj += [None]
  data = np.ma.masked_array(var[slices])
  if debug: print 'axisLabel=',axisLabel
  if debug: print 'coordObj=',coordObj

  # Optionally mask out a specific value
  if optCmdLineArgs.ignore:
    data = np.ma.masked_array(data, mask=[data==optCmdLineArgs.ignore])

  if optCmdLineArgs.list: print 'Data =\n',data
  if optCmdLineArgs.stats:
    dMin = np.min(data); dMax = np.max(data)
    print 'Mininum=',dMin,'Maximum=',dMax
    dMin = np.min(data[data!=0]); dMax = np.max(data[data!=0])
    print 'Mininum=',dMin,'Maximum=',dMax,'(ignoring zeros)'
  # Now plot
  if rank==0: print data[0]
  elif rank==1: # Line plot
    if len(data)==1: data=data[0]
    if isAttrEqualTo(coordObj[0],'cartesian_axis','z'): # Transpose 1d plot
      plt.plot(data,coordData[0])
      plt.xlabel(constructLabel(var))
      plt.ylabel(axisLabel[0])
      plt.ylim(coordData[0][0], coordData[0][-1])
      if coordData[0][0]>coordData[0][-1]: plt.gca().invert_yaxis()
      if isAttrEqualTo(coordObj[0],'positive','down'): plt.gca().invert_yaxis()
    else: # Normal 1d plot
      plt.plot(coordData[0],data)
      plt.xlabel(axisLabel[0])
      plt.xlim(coordData[0][0], coordData[0][-1])
      plt.ylabel(constructLabel(var))
  elif rank==2: # Pseudo color plot
    if debug: print 'coordData[1]=',coordData[1]
    if debug: print 'coordData[0]=',coordData[0]
    if isAttrEqualTo(coordObj[1],'cartesian_axis','z'): # Transpose 1d plot
      plt.pcolormesh(coordData[0],coordData[1],np.transpose(np.squeeze(data)))
      plt.xlabel(axisLabel[0])
      plt.xlim(coordData[0][0], coordData[0][-1])
      plt.ylabel(axisLabel[1])
      plt.ylim(coordData[1][0], coordData[1][-1])
      if isAttrEqualTo(coordObj[1],'cartesian_axis','z'): # Z on y axis ?
        if coordData[1][0]>coordData[1][-1]: plt.gca().invert_yaxis()
        if isAttrEqualTo(coordObj[1],'positive','down'): plt.gca().invert_yaxis()
    else:
      plt.pcolormesh(coordData[1],coordData[0],np.squeeze(data))
      plt.xlabel(axisLabel[1])
      plt.xlim(coordData[1][0], coordData[1][-1])
      plt.ylabel(axisLabel[0])
      plt.ylim(coordData[0][0], coordData[0][-1])
      if isAttrEqualTo(coordObj[0],'cartesian_axis','z'): # Z on y axis ?
        if coordData[0][0]>coordData[0][-1]: plt.gca().invert_yaxis()
        if isAttrEqualTo(coordObj[0],'positive','down'): plt.gca().invert_yaxis()
    plt.title(constructLabel(var))
    makeGuessAboutCmap()
    plt.colorbar()
  if optCmdLineArgs.output:
    plt.savefig(optCmdLineArgs.output,pad_inches=0.)
  else: plt.show()


# Interpret strSpec and return list of indices
def iRange(ncDim, strSpec, ncVar):
  equalParts = strSpec.split('='); dLen = len(ncDim)
  if debug: print '    = split',equalParts
  if len(equalParts)>2: error('Syntax error in "'+strSpec+'".')
  elif len(equalParts)==0: error('Impossible!')
  elif len(equalParts)==1: # Format : is: :ie is:ie
    colonParts = equalParts[0].split(':')
    if debug: print '    : split',colonParts
    if len(colonParts)>2: error('Too many :\'s in "'+strSpec+'".')
    # Interpret left of : to set indS
    if colonParts[0]=='': indS = 0
    else: indS = int(colonParts[0])-1 # Convert fortran/matlab index to python
    # Interpret right of : to set indE
    if len(colonParts)==1: indE = indS+1
    elif colonParts[1]=='': indE = dLen
    else: indE = int(colonParts[1]) # No conversion necessary because of python ranges
    if indS>=indE: error('In '+strSpec+' the index range is reversed.\n')
    if indS<0: error('In '+strSpec+' the start of the index range must >=1.\n')
    if indE>dLen: error('In '+strSpec+' the end of the index range is out of bounds.\n'
        +'The corresponding dimension, '+str(ncDim._name)+', has length '+str(dLen)+'.')
  elif len(equalParts)==2: # Format =: =xs: =:xe =xs:xe
    coordVals = ncVar[:]; coordSign = 1
    if coordVals[-1] < coordVals[0]: coordSign = -1 # In lieu of a sign() function
    colonParts = equalParts[1].split(':')
    if debug: print '    : split',colonParts
    if len(colonParts)==1 and colonParts[0]=='': error('Nothing on the r.h.s. of =\'s!')
    if len(colonParts)>2: error('Too many :\'s in "'+strSpec+'".')
    # Interpret left of : to set coordS
    if colonParts[0]=='': coordS = coordVals[0]
    else: coordS = float(colonParts[0])
    # Interpret right of : to set coordE
    if len(colonParts)==1: coordE = coordS
    elif colonParts[1]=='': coordE = coordVals[-1]
    else: coordE = float(colonParts[1])
    if debug: print '    coord range=',coordS,coordE
    if coordSign*(coordE-coordS) < 0: error('The coordinate range "'+strSpec+'" is inverted!')
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


# Returns a string combing CF attiributes "long_name" and "units"
def constructLabel(ncObj, default=''):
  if debug: print 'ncObj=',ncObj
  label = ''
  if 'long_name' in ncObj.ncattrs():
    label += str(ncObj.getncattr('long_name'))+' '
  if 'units' in ncObj.ncattrs():
    label += '('+str(ncObj.getncattr('units'))+')'
  if len(label)==0: label = default+' (index)'
  return label


# Returns True if ncObj has attribute "name" that matches "value"
def isAttrEqualTo(ncObj, name, value):
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


# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': parseCommandLine()
