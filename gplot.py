#!/usr/bin/env python

def error(msg,code=9):
  print 'Error: ' + msg
  exit(code)

import re
try: import argparse
except: error('This version of python is not new enough. python 2.7 or newer is required.')
try: from netCDF4 import Dataset
except: error('Unable to import netCDF4 module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_netcdf4')
#try: import numpy as np
#except: error('Unable to import numpy module. Check your PYTHONPATH.\n'
#          +'Perhaps try:\n   module load python_numpy')
try: import matplotlib.pyplot as plt
except: error('Unable to import matplotlib.pyplot module. Check your PYTHONPATH.\n'
          +'Perhaps try:\n   module load python_matplotlib')

debug = False # Global debugging

def doTheThing(fileName, variableName, sliceSpecs):
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
  coordData=[]; axisLabel=[]
  for i,dim in enumerate(dims):
    if len(slices[i])>1:
      if dim in vars:
        coordData += [rg.variables[dim][slices[i]]]
        axisLabel += [constructLabel(rg.variables[dim],dim)]
      else:
        coordData += [slices[i]]
        axisLabel += [dim+' (index)']
  data = var[slices]
  if debug: print 'axisLabel=',axisLabel

  # Now plot
  if rank==0: print data[0]
  elif rank==1: # Line plot
    if len(data)==1: data=data[0]
    plt.plot(coordData[0],data)
    plt.xlabel(axisLabel[0])
    plt.xlim(coordData[0][0], coordData[0][-1])
    plt.ylabel(constructLabel(var))
    plt.show()
  elif rank==2: # Pseudo color plot
    plt.pcolormesh(coordData[1],coordData[0],data)
    plt.xlabel(axisLabel[1])
    plt.xlim(coordData[1][0], coordData[1][-1])
    plt.ylabel(axisLabel[0])
    plt.ylim(coordData[0][0], coordData[0][-1])
    plt.title(constructLabel(var))
    plt.show()

def iRange(ncDim, strSpec, ncVar): # Interpret strSpec and return list of indices
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

def coord2index( coordList, coordVal, roundUp=False): # Returns index of nearest coordList
  ind = min(range(len(coordList)), key=lambda i:abs(coordList[i]-coordVal))
  if roundUp and (ind+1<len(coordList)):
    # Correct for roundin 0.5 down to 0.
    if abs(coordList[ind+1]-coordVal)<=abs(coordList[ind]-coordVal): ind=ind+1
  if debug: print '      coord(',ind,')=',coordList[ind],' matches coord=',coordVal
  return ind

def constructLabel(ncObj,default=''):
  if debug: print 'ncObj=',ncObj
  label = ''
  if 'long_name' in ncObj.ncattrs():
    label += str(ncObj.getncattr('long_name'))+' '
  if 'units' in ncObj.ncattrs():
    label += '('+str(ncObj.getncattr('units'))+')'
  if len(label)==0: label = default+' (index)'
  return label

def summarizeFile(rg):
  vars = rg.variables
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

def main():
  global debug # Declared global in order to set it

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
  parser.add_argument('-d','--debug', action='store_true',
                      help='Turn on debugging information.')
  args = parser.parse_args()

  if args.debug: debug = True

  doTheThing(args.filename, args.variable, args.pos)

# Invoke main()
if __name__ == '__main__': main()
