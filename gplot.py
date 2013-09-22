#!/usr/bin/env python

from netCDF4 import Dataset
import argparse
import re

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
  print 'sl',sliceSpecs
  reGen = re.compile('[a-zA-Z_]')
  generalSpecs = [s for s in sliceSpecs for m in [reGen.match(s)] if not m]
  print 'generalSpecs=',generalSpecs
  slices = []
  dims = var.dimensions
  for dim in dims:
    print 'Processing',dim,' sliceSpecs=',sliceSpecs
    if len(sliceSpecs)>0:
      reDim = re.compile(dim+'=', re.IGNORECASE)
      matchingSliceSpec = [s for s in sliceSpecs for m in [reDim.match(s)] if m]
      print 'ms=',matchingSliceSpec
      if len(matchingSliceSpec)>1: error('Only one specification per dimension is allowed.\n'
          'The following specs appear to be for the same dimension: '+str(matchingSliceSpec))
      elif len(matchingSliceSpec)==0: # No spec is specific to this dimesion
        if len(generalSpecs):
          matchingSliceSpec = generalSpecs[0]
          generalSpecs.remove(matchingSliceSpec)
          print 'generalSpecs=',generalSpecs
          sliceSpecs.remove(matchingSliceSpec) # Pop this spec from the stack
      elif len(matchingSliceSpec)==1:
        matchingSliceSpec=matchingSliceSpec[0]
        sliceSpecs.remove(matchingSliceSpec) # Pop this spec from the stack
    else: matchingSliceSpec = ':' # Stack was empty
    slices += [ iRange(rg.dimensions[dim], str(matchingSliceSpec)) ]
  if len(sliceSpecs)>0: error('Some dimension specifications were not used.\n'
    +'Specifically, specifications '+''.join(str(s)+' ' for s in sliceSpecs)+' were unusable.\n'
    +'Variable "'+variableName+'" has dimensions: '+''.join(str(d)+' ' for d in dims)+'\n')
  print 'slices=',slices

def iRange(ncDim, strSpec): # Interpret strSpec and return indices
  parts = strSpec.split('='); dLen = len(ncDim)
  print 'parts=',parts
  if len(parts)>2: error('Syntax error in "'+strSpec+'".')
  elif len(parts)==0: error('Impossible!')
  elif len(parts)==1: # Format : is: :ie is:ie
    iParts = parts[0].split(':')
    print 'iParts=',iParts
    if iParts[0]=='': indS = 0
    else: indS = int(iParts[0])-1 # Convert fortran/matlab index to python
    if len(iParts)==1: indE = indS+1
    elif iParts[1]=='': indE = dLen
    else: indE = int(iParts[1]) # No conversion necessary because of python ranges
    if indS>=indE: error('In '+strSpec+' the index range is reversed.\n')
    if indS<0: error('In '+strSpec+' the start of the index range must >=1.\n')
    if indE>dLen: error('In '+strSpec+' the end of the index range is out of bounds.\n'
        +'The corresponding dimension, '+str(ncDim._name)+', has length '+str(dLen)+'.')
  elif len(parts)==2:
    error('Not yet implemented!')
  if indE-indS==1: return indS
  else: return [indS,indE]

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

def error(msg,code=9):
  print 'Error: ' + msg + ' Aborting!'
  exit(code)

def main():

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
  args = parser.parse_args()

  doTheThing(args.filename, args.variable, args.pos)

# Invoke main()
if __name__ == '__main__': main()
