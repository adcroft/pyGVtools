#!/usr/bin/env python

from netCDF4 import Dataset
import argparse

def doTheThing(fileName, variableName, sliceSpecs):
  # Open netcdf file
  try: rg=Dataset( fileName, 'r' );
  except: error('There was a problem opening "'+fileName+'".')

  if len(variableName)==0:
    dumpFile(rg)

  print 'sl',sliceSpecs

def dumpFile(rg):
  dims = rg.dimensions
  vars = rg.variables
  print 'Dimensions:'
  for dim in dims:
    oString = dim+' ['+str(len( dims[dim] ))+']'
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
    oString = var+' [ '; dString = ''
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
    'Yada yada yada')
  parser.add_argument('filename', type=str,
                      help='netcdf file to use')
  parser.add_argument('variable', type=str,
                      nargs='?', default='',
                      help='Name of variable to plot')
  parser.add_argument('pos', type=str,
                      nargs='*', default='',
                      help='Indices or location specification')
  args = parser.parse_args()

  doTheThing(args.filename, args.variable, args.pos)

# Invoke main()
if __name__ == '__main__': main()
