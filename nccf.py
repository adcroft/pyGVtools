# Try to import required packages/modules
import os
import netCDF4 as nc4
import warnings
import numpy

warnings.simplefilter('error', UserWarning)

debug = False # Global debugging

def openNetCDFfileForReading(fileName):
  """
  Return Dataset type for file.
  """

  try: rg = nc4.Dataset(fileName, 'r')
  except:
    if os.path.isfile(fileName): raise Exception('There was a problem opening "'+fileName+'".')
    raise Exception('Could not find file "'+fileName+'".')
  return rg


def dump(fileName):
  """
  A succinct dump of a netCDF4 file.
  """
  rg = openNetCDFfileForReading(fileName)
  dims = rg.dimensions; vars = rg.variables
  print 'Summary of %s:'%fileName
  print 'Dimensions: -------------------------------------'
  for dim in dims:
    oString = ' '+dim+' ['+str(len( dims[dim] ))+']'
    if dim in vars:
      n = len( dims[dim] ); obj = rg.variables[dim]
      if n>5: oString += ' = '+str(obj[0])+'...'+str(obj[n-1])
      else: oString += ' = '+str(obj[:])
      if 'long_name' in obj.ncattrs(): oString += ' "'+obj.long_name+'"'
      if 'units' in obj.ncattrs(): oString += ' ('+obj.units+')'
    print oString
  print 'Variables: --------------------------------------'
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
  rg.close()


def readVariableFromNetCDFfile(fileName, variableName, *args):
  """
  Reads a variable from a netCDF file.

  Optional arguments are ranges for each dimension.
  Missing ranges fetch the entire dimensions.

  Examples:
  >>> T,_,_ = readVariableFromNetCDFfile('test.nc','xyz')
  >>> T,dims,atts = readVariableFromNetCDFfile('test.nc','xyz',rang(1,4),3)

  Returns: data, dimensions, attributes.
    data       will be a numpy masked array
    dimensions will be a list of numpy 1D vectors
    attributes will be a dictionary
  """

  rg = openNetCDFfileForReading(fileName)
  if not variableName:
    print 'No variable name specified! Specify a varible from the following summary of "'\
          +fileName+'":\n'
    dump(fileName)
    exit(0)

  # Check that the variable is in the file (allowing for case mismatch)
  for v in rg.variables:
    if variableName.lower() == v.lower(): variableName=v ; break
  if not variableName in rg.variables:
    raise MyError('Did not find "'+variableName+'" in file "'+fileName+'".')

  vh = rg.variables[variableName] # Handle for variable

  dimensions = []
  for n, d in enumerate(vh.dimensions):
    if n < len(args):
      if d in rg.variables: dimensions.append( rg.variables[d][args[n]] )
      else: dimensions.append( args[n] )
    else:
      if d in rg.variables: dimensions.append( rg.variables[d][:] )
      else: dimensions.append( range( len(rg.dimensions[d] ) ) )

  attributes = {}
  for a in vh.ncattrs():
    attributes[a] = vh.getncattr(a)

  data = vh[args]
  rg.close()
  return data, dimensions, attributes


def writeVariableToNetCDFfile(fileName, variableName=None, variable=None, dimensions=None, attributes=None, dataType='f8'):
  """
  Writes a variable to a netCDF file.

  Arguments:
  fileName     name of the file
  variableName name of variable to appear in the file
  variable     a numpy masked array
  dimensions   a dictionary of dimension names and 1D dimension data 
               or a list of names, or a list of 1D dimension data
  """

  print 'variableName=',variableName
  print 'variable=',type(variable)
  print 'dimensions=',type(dimensions)
  print 'attributes=',type(attributes)

  if os.path.isfile(fileName): rg = nc4.Dataset(fileName,'a')
  else: rg = nc4.Dataset(fileName,'w')

  def createDimIfMissing(rg, name, size):
    if name in rg.dimensions:
      if not len(rg.dimensions[name])==size:
        raise Exception('Dimension "%s" has size %i in file and differs from provided size %i'
                 %(name, len(rg.dimensions[name]), size))
    else:
      rg.createDimension(name, size)
    return name

  def createDimDataIfMissing(rg, name, data, dataType):
    createDimIfMissing(rg, name, len(data))
    if name in rg.variables:
      if any(rg.variables[name][:]!=data):
        raise Exception('Dimension data "%s" does not match provided data'%name)
    else:
      rg.createVariable(name, dataType, name)
      rg.variables[name][:] = data
    return name

  def matchingDimsByData(rg, data):
    matchingDims = []
    for d in rg.dimensions:
      if d in rg.variables:
        if rg.variables[d].shape==data.shape:
          if not any(rg.variables[d][:]!=data): matchingDims.append(d)
    if len(matchingDims)>1: raise Exception('Too many dimension-data match the provided data')
    elif len(matchingDims)==0: return None
    return matchingDims[0]

  variableDimensions = None
  if dimensions==None:
    if not variable==None:
      # Create or match some simple dimensions with made up names
      variableDimensions = []
      for i in range(len(variable.shape)):
        matchingDims = [d for d in rg.dimensions if len(rg.dimensions[d])==variable.shape[i] and not d in variableDimensions]
        print 'matchingDims=',matchingDims
        if len(matchingDims)>1: raise Exception(
          'Too many matching-length dimensions to choose from. Please provide specific dimensions')
        elif len(matchingDims)==1:
          variableDimensions.append( createDimIfMissing(rg, matchingDims[0], variable.shape[i]) )
        else:
          variableDimensions.append( createDimIfMissing(rg, 'dim%i'%i, variable.shape[i]) )
  elif isinstance(dimensions, list):
    if not variable==None:
      # Create or match dimensions based on names or vectors 
      variableDimensions = []
      for i,dim in enumerate(dimensions):
        if isinstance(dim, basestring):
          variableDimensions.append( createDimIfMissing(rg, dim, variable.shape[i]) )
        elif isinstance(dim, numpy.ndarray):
          dName = matchingDimsByData(rg, dim)
          if dName==None: dName = 'dim%i'%i
          variableDimensions.append( createDimDataIfMissing(rg, dName, dim, dataType) )
        else:
          print '******* Not sure what to do with',dim
          print 'type=',type(dim)
  elif isinstance(dimensions, dict):
    # Create dimensions from dictionary provided
    variableDimensions = []
    for n in dimensions:
      variableDimensions.append( createDimDataIfMissing(rg, n, dimensions[n], dataType) )
  else: raise Exception('Not sure what to do with the dimensions argument!')
  print 'variableDimensions=',variableDimensions

  if not variableName==None:
    if variableName in rg.variables: vh = rg.variables[variableName]
    elif not variableDimensions==None: vh = rg.createVariable(variableName, dataType, variableDimensions, fill_value=None)
  else: vh = None

  if not attributes==None and not vh==None:
    for a in attributes:
      if not a in ['_FillValue']:
        vh.setncattr(a,attributes[a])

  if not variable==None and not vh==None: vh[:] = variable
  rg.close


def testNCCF():
  """
  A simple test of writing a netcdf file
  """

  testFile = 'baseline.1900-1909.salt_temp_e.nc'
  dump(testFile)
  print '======= dump finished' ; print

  T, d, a = readVariableFromNetCDFfile(testFile,'Temp',0,4,range(580,593),range(40,50))
  print 'T=',T
  print 'd=',d
  print 'a=',a
  print '======= read T finished' ; print

  print 'Testing creation with dictionary dimensions'
  writeVariableToNetCDFfile('q.nc', 'w1', -T, dimensions={'y':d[-2],'x':d[-1]})
  dump('q.nc')
  print 'Testing creation with just data dimensions'
  writeVariableToNetCDFfile('q.nc', 'w1', T, dimensions=d)
  dump('q.nc')
  print 'Testing creation with just named dimensions'
  writeVariableToNetCDFfile('q.nc', 'w1', -T, dimensions=['y','x'])
  dump('q.nc')
  print 'Testing creation with no dimensions'
  writeVariableToNetCDFfile('q.nc', 'w1', T)
  dump('q.nc')
  print 'Testing creation with just attributes'
  writeVariableToNetCDFfile('q.nc', 'w1', attributes=a)
  dump('q.nc')
  print '======= write T finished' ; print


def enableDebugging(newValue=True):
  """
  Sets the global parameter "debug" to control debugging information. This function is needed for
  controlling debugging of routine imported from gplot.py in other scripts.
  """
  global debug
  debug = newValue


# Invoke parseCommandLine(), the top-level prodedure
if __name__ == '__main__': testNCCF()
