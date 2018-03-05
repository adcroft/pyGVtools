"""
Functions for calculations on a mesh

Conventions:
  * Use [j,i] indexing conventions (consistent with netcdf) within arrays.
  O In coordinate tuples use (x,y) ordering of coordinates.

A mesh is defined as a a set of (ni+1,nj+1) nodes connected in a rectangular grid forming (ni,nj) logically
rectangular cells. 

A "box" refers to a logically rectangular outline of a mesh or subset of a mesh.
"""

import numpy as np


def findIndicesOfCell(xMesh, yMesh, xPoint, yPoint):
  """
  Returns the indices of a lower-left corner of a mesh-cell that contains the
  specified point.
  """

  if xMesh.shape!=yMesh.shape: raise Exception('The x,y coordinates of the mesh must be the same shape')
  if xMesh.ndim!=2: raise Exception('The x,y coordinates of the mesh must be 2-dimensional')

  nj, ni = xMesh.shape
  #print 'ni=',ni,'nj=',nj
  if ni<=1: raise Exception('The mesh must have at least two nodes in the i-direction')
  if nj<=1: raise Exception('The mesh must have at least two nodes in the j-direction')

  iLeft = 0; iRight = ni-1; jBottom = 0; jTop = nj-1 # Starting serach box
  iMesh, jMesh = (None, None)

  stack=set()
  stack.add( (iLeft, jBottom, iRight, jTop) )
  while stack:
    iLeft, jBottom, iRight, jTop = stack.pop()
    #print 'left, right=',iLeft,iRight,'bottom, top=',jBottom,jTop,'?'
    if iRight-iLeft==1 and jTop-jBottom==1: # This detects that we have refined down to a single cell
      if pointIsInBoundingBox(xPoint, yPoint, xMesh, yMesh, iLeft, jBottom, iRight, jTop):
        if pointIsInCell(xPoint, yPoint, xMesh, yMesh, iLeft, jBottom):
          iMesh, jMesh = (iLeft, jBottom)
          #print 'Hit: i,j=',iMesh,jMesh
      continue
    if pointIsInBoundingBox(xPoint, yPoint, xMesh, yMesh, iLeft, jBottom, iRight, jTop):
      iMiddle = ( iLeft + iRight )/2; jMiddle = ( jBottom + jTop )/2 # Bisect into quadrants
      #print 'left, middle, right=',iLeft,iMiddle,iRight,'bottom, middle, top=',jBottom,jMiddle,jTop,'In'
      if iMiddle>iLeft:
        if jMiddle>jBottom: stack.add( (iLeft, jBottom, iMiddle, jMiddle) )
        stack.add( (iLeft, jMiddle, iMiddle, jTop) )
      if jMiddle>jBottom: stack.add( (iMiddle, jBottom, iRight, jMiddle) )
      stack.add( (iMiddle, jMiddle, iRight, jTop) )

  return iMesh, jMesh


def pointIsInBoundingBox(xPoint, yPoint, xMesh, yMesh, iLeft=0, jBottom=0, iRight=None, jTop=None):
  """
  Returns True if the point (x,y) is within the quadrant bounding box.
  """

  nj, ni = xMesh.shape
  if iRight==None: iRight = ni-1
  if jTop==None: jTop = nj-1

  xBox, yBox = pointsAroundBox(xMesh, yMesh, iLeft, jBottom, iRight, jTop)
  if xPoint<min(xBox) or xPoint>max(xBox) or yPoint<min(yBox) or yPoint>max(yBox): return False
  else: return True


def pointIsInCell(xPoint, yPoint, xMesh, yMesh, iLeft, jBottom):
  """
  Returns True if the point (x,y) is within the cell with the given indices.
  """

  xBox, yBox = pointsAroundBox(xMesh, yMesh, iLeft, jBottom, iLeft+1, jBottom+1)
  return pointIsInConvexPolygon(xBox, yBox, xPoint, yPoint)


def boundingBox(xMesh, yMesh, iLeft=0, jBottom=0, iRight=None, jTop=None):
  """
  Returns the bounding box (xMin,yMin,xMax,yMax) of the sub-mesh.

  The bounding box will generally be larger than the coordinate range covered by the 
  """

  nj, ni = xMesh.shape
  if iRight==None: iRight = ni-1
  if jTop==None: jTop = nj-1

  xPoints, yPoints = pointsAroundBox(xMesh, yMesh, iLeft, jBottom, iRight, jTop)
  return min(xPoints), min(yPoints), max(xPoints), max(yPoints)


def pointsAroundBox(xMesh, yMesh, iLeft=0, jBottom=0, iRight=None, jTop=None):
  """
  Returns points in a box defined by rectangular range of indices that extract a polygon from a mesh,
  starting at lower-left logical indices and proceeding in a counter-clockwise order.

  If the mesh is a non-rotated cartesian coordinate system then the result will be a rectangle, but for
  a general curvilinear coordinate system the result will be a closed shape of four curves joining four
  corners.
  """

  nj, ni = xMesh.shape
  if iRight==None: iRight = ni-1
  if jTop==None: jTop = nj-1
  if iRight<=iLeft: raise Exception('The i-index range must span at least one cell (two nodes)')
  if jTop<=jBottom: raise Exception('The j-index range must span at least one cell (two nodes)')

  xPoints=\
      xMesh[jBottom, list(range(iLeft, iRight))].tolist()+\
      xMesh[list(range(jBottom, jTop)), iRight].tolist()+\
      xMesh[jTop, list(range(iRight, iLeft, -1))].tolist()+\
      xMesh[list(range(jTop, jBottom, -1)), iLeft].tolist()
  yPoints=\
      yMesh[jBottom, list(range(iLeft, iRight))].tolist()+\
      yMesh[list(range(jBottom, jTop)), iRight].tolist()+\
      yMesh[jTop, list(range(iRight, iLeft, -1))].tolist()+\
      yMesh[list(range(jTop, jBottom, -1)), iLeft].tolist()
  return xPoints, yPoints


def pointIsInConvexPolygon(xPolygon, yPolygon, xPoint, yPoint):
  """
  Returns True if the point is within the convex polygon.
  """
  ccw = 0; cw = 0
  P = (xPoint, yPoint)
  A = (xPolygon[-1], yPolygon[-1])
  for bx, by in zip(xPolygon, yPolygon):
    B = (bx, by)
    if crossProduct(P, A, B) >= 0: ccw = ccw + 1
    else: cw = cw + 1
    A = B
  #print 'pointIsInConvexPolygon: xPoint=',xPoint,'xPolygon=',xPolygon
  #print 'pointIsInConvexPolygon: yPoint=',yPoint,'yPolygon=',yPolygon
  #print 'pointIsInConvexPolygon: ccw=',ccw,'cw=',cw
  return ccw==0 or cw==0


def crossProduct(A, B, C):
  """
  Returns cross product of AB and AC.

  A, B and C are tuples of (x,y) coordinates.

  If crossProduct(A, B, C)==0, then A, B and C fall on a line.
  If crossProduct(A, B, C)>0, then A, B, C are in a counter-clockwise order.
  """
  return (C[1]-A[1])*(B[0]-A[0]) - (B[1]-A[1])*(C[0]-A[0])


def segmentsIntersect(A, B, C, D):
  """
  Returns true if segment A-B intersects segment C-D. A, B, C and D are tuples of (x,y) coordinates.
  """

  ccw = crossProduct

  #print 'segmentsIntersect: A=',A,'B=',B,'C=',C,'D=',D
  ABC = crossProduct(A, B, C); ABD = crossProduct(A, B, D)
  #print 'segmentsIntersect: cp(A,B,C)=',ABC,'cp(A,B,D)=',ABD
  if ABC*ABD > 0: return False
  CDA = crossProduct(C, D, A); CDB = crossProduct(C, D, B)
  #print 'segmentsIntersect: cp(C,D,A)=',CDA,'cp(C,D,B)=',CDB
  if CDA*CDB > 0: return False
  return True

# Tests
if __name__ == '__main__':

  import meshtools

  # Test data
  X, Y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 6))
  print('X=',X)
  print('Y=',Y)

  print('Bounding box=',boundingBox(X, Y))
  print('Bounding box(0,2,3,4)=',boundingBox(X, Y, 0, 2, 3, 4))

  def test_findIndicesOfCell(xMesh, yMesh, xPoint, yPoint, result):
    i, j = findIndicesOfCell(xMesh, yMesh, xPoint, yPoint)
    if (j,i) == result: test = 'Correct'
    else: test = 'Wrong'
    print((xPoint,yPoint),'is in cell',(i,j),test)
  test_findIndicesOfCell(X, Y, X[1,2]+.1, Y[1,2]+.1, (1, 2))
  test_findIndicesOfCell(X, Y, X[1,2], Y[1,2]+.1, (1, 2))
  test_findIndicesOfCell(X, Y, X[1,2], Y[1,2]+.1, (1, 2))
  test_findIndicesOfCell(X, Y, X[1,2], Y[1,2], (1,2))
  test_findIndicesOfCell(X, Y, X[-1,-1], Y[-1,-1], (4,3))
  test_findIndicesOfCell(X, Y, X[0,0], Y[0,0], (0,0))
  test_findIndicesOfCell(X, Y, X[0,0]-.1, Y[0,0], (None,None))

  def intersectTest(A, B, C, D, result):
    intersects = segmentsIntersect(A, B, C, D)
    if intersects == result: test = 'Correct'
    else: test = 'Wrong'
    print('(%d,%d)-(%d,%d) intersects (%d,%d)-(%d,%d)?'%(
        A[0],A[1],B[0],B[1],C[0],C[1],D[0],D[1] ), intersects, test)
  intersectTest( (0,0), (1,0), (0,1), (1,1), False )
  intersectTest( (0,0), (0,1), (1,1), (0.2,0), False )
  intersectTest( (0,0), (1,1), (1,0), (0,1), True )
  intersectTest( (0,0), (1,1), (-1,0.1), (1,0), True )
  intersectTest( (0,0), (1,1), (-1,0), (1,0), True )
  intersectTest( (0,0), (1,1), (0,0), (0,1), True )
  intersectTest( (0,0), (1,1), (0,1), (1,1), True )
