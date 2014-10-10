# -*- coding: utf-8 -*-
import vtk,numpy

def readPicture(filename):
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    grid = reader.GetOutput()
    Dimensions=grid.GetExtent()
    NumPixX=Dimensions[1]+1
    NumPixY=Dimensions[3]+1
    centre = grid.GetCenter()
    bounds = grid.GetBounds()
    reader.SetScalarsName(reader.GetScalarsNameInFile(0));
    reader.Update(); 
    structuredPoints = reader.GetOutput();
    pd=structuredPoints.GetPointData();
    scalars=pd.GetScalars(reader.GetScalarsNameInFile(0));
    numPoints = scalars.GetSize()
    Img=numpy.zeros([numPoints,1])
    for i in range(0,(numPoints-1)):
      Img[i]=scalars.GetValue(i);
    return Img.reshape(NumPixY,NumPixX);

def readDisplacement(filename):
    reader = vtk.vtkStructuredPointsReader();reader.SetFileName(filename);
    reader.Update();
    grid = reader.GetOutput();
    Dimensions=grid.GetExtent();
    NumPixX=Dimensions[1]+1;
    NumPixY=Dimensions[3]+1;
    centre = grid.GetCenter();
    bounds = grid.GetBounds();
    reader.SetScalarsName(reader.GetScalarsNameInFile(0));reader.Update();structuredPoints = reader.GetOutput();
    pd=structuredPoints.GetPointData();
    scalars=pd.GetScalars(reader.GetScalarsNameInFile(0));
    numPoints = scalars.GetSize();
    U=numpy.zeros([numPoints/2,1]);
    V=numpy.zeros([numPoints/2,1]);
    for i in range(0,(numPoints-1)/2):
      U[i]=scalars.GetValue(2*i);
      V[i]=scalars.GetValue(2*i+1);
    return {'U':U.reshape(NumPixY,NumPixX), 'V':V.reshape(NumPixY,NumPixX) }
    
def readStrain(filename):
    reader = vtk.vtkStructuredPointsReader();reader.SetFileName(filename);
    reader.Update();
    grid = reader.GetOutput();
    Dimensions=grid.GetExtent();
    NumPixX=Dimensions[1]+1;
    NumPixY=Dimensions[3]+1;
    centre = grid.GetCenter();
    bounds = grid.GetBounds();
    reader.SetScalarsName(reader.GetScalarsNameInFile(0));reader.Update();structuredPoints = reader.GetOutput();
    pd=structuredPoints.GetPointData();
    scalars=pd.GetScalars(reader.GetScalarsNameInFile(0));
    numPoints = scalars.GetSize();
    grad11=numpy.zeros([numPoints/4,1]);
    grad12=numpy.zeros([numPoints/4,1]);
    grad21=numpy.zeros([numPoints/4,1]);
    grad22=numpy.zeros([numPoints/4,1]);    
    for i in range(0,(numPoints-1)/4):
      grad11[i]=scalars.GetValue(4*i);
      grad12[i]=scalars.GetValue(4*i+1);
      grad21[i]=scalars.GetValue(4*i+2);
      grad22[i]=scalars.GetValue(4*i+3);
    return {'grad11':grad11.reshape(NumPixY,NumPixX), 'grad12':grad12.reshape(NumPixY,NumPixX),'grad21':grad21.reshape(NumPixY,NumPixX),'grad22':grad22.reshape(NumPixY,NumPixX)}
    
def writePicture(filename,image):
  #image reshape
    imageLength=image.shape[0]
    imageWidth=image.shape[1]
    image=image.reshape(imageLength*imageWidth,1)
  #create a data array to stock these values 
    data_array=vtk.vtkFloatArray()
    data_array.SetNumberOfValues(len(image))
    for i in range(0,len(image)):
      data_array.SetValue(i,image[i][0])

  #create an ImageData called vtkStructuredPoints which stocks the pixel/voxel value of the image 
    strPts=vtk.vtkStructuredPoints()
    strPts.SetDimensions(imageWidth,imageLength,1)
    strPts.SetOrigin(0.,0.,0.)
    strPts.SetExtent(0,imageWidth-1,0,imageLength-1,0,0)
    strPts.SetNumberOfScalarComponents(1)
    strPts.SetScalarTypeToFloat()
    strPts.AllocateScalars()

    strPts.GetPointData().SetScalars(data_array)
    strPts.Update()
  #print the image to an output file 
    writer=vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInput(strPts)
    writer.Update()
    writer.Write()
