# -*- coding: utf-8 -*-
import os

def RunElastix(ImgFixed,
	       ImgMoving,
	       DeformedImage,
	       ParameterFile,
	       ResultDirectory='./Results',
	       MaskFixed=None,
	       MaskMoving=None,
	       RunInitialtransform=None,
	       InitialtransformComputed=None,
	       VerboseMode='off'):
    if not ((type(ImgFixed) is str) and type(ImgMoving is str) and type(ParameterFile is str) and type(ResultDirectory is str)) and (type(DeformedImage) is str):
	raise ValueError, "Function members must be strings, you need at least to specify ImgFixed,ImgMoving DeformedImage and ParameterFile !"
    else :
	if not os.path.exists(ResultDirectory):
	    os.makedirs(ResultDirectory)
	Mask='';Init='';ResInd=0;Veb=''
	Param="-p %s" %(ParameterFile)
	Base="elastix -f %s -m %s -out %s " %(ImgFixed,ImgMoving,ResultDirectory)
	if (MaskFixed!=None) and (type(MaskFixed) is str) and MaskMoving==None:
	    Mask= "-fMask %s -mMask %s " %(MaskFixed,MaskFixed)
	elif (MaskFixed!=None) and (type(MaskFixed) is str) and (MaskMoving!=None) and (type(MaskMoving) is str):
	    Mask= "-fMask %s -mMask %s " %(MaskFixed,MaskMoving)
	elif (MaskFixed==None) and (MaskMoving!=None) and (type(MaskMoving) is str):
	    Mask= "-mMask %s " %(MaskMoving)	    
	if (RunInitialtransform!=None) and (type(RunInitialtransform) is str):
	    Param= "-p %s " + Param %(RunInitialtransform)
	    ResInd=1
	if os.name=='posix':
	    if (InitialtransformComputed!=None) and (type(InitialtransformComputed) is str):
		Init= "-t0 %s " %(InitialtransformComputed)
	    if VerboseMode=='off':
		Veb=" >> /dev/null"
	    elif VerboseMode=='on':
		Veb=''
	    elif VerboseMode=='log':
		Veb=" >> " + ResultDirectory + "/Biglog.txt"    
	ElastixCommandLine= Base + Mask + Init + Param + Veb
	print "Launching the following elastix Registration:"
	print "\t" + ElastixCommandLine
	os.system(ElastixCommandLine)
	if os.name=='posix':
	    fileName, fileExtension = os.path.splitext(DeformedImage)
	    if (fileExtension != 'vtk'):
		print "Export and import only mastered for vtk change your ParameterFile to match it !"
	    if ResInd == 0:
		os.system("mv " + ResultDirectory + "/result.%1.1d.vtk "%ResInd + ResultDirectory + "/" + fileName + ".vtk" )
		os.system("mv " + ResultDirectory + "/TransformParameters.%1.1d.txt "%ResInd + ResultDirectory + "/TransformParameters." + fileName + ".txt")
		os.system("rm "+ ResultDirectory + "/IterationInfo*")
		os.system("rm "+ ResultDirectory + "/*.log")		
	    elif ResInd == 1:
		os.system("mv " + ResultDirectory + "/result.%1.1d.vtk "%0 + ResultDirectory + "/" + fileName + ".ini.vtk" )
		os.system("mv " + ResultDirectory + "/TransformParameters.%1.1d.txt "%0 + ResultDirectory + "/TransformParameters." + fileName + "ini.txt")
		os.system("mv " + ResultDirectory + "/result.%1.1d.vtk "%ResInd + ResultDirectory + "/" + fileName + ".vtk" )
		os.system("mv " + ResultDirectory + "/TransformParameters.%1.1d.txt "%ResInd + ResultDirectory + "/TransformParameters." + fileName + ".txt")
	else:
	    raise ValueError, "Find the good Command to move the results on your system" 
	    # maybe under W$ : os.system("ren " + ResultDirectory + "\result.*.vtk " + ResultDirectory + "\" + DeformedImage)
    return 0
   

def ComputeFields(TransformParameters,
	          DisplacementField='',
	          StrainField='',
		  ResultDirectory='./Results'):
    Disp='';Strain=''
    if not ((type(TransformParameters) is str) and type(StrainField is str) and type(ResultDirectory is str)):
	raise ValueError, "Function members must be strings, you need at least to specify TransformParameters and a field to compute !"
    else :
	if not os.path.exists(ResultDirectory):
	    os.makedirs(ResultDirectory)
	Base="transformix -tp %s -out %s " %(TransformParameters,ResultDirectory)
	if StrainField=='' and DisplacementField=='':
	    print "You must at least ask for a field to be created";
	    return 0
	if DisplacementField!='':
	    Disp="-def all "
	if StrainField!='':
	    Strain="-jacmat all "	    
	TransformixCommandLine=Base + Disp + Strain + ">> /dev/null"   
	print "Launching the following transformix Operation:"
	print "\t" + TransformixCommandLine
	os.system(TransformixCommandLine)   
	if Strain=="-jacmat all ":
	    fileName, fileExtension = os.path.splitext(StrainField)
	    os.system("mv " + ResultDirectory + "/fullSpatialJacobian.vtk " + ResultDirectory + "/" + fileName + ".vtk" )
	if Disp=="-def all ":
	    fileName, fileExtension = os.path.splitext(DisplacementField)
	    os.system("mv " + ResultDirectory + "/deformationField.vtk " + ResultDirectory + "/" + fileName + ".vtk" )
	return 0
	
## This is to deform a given image by a specific parameter file a prior
def DeformImage(TransformParameters,
	          InputImage,
		  ResultDirectory='./Results'): 
    if not ((type(TransformParameters) is str) and type(InputImage is str) and type(ResultDirectory is str)):
	raise ValueError, "Function members must be strings, you need at least to specify TransformParameters and a field to compute !"
    else :
	if not os.path.exists(ResultDirectory):
	    os.makedirs(ResultDirectory)
	Base="transformix -tp %s -in %s -out %s " %(TransformParameters,InputImage,ResultDirectory)
	TransformixCommandLine=Base + ">> /dev/null"   
	print "Launching the following transformix Operation:"
	print "\t" + TransformixCommandLine
	os.system(TransformixCommandLine)   
        #fileName, fileExtension = os.path.splitext(InputImage)
	#os.system("mv " + ResultDirectory + "/result.vtk " + ResultDirectory + "/" + fileName.split('/')[-1].split('.')[0] + ".vtk" )
	return 0