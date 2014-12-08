# -*- coding: utf-8 -*-
import numpy as np
import scipy
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import math as m
import GetPTW as PTW
import copy
import numpy.ma as ma
import skimage.io as io
io.use_plugin('freeimage')


class Calibration:
  def __init__(self,sigma,BB_Emissivity,directory,result_directory,video_directory,Tmin,Tmax,Tstep,polynomial_degree,bad_pix_critere,area_size,NUC,Flux,extension,camera): 
    self.sigma = sigma # sigma is use in the boltzman equation
    self.BB_Emissivity = BB_Emissivity
    self.directory= directory # directory of th BB images for calibration
    self.result_directory=result_directory #directory for storage of the coefficients
    self.video_directory=video_directory
    self.Tmin=Tmin 
    self.Tmax=Tmax
    self.Tstep=Tstep
    self.polynomial_degree=polynomial_degree
    self.bad_pix_critere=bad_pix_critere
    self.area_size=area_size# size, in pixels, of the side of the square area used for the mean(DL) in the NUC 
    self.NUC=NUC # set True if you want to use the NUC (DL=>DL) function
    self.Flux=Flux # set True if you want to convert DL to flux
    self.extension=extension
    self.camera=camera


# # # # Definition of the functions

  def import_imageBB(self):
"""
  This function imports ths BB-image and shape them in vectors. 
  Returns:  
    imageBB[i]: the original array of DL, where i is image number
"""
    imageBB={}
    imageBB_={}
    if self.extension=="ptm":
      for i in range(0,(self.Tmax-self.Tmin)/self.Tstep):
	imageBB_[i]=PTW.GetPTW(self.directory+str(i*self.Tstep+self.Tmin)+".ptm")
	imageBB_[i].get_frame(0)
	imageBB[i]=1.*imageBB_[i].frame_data 
    if self.extension=="ptw":
      for i in range(0,(self.Tmax-self.Tmin)/self.Tstep):
	imageBB_[i]=PTW.GetPTW(self.directory+str(i*self.Tstep+self.Tmin)+".ptw")
	nbr_of_frames=imageBB_[i].number_of_frames
	imageBB[i]=0
	for j in range(nbr_of_frames):  # This loop calculate the mean of the images. use it if you didn't save in .ptm
	  imageBB_[i].get_frame(j)
	  imageBB[i]+=(1.*imageBB_[i].frame_data)/(1.*nbr_of_frames)
    return imageBB


  def reshape(self,DL):
"""
 This shape images into vectors. 
  Returns: 
      matrice: an array in wich each line is a vectorised BB-image, one line by T°C
      DL_mean[i]: the mean DL in the central area, for the NUC function. i depend on the T°C
      shape : the original shape of ths images
"""
    imageBB=DL
    imageBB2={}
    matrice=[]
    mat2vect={}
    DL_mean=[]
    if self.camera=="titanium":
      for i in range(len(imageBB)):
	shape=np.shape(imageBB[0])
	b=shape[0]
	c=shape[1]
	DL_mean.append(np.mean(imageBB[i][m.floor((b-self.area_size)/2):m.floor((b+self.area_size)/2),m.floor((c-self.area_size)/2):m.floor((c+self.area_size)/2)]))
	mat2vect[i]=np.reshape(imageBB[i],(1,b*c))
      matrice = np.concatenate([mat2vect[i] for i in range(len(imageBB))],axis=0)
    if self.camera=="jade":
      sortie={}
      DL_mean=[]
      matrice=[]
      for h in range(len(imageBB)):
	im_cmox=np.zeros((120,160,4))
	for k in range(0,4):
	  for i in range(0,120):
	    for j in range(0,80):
	      im_cmox[i,(2*j-1)+1:2*j+2,k] = imageBB[h][2*i:2*i+2 ,4*(j-1)+1+k-1+4].T
	imageBB2[h] = np.fliplr(np.concatenate( (np.concatenate((im_cmox[:,:,0],im_cmox[:,:,1]),axis=1),np.flipud(np.concatenate((im_cmox[:,:,2],im_cmox[:,:,3]),axis=1))), axis=0));
	shape=np.shape(imageBB[0])
	b=shape[0]
	c=shape[1]
	DL_mean.append(np.mean(imageBB2[h][m.floor((b-self.area_size)/2):m.floor((b+self.area_size)/2),m.floor((c-self.area_size)/2):m.floor((c+self.area_size)/2)]))
	mat2vect[h]=np.reshape(imageBB2[h],(1,b*c))
      matrice = np.concatenate([mat2vect[h] for h in range(len(imageBB))],axis=0)
    elif self.camera!="jade" and self.camera!="titanium":
      print "camera not recognized, please specify titanium or jade"
    return matrice, DL_mean, shape


  def NUC_DL(self,DL,DL_mean):
"""
This function calculate the NUC coefficient for each pixels, based on the central-zone mean DL
  Returns:
      The matrix of vectorised BB-images, with the NUC applied
  Save:
      Save the NUC coefficients in the result directory, under coeffs_NUC.pyc
"""
    DLC_NUC=copy.copy(DL)
    a,d = np.shape(DL)
    interp_degree=3
    K=np.zeros((d,interp_degree+1))

    for i in range (d):
      if 0 not in DL[:,i]:
	K[i]=np.polyfit(DL[:,i],DL_mean,interp_degree)
	DLC_NUC[:,i]=np.polyval(K[i],DL[:,i])
    np.save(self.result_directory+'coeffs_NUC',K)
    return DLC_NUC


  def DL2Flux(self,DL):
"""
This function calculate the calibration coefficients for each pixels
  Returns:
      The matrix of vectorised BB-images, with the (NUC and) calibration applied
  Save:
      Save the calibration coefficients in the result directory, under coeffs_DL2Flux.pyc
"""
    DLC_Calibrated=copy.copy(DL)
    a,d=np.shape(DL)
    T = np.arange(self.Tmin,(self.Tmax+self.Tstep),self.Tstep)
    flux=self.BB_Emissivity*self.sigma*(T+273.16)**(4)
    K=np.zeros((d,self.polynomial_degree+1))
    
    for i in range (d):
      if 0 not in DL[:,i]:
	K[i]=np.polyfit(DL[:,i],flux,self.polynomial_degree)
	DLC_Calibrated[:,i]=np.polyval(K[i],DL[:,i])
    np.save(self.result_directory+'coeffs_DL2Flux',K)  # Saves the 3D-array into a .npy file, wich can be re-open with np.load()
    return DLC_Calibrated


  def bad_pixels(self,DL,shape):
"""
This function spots the bad pixels in each image, and set their value to 0. It keeps the bad pixels of the worst image(with the maximum numbers of BP)
  Returns:
    DL_final, wich is the DL with mean value replacing the bad pixels. You are not supposed to use this for anything other than display purpose.
    mouchard_final : the matrix of bad pixels (1 =good, 0=bad)
    last_nbr_BP : the number of bad pixels spoted
  Saves the mouchard matrix in the result directory.
"""
    DL_final={}
    DL_mat={}
    DL_reshaped=[]
    last_nbr_BP=0
    best_mouchard=[]
    
    
    for i in range(0,(self.Tmax-self.Tmin)/self.Tstep):
      DL_mat[i]=np.reshape(DL[i],shape)
      DL_mat[i],mouchard,nbr_BP=self.bad_pixels_detection_grad(DL_mat[i])
      DL_reshaped.append(np.reshape(DL_mat[i],(1,shape[0]*shape[1])))
      if nbr_BP>last_nbr_BP:
	best_mouchard=mouchard
	last_nbr_BP=nbr_BP
    DL_final = np.concatenate([DL_reshaped[i] for i in range(0,(self.Tmax-self.Tmin)/self.Tstep)],axis=0)
    mouchard_final=np.reshape(best_mouchard,(1,shape[0]*shape[1]))
    np.save(self.result_directory+'mouchard',mouchard_final)
    return DL_final,mouchard_final,last_nbr_BP



  def bad_pixels_detection_grad(self,DL):
"""
This function is called by the bad_pixels function. It spots the bad pixels, based on the gradient of energy of each pixel
"""
    DL_corrected=copy.copy(DL)
    DLC_NUC=copy.copy(DL)
    b,c = np.shape(DL)
    mouchard=np.zeros((b,c));
    nbr_BP=0;

    gx,gy=np.gradient(DL)
    E=(gx*gx+gy*gy)**0.5
    seuil=0
    k=0
    crit=self.bad_pix_critere*np.mean(E[:]) # critere de jugement des mauvais pixels
    for i in range(1,b-1):
	for j in range(1,c-1):
	  seuil=E[i,j]+crit
	  if E[i-1,j] > seuil:
	      if(E[i+1,j] > seuil):
		  if(E[i,j-1] > seuil):
		      if(E[i,j+1] > seuil):
			  DL_corrected[i,j]=0
			  nbr_BP +=1
			  mouchard[i,j]=1
			  DLC_NUC[i,j]= 0.25*(DL[i+1,j] + DL[i-1,j]+ DL[i,j-1]+ DL[i,j+1])  # This DLC is supposed to be used ONLY for the NUC. 
    return DLC_NUC,mouchard,nbr_BP



  def apply_coeffs(self,DL):
"""
This function applies all the transformations (NUC, Calibration, and BP) to an image(NOT vectorised), with the coefficients stored in the result directory     
   return:
      masked_reshaped: map of temperatures if Flux=True, otherwise it's a map of the DL with NUC applied
"""
    b,c=np.shape(DL[0])
    DL_vect, DL_mean, shape=self.reshape(DL)
    mouchard=np.load(self.result_directory+'mouchard.npy')
    if self.NUC==True:
      self.NUC_coeffs=np.transpose(np.load(self.result_directory+'coeffs_NUC.npy'))
      DLC_NUC=np.polyval(self.NUC_coeffs,DL_vect)
      matrix=DLC_NUC
      if self.Flux==True:
	DL2Flux=np.transpose(np.load(self.result_directory+'coeffs_DL2Flux.npy'))
	flux=np.polyval(DL2Flux,DLC_NUC)
    else:
      if self.Flux==True:
	DL2Flux=np.transpose(np.load(self.result_directory+'coeffs_DL2Flux.npy'))
	flux=np.polyval(DL2Flux,DL_vect)
    if self.Flux== True:
      T=((flux/(self.BB_Emissivity*self.sigma))**(0.25)-273.16)  
      matrix=T
    masked=ma.masked_array(matrix,mouchard,fill_value=0)  ### applique une matrice mask
    masked_reshaped=np.reshape(masked,(b,c))  
    return masked_reshaped
    

  def heat_map(self,DL):
"""
This function plots the image with colorbar and 3-standart-deviation rule.
"""
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(DL)
    plt.clim(DL.mean()-3*DL.std(),DL.mean()+3*DL.std())
    plt.colorbar()
    plt.show(block=False)
    

  def verif_calibration(self,imageBB):
"""
This function displays the BB-images after applying all the calibration-corrections.
"""
    T_map_reshaped={}
    DL={}
    for i in range(0,(self.Tmax-self.Tmin)/self.Tstep): 
      DL[0]=imageBB[i]
      T_map_reshaped[i]=self.apply_coeffs(DL)
      self.heat_map(T_map_reshaped[i])
      

  def apply_to_essay(self,save_tif,video):
"""
This function applies the calibration to ALL the images in the essay. MAKE SURE your calibration is right before launching this. 
Images are saved in .tiff 16 bits, wich allow to store temperature information directly inside:
    - for T<65°C. Temperature = pixelValue /1000. mK precision
    - for 65<T<650 , Temperature = Value/100, 10mK precision
    - for T >650, Temperature = Value /10, 100mK precision
"""
    test1=PTW.GetPTW(self.video_directory+video)
    frame_nbr=test1.number_of_frames
    frame_rate=test1.frame_rate
    frame={}
    for i in range (0,frame_nbr):
      test1.get_frame(i)
      frame[0]=(test1.frame_data)
      print frame[0].max()
      T_map2=self.apply_coeffs(frame)
      if self.Flux==True:
	T_map=np.clip(T_map2,self.Tmin,self.Tmax)
	power=np.floor(np.log10(2**16/(max(T_map)))) # For storing Temperature information in a 16 bits tiff as integer 
	T_map2=(np.around(T_map,decimals=power))*10**power
      T_map3=(T_map2).astype(np.uint16)
      print T_map2.max()
      print T_map3.max()
      if save_tif==True:
	io.imsave(self.result_directory+"IR_images/"+"img_"+str(video)+".tiff",T_map3) #.astype(np.int16)


  def apply_to_essay_mean(self,save_tif,video):
"""
This function applies the calibration to ALL the images in the essay and evaluate the mean value for each pixels. MAKE SURE your calibration is right before launching this. 
Images are saved in .tiff 16 bits, wich allow to store temperature information directly inside:
    - for T<65°C. Temperature = pixelValue /1000. mK precision
    - for 65<T<650 , Temperature = Value/100, 10mK precision
    - for T >650, Temperature = Value /10, 100mK precision
"""
    test1=PTW.GetPTW(self.video_directory+video)
    frame_nbr=test1.number_of_frames
    frame[0]=0
    for i in range (0,frame_nbr):
      test1.get_frame(i)
      frame[0]+=(test1.frame_data)/(1.*frame_nbr)
    T_map2=self.apply_coeffs(frame)
    if self.Flux==True:
      T_map=np.clip(T_map2,self.Tmin,self.Tmax)
      power=np.floor(np.log10(2**16/(max(T_map)))) # For storing Temperature information in a 16 bits tiff as integer 
      T_map2=(np.around(T_map,decimals=power))*10**power
    T_map3=(T_map2).astype(np.uint16)
    if save_tif==True:
	io.imsave(self.result_directory+"IR_images/"+"img_"+str(video)+"_mean.tiff",T_map3) #.astype(np.int16)


