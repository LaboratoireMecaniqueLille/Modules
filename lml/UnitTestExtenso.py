"""
Fake experiment allowing to use the extension measurement method
Simulation of a tensile test on a on-compressible material such as rubber.
One generates white circular marks that can be measured on real experiment.
"""

from skimage.draw import ellipse
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filter import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import label, closing,erosion, square,dilation,disk
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.filter import threshold_otsu, rank, threshold_yen
import time
import matplotlib.patches as mpatches
import matplotlib.mathtext as mathtext

# Number pixels in x & y dimensions
Npixx=2048.
Npixy=1024.
# Radius of the circular mark
radius=50.

xmin=Npixx/2-250.
ymin=Npixy/2.-250.
ymax=Npixy/2+250.
xmax=Npixx/2+250.
L0x=xmax-xmin
L0y=ymax-ymin

def specimen(Npixx,Npixy,xmin,ymin,xmax,ymax,radius,L0x,L0y):
  """ 
  Function that create an image with 4 centered ellpitical marks
  """
  image=np.zeros((Npixx,Npixy))
  Lx=xmax-xmin
  Ly=ymax-ymin
  rr, cc = ellipse(xmin, (ymax+ymin)/2, Lx/L0x*radius,Ly/L0y*radius)
  image[rr,cc]=255
  rr, cc = ellipse(xmax, (ymax+ymin)/2, Lx/L0x*radius,Ly/L0y*radius)
  image[rr,cc]=255
  rr, cc = ellipse((xmin+xmax)/2, ymin, Lx/L0x*radius,Ly/L0y*radius)
  image[rr,cc]=255
  rr, cc = ellipse((xmin+xmax)/2, ymax, Lx/L0x*radius,Ly/L0y*radius)
  image[rr,cc]=255
  return image.astype(np.uint8)


def barycenter(image,minx,miny,maxx,maxy,thresh,border,White_Mark):
  """
  Alternative computatition of the barycenter (moment 1 of image) on ZOI
  """
  bw=rank.median(image[minx:maxx+1,miny:maxy+1],square(5))>thresh*255
  if(White_Mark==False):
    bw=1-bw  
  [Y,X]=np.meshgrid(range(miny,maxy+1),range(minx,maxx+1))
  Px=(X*bw).sum().astype(float)/bw.sum()
  Py=(Y*bw).sum().astype(float)/bw.sum()
  Onex,Oney=np.where(bw==1)
  minxi=Onex.min()
  maxxi=Onex.max()
  minyi=Oney.min()
  maxyi=Oney.max()
  minx=X[minxi,minyi]-border
  miny=Y[minxi,minyi]-border
  maxx=X[maxxi,maxyi]+border
  maxy=Y[maxxi,maxyi]+border
  return Px,Py,minx,miny,maxx,maxy

# Creation of the first non deformed image
image = specimen(Npixx,Npixy,xmin,ymin,xmax,ymax,radius,L0x,L0y)
# Initialization of the video extensometer
thresh = threshold_otsu(image)
bw= image>thresh
label_image = label(bw)

# Create the empty vectors for corners of each ZOI
regions=regionprops(label_image)
NumOfReg=len(regions)
minx=np.empty([NumOfReg,1])
miny=np.empty([NumOfReg,1])
maxx=np.empty([NumOfReg,1])
maxy=np.empty([NumOfReg,1])

Points_coordinates=np.empty([NumOfReg,2])

# Definition of the ZOI margin regarding the regionprops box
border=4

# Definition of the ZOI and initialisation of the 
i=0
for region in regions:
  minx[i], miny[i], maxx[i], maxy[i]= region.bbox
  i+=1

speed=3
#initialisation of the figure
plt.ion()
fig=plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(image,cmap='gray')
rec={}
center={}
for i in range(0,NumOfReg): 
  rect = mpatches.Rectangle((miny[i], minx[i]), maxy[i] - miny[i], maxx[i] - minx[i],fill=False, edgecolor='red', linewidth=2)
  rec[i]=ax.add_patch(rect)
  center[i],= ax.plot(Points_coordinates[i,1],Points_coordinates[i,0],'+g',markersize=5)
im.set_extent((0,image.shape[1],0,image.shape[0]))
Exx = "Exx = 0 %%"
Eyy = "Eyy = 0 %%"
exx=ax.text(10, 10, Exx, fontsize=12,color='white', va='bottom')
eyy=ax.text(10, 110, Eyy, fontsize=12,color='white', va='bottom')
plt.draw()
time.sleep(0.1)

# Main loop of the simulation
for i in range(0,100):
  xmin-=speed
  ymin+=0.5*speed # rubber poisson coef = 0.5
  xmax+=speed
  ymax-=0.5*speed # rubber poisson coef = 0.5  
  image = specimen(Npixx,Npixy,xmin,ymin,xmax,ymax,radius,L0x,L0y)
  for i in range(0,NumOfReg): 
    Points_coordinates[i,0],Points_coordinates[i,1],minx[i],miny[i],maxx[i],maxy[i]=barycenter(image,int(minx[i]),int(miny[i]),int(maxx[i]),int(maxy[i]),thresh,border,True)
    # Update of the boundig box and center
    rec[i].set_x(miny[i])
    rec[i].set_y(minx[i])
    rec[i].set_width(maxy[i] - miny[i])
    rec[i].set_height(maxx[i] - minx[i])
    center[i].set_xdata(Points_coordinates[i,1])
    center[i].set_ydata(Points_coordinates[i,0])
  Lx=Points_coordinates[:,0].max()-Points_coordinates[:,0].min()
  Ly=Points_coordinates[:,1].max()-Points_coordinates[:,1].min()
  #print "EpsXX = %s" %(Lx/L0x)
  #print "EpsYY = %s" %(Ly/L0y)
  exx.set_text("Exx = %d %%"%(100*(Lx/L0x-1)))
  eyy.set_text("Eyy = %d %%"%(100*(Ly/L0y-1)))
  # Update of the image
  im.set_array(image)
  plt.draw()
  