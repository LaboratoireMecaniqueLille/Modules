import zmq
import random
import sys
import time
import numpy as np
import cv2
import skimage.io as io
import zmqnumpy as zmqnp

class fake_camera():
  """
  Create a dummy camera class, wich allows one to test a script with the same syntax but without a camera plugged in
  """
  def __init__(self,image_directory):
    self.img=(io.imread(image_directory)).astype(np.uint16)

  def read(self):
    return 1, self.img


def ZOI_selection(image):
  """
  Do a mouseclick somewhere, move the mouse to some destination, release
  the button.  This class gives click- and release-events and also draws
  a line or a box from the click-point to the actual mouseposition
  (within the same axes) until the button is released.  Within the
  method 'self.ignore()' it is checked wether the button from eventpress
  and eventrelease are the same.
  """
  from matplotlib.widgets import RectangleSelector
  import numpy as np
  import matplotlib.pyplot as plt
  rectprops = dict(facecolor='red', edgecolor = 'red', alpha=0.5, fill=True)

  #image=plt.imread("taches.png")
  #image=image[::,::,0]

  def line_select_callback(eclick, erelease):
      'eclick and erelease are the press and release events'
      global xmin, ymin, xmax, ymax
      x1, y1 = eclick.xdata, eclick.ydata
      x2, y2 = erelease.xdata, erelease.ydata
      #var=x1
      xmin=round(min(x1,x2))
      xmax=round(max(x1,x2))
      ymin=round(min(y1,y2))
      ymax=round(max(y1,y2))
      print ("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (xmin, ymin, xmax, ymax))
      

  def toggle_selector(event):
      toggle_selector.RS.set_active(False)



  fig, current_ax = plt.subplots()                    # make a new plotingrange
  #N = 100000                                       # If N is large one can see
  #x = np.linspace(0.0, 10.0, N)                    # improvement by use blitting!

  plt.imshow(image,cmap='gray')  # plot something



  # drawtype is 'box' or 'line' or 'none'
  toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
					drawtype='box', useblit=True,
					button=[1,3], # don't use middle button
					minspanx=5, minspany=5,rectprops=rectprops,
					spancoords='pixels')

  #plt.connect('key_press_event', toggle_selector)
  plt.show()
  return xmin, xmax, ymin, ymax


def stream_server(device,topics,*args):
  """Read a video flux on a ximea device, and stream it with said topics (1 by reader), on port defined as *args (5556 by default)
    device: a cv2.VideoCapture instance, with the proper set up done (shape, exposure, ...)
    topics: number of output you need
    you can define the port as args.
  """
  #cap=device
  
  numdevice=0 # n° of the camera. 0 if you only have one plugged
  exposure=5000 # exposition time, in microseconds
  gain=1 
  height=2048# reducing this one allows one to increase the FPS
  width=2048 # doesn't work for this one
  data_format=0 #0 = 8 bits. BE AWARE, at this point, errors may occur if you switch to 10 or 16 bits
  offset_X=0 # change this parameters only if height and width < 2048. It allows you to change the position of the cropped region you want to see. See documentation for acceptable values
  offset_Y=0
  cap = cv2.VideoCapture(cv2.CAP_XIAPI + numdevice)
  cap.set(cv2.CAP_PROP_XI_DATA_FORMAT,data_format)
  cap.set(cv2.CAP_PROP_XI_AEAG,0)#auto gain auto exposure
  cap.set(cv2.CAP_PROP_FRAME_WIDTH,width);  # doesn't work for this one
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height); # reducing this one allows one to increase the FPS
  cap.set(cv2.CAP_PROP_XI_OFFSET_Y,offset_X); # Vertical axis
  cap.set(cv2.CAP_PROP_XI_OFFSET_X,offset_Y) # horizontal axis from the left
  cap.set(cv2.CAP_PROP_EXPOSURE,exposure) # setting up exposure
  cap.set(cv2.CAP_PROP_GAIN,gain) #setting up gain
  #topics=1 # number of output you want to generate
  time.sleep(1)
  
  ret,frame=cap.read()
  print frame.shape
  
  
  port = "5556" #default port
  if len(args) != 0:  # you can pass another port in argument
      port =  int(args[0])

  context = zmq.Context()
  socket = context.socket(zmq.PUB)  # open a PUBlishing socket
  socket.bind("tcp://*:%s" % port) #bind socket to said port

  #j=0
  #t0=time.time()
  #t100=t0
  #messagedata = np.ones((2000,2000))
  i=0
  while True:
    #if j%100==0 and j>0:
      #print "mean_FPS_send= ",(100/(time.time()-t100))
      #t100=time.time()
    ret, messagedata = cap.read()
    #print i,ret
    #print "messagedata.shape= ",messagedata.shape
    str_shape=(np.array((messagedata.shape[0],messagedata.shape[1]))).tostring()
    str_messagedata=(messagedata.astype(np.uint8)).tostring()
    #messagedata = np.ones((2000,2000))*j
    #print "%s %s" % (topic, messagedata)
    for topic in range(topics):
      socket.send("%s+DECOUPAGE++%s+DECOUPAGE++%s" % (topic,str_shape,str_messagedata)) # send messagedata with topic. Only a socket reading this topic will receive the message
    i+=1
    #print "FPS_send= ", (1/(time.time()-t0))
    #print "j = ", j
    #print "loop"
    #time.sleep(0.001)
    #j+=1
    
class stream_reader():
  """Read a ximea video stream, with said topic, on port defined as *args (5556 by default)
    topic: number of the input you need to read
    you can define the port as args.
  """
  def __init__(self,topic,*args):
    port = "5556" #default port
    if len(args) !=0:  # you can pass another port in argument
      port =  int(args[0])
    context = zmq.Context() #open context
    socket = context.socket(zmq.SUB) #open SUBscribe socket to receive 
    topicfilter = str(topic) # choose the topic you want to read
    socket.setsockopt(zmq.SUBSCRIBE, topicfilter) # set topic option
    self.socket=socket
    self.port = port

  def read(self):
    """read once and returns the data
    """
    #t0=time.time()
    #print "loop"
    self.socket.connect ("tcp://localhost:%s" % self.port) # connect the socket
    string = self.socket.recv() # read message 
    self.socket.disconnect ("tcp://localhost:%s" % self.port) # disconnect to clear the buffer
    #topic, messagedata = string.split("+") # split topic from message
    data=string.split("+DECOUPAGE++")
    #print len(data)
    topic=data[0]
    str_shape=data[1]
    str_messagedata=data[2]
    shape=np.fromstring(str_shape,dtype=int,count=2)
    #print "shape= ",shape[0], shape[1]
    messagedata=np.fromstring(str_messagedata,dtype=np.uint8)
    #print len(messagedata)
    #print messagedata.shape
    frame=messagedata.reshape(shape[0],shape[1])
    #frame=messagedata
    #print  messagedata
    #print "FPS_recv= ", (1/(time.time()-t0))
    return frame