import zmq
import random
import sys
import time
import numpy as np
import cv2

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
  cap=device
  
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
  while True:
    #if j%100==0 and j>0:
      #print "mean_FPS_send= ",(100/(time.time()-t100))
      #t100=time.time()
    ret, messagedata = cap.read()
    #messagedata = np.ones((2000,2000))*j
    #print "%s %s" % (topic, messagedata)
    for topic in range(topics):
      socket.send("%s + %s" % (topic,messagedata)) # send messagedata with topic. Only a socket reading this topic will receive the message
    #print "FPS_send= ", (1/(time.time()-t0))
    #print "j = ", j
    #print "loop"
    #time.sleep(0.001)
    #j+=1
    
class stream_reader():
  """Read a ximea video stream, with said topic, on port defined as *args (5556 by default)
    topics: number of the input you need to read
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
    topic, messagedata = string.split("+") # split topic from message
    #print  messagedata
    #print "FPS_recv= ", (1/(time.time()-t0))
    return messagedata