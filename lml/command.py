import numpy as np
import time
from multiprocessing import Pipe
import math
import pandas as PD

def f(I,O,path_x,path_y,time_pipe,sensor_pipe): 
  """allows you to control one actuator depending on a path: on time path_x[i], set the output command to path_y[i]
  args:
  I : function returning time and position
  O : function needing args ((path_y),position), and returning time and command
  path_x: an array of time values
  path_y: an array of values, same size than path_x
  time pipe: one end of a pipe, used to send time
  sensor pipe: one end of a pipe, used to send the sensor value"""
  for i in range(len(path_x)):
    t1=time.time()
    while t1<(t0+(path_x[i])):  # Waits for the time set in the path file
      t1=time.time() 
    a,b=(I()) # measuring the position and saving it in the shared variables
    c,d=(O(path_y[i], b)) # setting the position to a new value and saving it in the saherd variables
    time_pipe.send(a) # send data to the save function
    sensor_pipe.send(b)
  time_pipe.send(0.0) # signal that the acquisition is over
  sensor_pipe.send(0.0)
  
def save(recv_pipe,saving_step):
  """This function saves data in a file. BEWARE the log file needs to be cleaned before starting this function, otherwise it just keep writing a the end of the file.
     - you save one point every saving_step (1 to save them all)
     - recv_pipe is the incoming pipes with all the data in it."""
### INIT
  condition=True
  save_number=0
### Main loop
  while condition==True:
## init data matrixes and receive data
    data=recv_pipe.recv()
    nbr=np.shape(data)[0]
## The following loops are used to save the data
    fo=open("log.txt","a") # "a" for appending
    fo.seek(0,2) #place the "cursor" at the end of the file, so every writing will not erase the previous ones
    data_to_save=""
    data1=np.empty((np.shape(np.array(data))[0],int(math.ceil(len(data[0])//saving_step))))
    if saving_step>1:  # This loop mCLRFAULTeans the data to decrease the number of points to save
      for x in range(int(math.ceil(len(data[0])//saving_step))): # euclidian division here
	for i in range(np.shape(np.array(data))[0]):
	  if x<(len(data[0])//saving_step):
	    data1[i][x]=(np.mean(data[i][x*saving_step:(x+1)*saving_step]))
	  else:
	    data1[i][x]=(np.mean(data[i][x*saving_step:]))
      data_to_save=str(np.transpose(data1))+"\n"
    else:  # this loop save all data
      data_to_save=str(np.transpose(data))+"\n"
    fo.write(data_to_save)
    fo.close()
    save_number+=1
    
def pipe_compactor(acquisition_step,send_pipes,*args):
  """Receive n pipes (as args),and send back all the data as one to every pipes in send_pipes, after receiveing acquisition_step number of points"""
  nbr=len(args)
  condition=True
  while condition==True:
    data=[[0 for x in xrange(acquisition_step)] for x in xrange(nbr)] 
    i=0
    while i<acquisition_step and condition==True:
      for z in range (nbr):
	data[z][i]=args[z].recv()
      if data[0][i]==0.0: # if acquisiton is over, save remaining data
	condition=False
      i+=1
    for i in range(len(send_pipes)):
      send_pipes[i].send(data)
      
def signal_filter(width, recv_pipe, send_pipe):
  data=recv_pipe.recv()
  for i in range(shape(data)[0]):
    data[i]=PD.rolling_mean(np.asarray(data[i]),width)
  send_pipe.send(data)