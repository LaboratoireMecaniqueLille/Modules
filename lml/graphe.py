import copy
import numpy as np
from multiprocessing import Process, Pipe
import matplotlib.pyplot as plt 
import scipy.signal as signal
import pandas as PD



def plot_time(data_size,graph_recv_n,graph_args): 
  """This function is supposed to be called by the graph function.
  plot up to 3 differents graphs in one figure, and keep a fixed abscisse range. 
  On update, old plots are erased and new ones are added.
  No memory overload, this plot is safe even for long plots.
  Arguments:
    - data_size is the number of column of the data sended through the pipe
    - graph_recv_n is the data input pipe
    - graph args are the number of the columns of data-array that needs to be plotted"""
  condition=True
  save_number=0
## init the plot
  nbr_graphs=len(graph_args)/2
  fig=plt.figure()
  ax=fig.add_subplot(111)
  li,= ax.plot(np.arange(5000),np.zeros(5000))
  if nbr_graphs ==2: # add a 2nd graph in the same plot
    lo,= ax.plot(np.arange(5000),np.zeros(5000))
  if nbr_graphs ==3: # add a 3rd graph
    la,= ax.plot(np.arange(5000),np.zeros(5000))
  #ax.set_ylim(0,1.2)
  fig.canvas.draw()     # draw and show it
  plt.show(block=False)
  #nbr=nbr_graphs*2
  nbr=data_size
  var=[[]]*nbr
  while condition==True:
    data=graph_recv_n.recv()
## this loop is used for the continous plotting
    if save_number>0:                              
      if save_number==1: # this loop init the first round of data
	for z in range(nbr):
	  var[z]=copy.copy(data[z])
      if save_number<10 and save_number>1: # This integer define the size of the plot: it plots "x" times the data. 
	for z in range(nbr):
	  var[z]=copy.copy(np.concatenate((var[z],(data[z])),axis=1))
      else :   # this loop delete the first values of the plot and add new value at the end to create a continuous plot
	for z in range(nbr):
	  var[z][:-np.shape(np.array(data))[1]] = var[z][np.shape(np.array(data))[1]:]
	  var[z][-np.shape(np.array(data))[1]:]= data[z]
      li.set_xdata(var[graph_args[0]])
      li.set_ydata(var[graph_args[1]])  # update the graph values #####################################################  delete rolling mean here
      if nbr_graphs ==2:
	lo.set_xdata(var[graph_args[2]])
	lo.set_ydata(var[graph_args[3]])
      if nbr_graphs ==3:
	la.set_xdata(var[graph_args[4]])
	la.set_ydata(var[graph_args[5]])
      ax.relim()
      ax.autoscale_view(True,True,True)
      fig.canvas.draw() 
    save_number+=1

def plot_value(graph_recv_n,graph_args):
  """This function is supposed to be called by the graph function.
  This function plot one or 2 graph of  y=f(x) , and you can choose y and x in the order variable.
  Autoscale, but doesn't reset. 
  BEWARE, long plots may cause data losses and slow the computer.
  arguments are:
    - graph_recv_n is the input data pipe
    - graph_args are the data to be plotted and args [-1] are the labels"""
  condition=True
  nbr=len(graph_args)-1 # number of variables (minus labels)
  plt.ion()
  fig=plt.figure()
  while condition==True:
    data=graph_recv_n.recv()  # receive data from main graph process
    plt.plot(data[graph_args[0]],data[graph_args[1]],'b-')
    plt.xlabel(graph_args[-1][0])
    plt.ylabel(graph_args[-1][1])
    if nbr ==4:
      plt.plot(data[graph_args[2]],data[graph_args[3]],'r-')
    plt.draw()
    
    
def graph(data_pipe,*args): 
  """This function as to be called in a process. It create the desired plots and updates the data in link with the save function.
  There is 2 possible plots:
    - value: plot y=f(x), no reset, scale expand without limitation : may cause latency and memory overflow if there is too many points.
    - time : plot y=f(x), fixed scale, it only plots the lasts few sets of points, allowing no memory leaks and almost infinite use.
  Syntax: 
  syntax is graph(data_pipe,[graph1],[graph2],....), with graph1/graph2 one of the following:
  value plot: ['value',x1,y1,x2,y2,['x_label','y_label']] . x2 and y2 are optionnal
  time plot : ['time', x1,y1,x2,y2,x3,y3] . x2->y3 are optionnal"""
  condition=True
  graph_send={}
  graph_recv={}
  graph_n={}
  data=data_pipe.recv() # this pipe receive data from the save function
  data_size=np.shape(data)[0]
  nbr_graphs=len(args)
  for i in range(nbr_graphs): 
    graph_type=args[i][0] # the first value of args[i] is the graph type
    graph_args=args[i][1:] # other values depend on the graph type
    graph_send[i],graph_recv[i]=Pipe() #creating pipes to communicate with the graphs to be created
    if graph_type=='values':
      graph_send[i].send(data) #init the pipe
      graph_n[i]=Process(target=plot_value,args=(graph_recv[i],graph_args)) # creating a new process for each graph
    if graph_type=='time':
      graph_send[i].send(data)#init the pipe
      graph_n[i]=Process(target=plot_time,args=(data_size,graph_recv[i],graph_args))# creating a new process for each graph
    graph_n[i].start() #start graphs processes
  while condition==True: # this loop will feed the pipes with new data received from the save process.
    data=data_pipe.recv()
    for i in range(nbr_graphs):
      graph_send[i].send(data)