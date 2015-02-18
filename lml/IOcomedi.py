import comedi as c
import time
import copy
import os
import sys, string, struct
from multiprocessing import Array
import numpy as np

class Out:
  """Define an output channel and allows one to send signal through it"""
  def __init__(self,K=1,Ki=0,Kd=0,device='/dev/comedi0',subdevice=1,channel=0,range_num=1,gain=1,offset=0,out_min=0,out_max=4.095):
    self.subdevice=subdevice
    self.channel=channel
    self.range_num=range_num
    self.device0=c.comedi_open(device)
    self.maxdata=c.comedi_get_maxdata(self.device0,self.subdevice,self.channel)
    self.range_ds=c.comedi_get_range(self.device0,self.subdevice,self.channel,self.range_num)
    self.out=0
    self.gain=gain
    self.offset=offset
    self.I_term=0
    self.last_sensor_input=0
    self.K=K
    self.Ki=Ki
    self.Kd=Kd
    self.last_time=time.time()
    self.out_min=out_min
    self.out_max=out_max
    self.last_output=0
    
  def set_(self,wanted_position):
    """send a signal"""
    self.out=(wanted_position/self.gain)-self.offset
    out_a=c.comedi_from_phys(self.out,self.range_ds,self.maxdata) # convert the wanted_position 
    c.comedi_data_write(self.device0,self.subdevice,self.channel,self.range_num,c.AREF_GROUND,out_a) # send the signal to the controler
    t=time.time()
    return (t,self.out)
      
  def set_PID(self,wanted_position,sensor_input):
    """send a signal through a PID, based on the wanted command and the sensor_input"""
    self.time= time.time()
    self.out=(wanted_position/self.gain)-self.offset

    self.error=self.out-sensor_input
    self.I_term += self.Ki*self.error*(self.last_time-self.time)
    
    if self.I_term>self.out_max:
      self.I_term=self.out_max
    elif self.I_term<self.out_min:
      self.I_term=self.out_min
    
    self.out_PID=self.last_output+self.K*self.error+self.I_term-self.Kd*(sensor_input-self.last_sensor_input)/(self.last_time-self.time)
    
    if self.out_PID>self.out_max:
      self.out_PID=self.out_max
    elif self.out_PID<self.out_min:
      self.out_PID=self.out_min
      
    self.last_time=copy.copy(self.time)
    self.last_sensor_input=copy.copy(sensor_input)
    self.last_output=copy.copy(self.out_PID)
    out_a=c.comedi_from_phys(self.out_PID,self.range_ds,self.maxdata) # convert the wanted_position 
    c.comedi_data_write(self.device0,self.subdevice,self.channel,self.range_num,c.AREF_GROUND,out_a) # send the signal to the controler
    t=time.time()
    return (t,self.out_PID)

class In:
  """Define an input channel and allows one to receive signal through it"""
  def __init__(self,device='/dev/comedi0',subdevice=0,channel=1,range_num=0,gain=1,offset=0): 
    self.subdevice=subdevice
    self.channel=channel
    self.range_num=range_num
    self.device0=c.comedi_open(device)
    self.maxdata=c.comedi_get_maxdata(self.device0,self.subdevice,self.channel)
    self.range_ds=c.comedi_get_range(self.device0,self.subdevice,self.channel,self.range_num)
    self.gain=gain
    self.offset=offset

  def get(self):
    """Read the signal"""
    data = c.comedi_data_read(self.device0,self.subdevice,self.channel,self.range_num, c.AREF_GROUND)
    self.position=(c.comedi_to_phys(data[1],self.range_ds,self.maxdata)*self.gain+self.offset)
    t=time.time()
    return (t, self.position)
  
def streamer(device,subdevice,chans,comedi_range,shared_array):
  ''' read the channels defined in chans, on the device/subdevice, and streams the values in the shared_array.
  The shared_array has a lock, to avoid reading and writing at the same time and it's process-proof.
  device: '/dev/comedi0'
  subdevice : 0=in, 1=out
  chans : [0,1,2,3,4....] : BE AWARE the reading is done crescendo, no matter the order given here. It means that [0,1,2] and [2,0,1] will both have [0,1,2] as result, but you can ask for [0,1,5].
  comedi_range: same size as chans, with the proper range for each chan. If unknown, try [0,0,0,....].
  shared_array: same size as chans, must be defined before with multiprocess.Array: shared_array= Array('f', np.arange(len(chans)))
  '''
  dev=c.comedi_open(device)
  if not dev: raise "Error openning Comedi device"

  fd = c.comedi_fileno(dev) #get a file-descriptor for use later

  BUFSZ = 10000 #buffer size
  freq=8000# acquisition frequency: if too high, set frequency to maximum.
 
  nchans = len(chans) #number of channels
  aref =[c.AREF_GROUND]*nchans

  mylist = c.chanlist(nchans) #create a chanlist of length nchans
  maxdata=[0]*(nchans)
  range_ds=[0]*(nchans)

  for index in range(nchans):  #pack the channel, gain and reference information into the chanlist object
    mylist[index]=c.cr_pack(chans[index], comedi_range[index], aref[index])
    maxdata[index]=c.comedi_get_maxdata(dev,subdevice,chans[index])
    range_ds[index]=c.comedi_get_range(dev,subdevice,chans[index],comedi_range[index])

  cmd = c.comedi_cmd_struct()

  period = int(1.0e9/freq)  # in nanoseconds
  ret = c.comedi_get_cmd_generic_timed(dev,subdevice,cmd,nchans,period)
  if ret: raise "Error comedi_get_cmd_generic failed"
	  
  cmd.chanlist = mylist # adjust for our particular context
  cmd.chanlist_len = nchans
  cmd.scan_end_arg = nchans
  cmd.stop_arg=0
  cmd.stop_src=c.TRIG_NONE

  t0 = time.time()
  j=0
  ret = c.comedi_command(dev,cmd)
  if ret !=0: raise "comedi_command failed..."

#Lines below are for initializing the format, depending on the comedi-card.
  data = os.read(fd,BUFSZ) # read buffer and returns binary data
  data_length=len(data)
  #print maxdata
  #print data_length
  if maxdata[0]<=65536: # case for usb-dux-D
    n = data_length/2 # 2 bytes per 'H'
    format = `n`+'H'
  elif maxdata[0]>65536: #case for usb-dux-sigma
    n = data_length/4 # 2 bytes per 'H'
    format = `n`+'I'
  #print struct.unpack(format,data)
    
# init is over, start acquisition and stream
  last_t=time.time()
  try:
    while True:
      #t_now=time.time()
      #while (t_now-last_t)<(1./frequency):
	#t_now=time.time()
	##print t_now-last_t
      #last_t=t_now
      data = os.read(fd,BUFSZ) # read buffer and returns binary data
      #print len(data), data_length
      if len(data)==data_length:
	datastr = struct.unpack(format,data) # convert binary data to digital value
	if len(datastr)==nchans: #if data not corrupted for some reason
	  #shared_array.acquire()
	  for i in range(nchans):
	    shared_array[i]=c.comedi_to_phys((datastr[i]),range_ds[i],maxdata[i])
	  #print datastr
	  #shared_array.release()
	#j+=1
	#print j
	#print "Frequency= ",(j/(time.time()-t0))
	#print np.transpose(shared_array[:])

  except (KeyboardInterrupt):	
    c.comedi_cancel(dev,subdevice)
    ret = c.comedi_close(dev)
    if ret !=0: raise "comedi_close failed..."