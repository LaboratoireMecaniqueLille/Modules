import comedi as c
import time
import copy


class Out:
  """Define an output channel and allows one to send signal through it"""
  def __init__(self, device='/dev/comedi0',subdevice=1,channel=0,range_num=1,gain=1,offset=0, out_min=0, out_max=4.095):
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
    self.out=(wanted_position-self.offset)/self.gain
    out_a=c.comedi_from_phys(self.out,self.range_ds,self.maxdata) # convert the wanted_position 
    c.comedi_data_write(self.device0,self.subdevice,self.channel,self.range_num,c.AREF_GROUND,out_a) # send the signal to the controler
    t=time.time()
    return (t,self.out)
      
  def set_PID(self,wanted_position,sensor_input):
    """send a signal through a PID, based on the wanted command and the sensor_input"""
    self.time= time.time()
    self.out=(wanted_position-self.offset)/self.gain

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