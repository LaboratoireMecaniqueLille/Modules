import numpy as np
import serial
import time

### Parameters
#limit = 0.00075 # limit for the eprouvette protection
##offset_=-0.0056
##protection_speed=1000. # nominal speed for the protection
#frequency=1000. # refreshing frequency (Hz)
##alpha = 1.05
#Vmax=1000


class Port:
  """This Class allow to define serial port parameters, open a port(see open_port), and move the corresponding motor(see move)"""
  def __init__(self,port_number,baud_rate=38400, timeout=1):
    self.port_number=port_number
    self.baud_rate=baud_rate
    self.timeout=timeout
    self.ser=None
    
    
  def open_port(self):
    """No arguments, open port, set speed mode and engage"""
    self.ser=serial.Serial(self.port_number,self.baud_rate,serial.EIGHTBITS,serial.PARITY_EVEN,serial.STOPBITS_ONE,self.timeout)
    self.ser.write("OPMODE 0\r\n EN\r\n")

  def close_port(self):
    """Close the designated port"""
    self.ser.close()

  def move(self,speed):
    """Set the speed"""
    self.ser.write("J "+str(speed)+"\r\n")    
    
  def CLRFAULT(self):
    self.ser.write("CLRFAULT\r\n")
    self.ser.write("OPMODE 0\r\n EN\r\n")


def protection_eprouvette(limit,frequency,Vmax,*args):
  """This function aim to keep the sensor value at the same level as the initial level, and moves the motor in consequence.
  args must be open Ports, paired with the corresponding sensor, and data pipes e.g. for each port: [port0, axe0,time_pipe,sensor_pipe,speed_pipe]"""
  condition=True
  speed=0
  speed_i=np.zeros(len(args))
  offset=np.zeros(len(args))
  for i in range(len(args)):
    print "Evaluating offset for port %s..." %i
    for j in range(int(2*frequency)):
      t_sensor, effort=args[i][1].get()
      offset[i]+=effort/(2.*frequency)   
    print "Done"
  t0=time.time()  #define origin of time for this test
  t=t0
  while condition==True:
    while (time.time()-t)<(1./(frequency*len(args))):
      indent=True
    t=time.time()
    for i in range(len(args)):
      t_sensor, effort=args[i][1].get()
      t_sensor-=t0 # use t0 as origin of time
      if (effort-offset[i]) >= limit:
	speed=-Vmax
      elif (effort-offset[i]) <= -limit:
	speed=Vmax
      else:
	speed=0
      if speed!=speed_i[i]:
	args[i][0].move(speed)
	#print "speed = %s" %speed
      speed_i[i]=speed
      args[i][2].send(t_sensor) # send data to the save function
      args[i][3].send(effort)
      args[i][4].send(speed)
      
    
  