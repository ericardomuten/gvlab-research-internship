import numpy as np
import pandas as pd
import serial
import time
from openpyxl import Workbook

#NEED TO BE CHANGE
period = 0.02 #period of time for taking 1 frame of data (1/frequency)
port = 'COM8'

#PREPARATION FROM HERE
#setting up the COM, baudrate, parity bit, stopbit, and data size for the serial port that connect to the sensor's board
ser = serial.Serial(
    port=port,\
    baudrate=230400,\
    parity=serial.PARITY_NONE,\
    stopbits=serial.STOPBITS_ONE,\
    bytesize=serial.EIGHTBITS,\
        timeout=0)

print("connected to: " + ser.portstr)

ser.write(b'020100') #0201XX > set the sampling cycle, 00 for 4 ms and FA for 254 ms, we want as fast as possible = 00
ser.write(b'020302') #02030X > read from X sensors (start from 01 to 04)
ser.write(b'020201') #02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)

file_read = pd.read_excel("D:/TUAT/GV Lab Research Internship/Machine Learning/Calibration Coefficient Sensor2.xlsx", "Sheet1")
coefficient = np.zeros((3,4))
standard_value = np.zeros((4,1))
voltage = np.zeros((4,4))
T = np.zeros((4,1))
force = np.zeros((3,4))
max_voltage = 3.3 #maximum voltage known from the sensor's instruction manual
return_value_bit = 10 #number of return value bit known from the sensor's instruction manual
voltage_percentage = max_voltage/((2**return_value_bit)-1)
#fill the coefficient matrix with numbers known from the sensor's instruction manual
for i in range (0, 3):
    for j in range (0, 4):
        coefficient[i, j] = file_read[j][i]
#fill the standard value matrix with numbers known from the sensor's instruction manual
for i in range (0, 4):
    standard_value[i, 0] = file_read[i][4]

#port_sampling = "2001F70201007E020A01E701ED0062021400DD0064003800260021001E001D001C"  #input from manual (notepad)
#num_channel = int(int(port_sampling[0] + port_sampling[1], 16) / 8)  #determining number of active sensor's channel if input from manual
port_sampling = str(ser.readline())  #input from port
num_channel = int(int(port_sampling[2] + port_sampling[3], 16) / 8)  #determining number of active sensor's channel

wb1 = Workbook()  # temporary excel file
ws1 = wb1.active  # choose the active worksheet
ws1['A1'] = "Sensor/Channel"
ws1['B1'] = "Offset X"
ws1['C1'] = "Offset Y"
ws1['D1'] = "Offset Z"
coordinate_column = ["B", "C", "D"]

ave = np.zeros((num_channel,3)) #average offset for each sensor each axis

loading = [">                    |","-->                  |", "---->                |", "------>              |", "-------->            |", "---------->          |",
           "------------>        |", "-------------->      |", "---------------->    |", "------------------>  |", "-------------------->|"]
#Preparation done until here-----------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


count=0
print(loading[0])
#ser.write(b'020201')  # 02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
while count<1000: #keep reading data from port, calculate, and send it to the model
    ser.write(b'020201')  # 02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
    time.sleep(period)  #period for taking 1 frame
    port_sampling = str(ser.readline())  # input from port
    port_sampling = port_sampling[2:(4+16*num_channel)] #cutting the excess b' and \r\n' in the read from port value

    # start from here is to calculate force from the port sampling and calibration coefficient
    for channel in range(0, num_channel):
        for axis in range(0, 4):  # 3 axis + 1 Temperature
            voltage[channel, axis] = voltage_percentage * int(
                port_sampling[2 + 16 * channel + 4 * axis] + port_sampling[3 + 16 * channel + 4 * axis] + port_sampling[
                    4 + 16 * channel + 4 * axis] + port_sampling[5 + 16 * channel + 4 * axis], 16)  # sampling voltage
            voltage[channel, axis] = voltage[channel, axis] - standard_value[axis, 0]  # delta voltage
    for channel in range(0, num_channel):  # fill T with the value of delta voltage of Temperature
        T[channel, 0] = voltage[channel, 3]
    for channel in range(0, num_channel):
        for axis in range(0, 3):  # back to 3 axis because we don't need to calculate temperature anymore
            voltage[channel, axis] = voltage[channel, axis] - coefficient[axis, 3] * T[
                channel, 0]  # delta voltage acent
        voltage[channel, 3] = 0  # delete all the delta voltage of Temperature
    voltage = voltage.transpose()  # transpose the voltage matrix so we can do the matrix multiplication with the coefficient
    np.matmul(coefficient, voltage, force)  # the results is 3x4 matrix, the column represent each channel and the row represent each axis (Fx, Fy, Fz)
    # ------------------------------------------------------------------
    for channel in range(0, num_channel):
        for axis in range (0, 3):
            ave[channel, axis] = ave[channel, axis] + force[axis, channel]

    count = count + 1
    if (count%100 == 0):
        print(loading[int(count/100)])
    #ser.write(b'020201')  # 02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
ser.close() #closing the serial port

print("saving ...")
for channel in range (0, num_channel):
    ws1["A" + str(channel + 2)] = "Sensor " + str(channel + 1)
    for axis in range (0, 3):
        ws1[coordinate_column[axis] + str(channel + 2)] = ave[channel, axis]/count

wb1.save("D:/TUAT/GV Lab Research Internship/Machine Learning/" + "Offset.xlsx")

print("sensor calibration finish")


