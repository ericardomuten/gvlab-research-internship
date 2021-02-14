import numpy as np
import pandas as pd
import serial
import time
from openpyxl import Workbook

#NEED TO BE CHANGE
subject = "Tatsuki" #name of experiment subject
sensor1 = "Sensor1" #name of the sensor that being used for this data taking
sensor2 = "Sensor2" #name of the sensor that being used for this data taking
movement_type = "scratch" #movement name that being recorded
period = 0.02 #period of time for taking 1 frame of data (1/frequency)
total_frames = 8000 #number of frame that we want to record
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

#var_name = ["Fx", "Fy", "Fz"]
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

#preparing the offset correction
offset_file = pd.read_excel("D:/TUAT/GV Lab Research Internship/Machine Learning/" + "Offset.xlsx", "Sheet")
offset = np.zeros((num_channel, 3))
offset_column = ["Offset X", "Offset Y", "Offset Z"]
for channel in range (0, num_channel):
    for axis in range (0, 3):
        offset[channel, axis] = offset_file[offset_column[axis]][channel]

wb1 = Workbook()  # temporary excel file
ws1 = wb1.active  # choose the active worksheet
ws1['C1'] = "Time"
ws1['D1'] = "X"
ws1['E1'] = "Y"
ws1['F1'] = "Z"
ws1['G1'] = "T"
wb2 = Workbook()  # temporary excel file
ws2 = wb2.active  # choose the active worksheet
ws2['C1'] = "Time"
ws2['D1'] = "X"
ws2['E1'] = "Y"
ws2['F1'] = "Z"
ws2['G1'] = "T"
#Preparation done until here------------------------------------------------------------------------------------

count=0
#ser.write(b'020201')  # 02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
while count<total_frames: #keep reading data from port and calculate then print it (for 1000 times)
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
    #correction of force using the known offset values
    for channel in range (0, num_channel):
        for axis in range (0, 3):
            force[axis, channel] = force[axis, channel] - offset[channel, axis]

    # ------------------------------------------------------------------
    count = count + 1
    for channel in range(0, num_channel):
        print(str(count)+str(': \t ')+"Fx"+": "+str(force[0, channel])+"\t"+"Fy"+": "+str(force[1, channel])+"\t"+"Fz"+": "+str(force[2, channel])+"\t"+"T: "+str(float(T[channel]+standard_value[3, 0]))+"\n") #print all the force+T
    ws1["C"+str(count+1)] = period*count
    ws1["D"+str(count+1)] = force[0, 0]
    ws1["E"+str(count+1)] = force[1, 0]
    ws1["F"+str(count+1)] = force[2, 0]
    ws1["G"+str(count+1)] = float(T[0]+standard_value[3, 0])
    ws2["C" + str(count+1)] = period*count
    ws2["D" + str(count+1)] = force[0, 1]
    ws2["E" + str(count+1)] = force[1, 1]
    ws2["F" + str(count+1)] = force[2, 1]
    ws2["G" + str(count+1)] = float(T[1] + standard_value[3, 0])
    #ser.write(b'020201')  # 02020X > stop sending the data(0), sending the data once(1), keeps sending the data(2)
ser.close() #closing the serial port

wb1.save("D:/TUAT/GV Lab Research Internship/Machine Learning/Data/Data_Raw/" + sensor1 + "/" + subject + "/" + movement_type + ".xlsx")
wb2.save("D:/TUAT/GV Lab Research Internship/Machine Learning/Data/Data_Raw/" + sensor2 + "/" + subject + "/" + movement_type + ".xlsx")
