# -*-# coding: utf-8 -*-

"""
This modele provides eneric loading, showing and saving of input data.

This includes some uncompleted function such as ...
"""

# import
import os
import numpy as np
import scipy
import sklearn as sk
import tensorflow as ts
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import subplot2grid, tight_layout, subplots_adjust
from scipy.interpolate import interp1d

import pylab

# custom import
from lowpass import butter_lowpass, butter_lowpass_filter

segment_points = []

def onclick(event):

    global segment_points
    print('event.button=%d, event.x=%d, event.y=%d, event.xdata=%f, \
    event.ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
    segment_point = int(round(event.xdata))
    segment_points.append(segment_point)


class Segment_data:

    def __init__(self, subjectName, fileName):

        self.subject = subjectName
        #self.cwd = os.getcwd()
        self.cwd = 'D:\\TUAT\\GV Lab Research Internship\\Machine Learning'
        #CHANGE THE FILE READ DIRECTORY HERE!!!!
        self.dataPath = 'D:\\TUAT\\GV Lab Research Internship\\Machine Learning' + '\\Data\\Data_Raw\\Sensor1\\' + self.subject + '\\'
        self.leftPath = fileName + '.csv'
        self.filename = fileName

        self.left = pd.read_csv(self.dataPath + self.leftPath)
        # self.left.drop(self.left.columns[len(self.left.columns)-1], axis=1, inplace=True)
        self.left = self.left.values[:, 3:6]
        #CHANGE THE SAVE PATH OF SEGMENT POINTS HERE!!!!
        self.savesp_path = self.cwd + '\\Data\\segment_points\\Sensor1\\' + self.subject + '\\'


    def load_ind_list(self):

        self.ind_list = pd.read_excel('Segment_Points_list2.xlsx')
        self.ind_list = self.sp_list.as_matrix()
        self.ind_name = self.sp_list[:, 0]
        self.ind1 = self.sp_list[:, 1]
        self.ind2 = self.sp_list[:, 2]

    def load_SP_list(self):

        self.segment_point_path = self.savesp_path + self.filename + '_segment_points.txt'

        x = []
        for l in open(self.segment_point_path).readlines():
            data = l[:-1].split(' ')
            x += [float(data[0])]
        self.sp_list = x
        print(self.sp_list)

        self.sp_list = np.array(self.sp_list)

    def calc_distance(self, ind1, ind2):

        self.ind1 = ind1
        self.ind2 = ind2

        self.distance_left = np.zeros([self.left.shape[0], 2])
        self.distance_right = np.zeros([self.right.shape[0], 2])

        if self.ind2 == []:
            self.distance_left = self.left[:, self.ind1]
        else:
            self.distance_left = np.linalg.norm(self.left[:, self.ind1*3-3:self.ind1*3]-self.left[:, self.ind2*3-3:self.ind2*3], axis=1)

        self.distance_left_filter = butter_lowpass_filter(self.distance_left, 2, 60)

    def plot_raw_data(self):

        plt.title(self.filename)
        plt.xlabel('Time')
        plt.ylabel('Position [mm]')
        self.time_raw = [i for i in range(0, self.left.shape[0])]
        plt.plot(self.time_raw, self.left[:, 0])

        plt.show()
        plt.close()

    def show_input(self, save_name):

        fig = plt.figure()
        # ax = fig.add_subplot(111, projection = '3d')
        ax = Axes3D(fig)

        plt.plt.ion()
        column = list(self.left.columns.values)
        l = range(0, self.left.shape[0])
        for t in l[0::10]:  # book1.shape[0]
            plt.pyplot.cla()
            for i in range(1, int(self.left.shape[1] / 3)):
                ax.set_xlabel("X-axis")
                ax.set_xlabel("Y-axis")
                ax.set_xlabel("Z-axis")
                # plt.xlim(xmin, xmax)
                # plt.ylim(ymin, ymax)
                # plt.zlim(zmin, zmax)
                x1 = self.left[t, i * 3 - 3]
                y1 = self.left[t, i * 3 - 2]
                z1 = self.left[t, i * 3 - 1]
                ax.scatter(x1, y1, z1, s=25, color='blue', edgecolor='blue')
                ax.text(x1, y1, z1, '%s' % (column[i * 3 - 2]), size=10)

            plt.pyplot.pause(0.05)

            plt.savefig(save_name + str(t) +".png")
            plt.pause(0.05)

    def plot_raw_distance(self):

        fig = plt.figure()
        plt.title(self.filename + '_distance_left')
        plt.xlabel('Time')
        plt.ylabel('Distance [mm]')
        self.time = [i for i in range(0, self.distance_left.shape[0])]
        plt.plot(self.time, self.distance_left)
        plt.plot(self.time, self.distance_left_filter)

        plt.show()
        plt.close()

    def splitout_manual(self):

        self.segment_point = []

        fig = plt.figure()
        plt.xlabel('Time')
        plt.ylabel('Distance [mm]')

        plt.plot(self.time, self.left, 'r')    # filtered value
        plt.plot(self.time, self.left_filter, 'k')  # raw value

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show

    def plot_segment(self):

        np.savetxt(self.savesp_path + self.filename + '_segment_points.txt', self.segment_points, delimiter=',')

        self.segment_middle = self.segment_points.shape[0]//2
        self.segment_start = self.segment_points[:self.segment_middle]
        self.segment_end = self.segment_points[self.segment_middle:]
        fig = plt.figure(figsize=(25, 12))
        plt.xlabel('Time')
        plt.ylabel('Distance [mm]')
        plt.plot(self.time, self.distance_left, 'r')  # filtered value
        plt.plot(self.time, self.distance_left_filter, 'k')  # raw value
        plt.scatter(self.segment_start, self.distance_left_filter[self.segment_start], marker='o', c='green')
        plt.scatter(self.segment_end, self.distance_left_filter[self.segment_end], marker='*', c='blue')

        plt.show()
        plt.savefig('train\\Shohei\\figure\\' + self.filename + '_SP.png')
        plt.close()

    def cut_and_spline(self, size = 100):

        self.nCycle = int(self.sp_list.shape[0] / 2)
        self.nJoint = self.left.shape[1]

        movement_left, movement_right = np.zeros((self.nCycle, size, self.nJoint)), np.zeros((self.nCycle, size, self.nJoint))
        movement_left_L, movement_right_L = np.zeros((self.nCycle, size, self.nJoint)), np.zeros((self.nCycle, size, self.nJoint))


        for a in range(0, self.nCycle):
            start = int(self.sp_list[a] + 1)
            end = int(self.sp_list[a + self.nCycle] + 1)
            length = end - start
            for i in range(0, self.nJoint):
                self.temp1_left = self.left[start:end, i]
                x = np.linspace(0, 1, length)
                x_new = np.linspace(0, 1, size)
                print(x.shape, 'x')
                print(x_new.shape, 'x_new')
                self.temp2_left = interp1d(x, self.temp1_left)(x_new)
                movement_left[a, :, i] = self.temp2_left

        for a in range(0, self.nCycle):

            self.train_path = self.cwd + '\\Data\\Cut_Data\\Sensor1\\' + self.subject + '\\' + str(size) + '\\'

            dfTime = pd.DataFrame([i for i in range(0, size)], columns=['Time'])

            self.movement_left = movement_left[a, :, :]
            df1 = pd.DataFrame(self.movement_left)

            df1.to_csv( self.train_path + self.filename + '_' + str(size) + '_' + str(a) + '.csv')

    def plot_temp(self, dataPath, fileName, ind1, ind2):

        self.dataPath = dataPath
        self.filename = fileName
        self.ind1 = ind1
        self.ind2 = ind2

        self.raw1 = pd.read_csv(self.dataPath + self.filename + '_30.csv')
        self.raw2 = pd.read_csv(self.dataPath + self.filename + '_40.csv')
        self.raw3 = pd.read_csv(self.dataPath + self.filename + '_50.csv')
        self.raw4 = pd.read_csv(self.dataPath + self.filename + '_60.csv')
        self.raw5 = pd.read_csv(self.dataPath + self.filename + '_70.csv')
        self.raw6 = pd.read_csv(self.dataPath + self.filename + '_80.csv')
        self.raw7 = pd.read_csv(self.dataPath + self.filename + '_90.csv')
        self.raw8 = pd.read_csv(self.dataPath + self.filename + '_100.csv')
        self.raw1 = self.raw1.as_matrix()
        self.raw2 = self.raw2.as_matrix()
        self.raw3 = self.raw3.as_matrix()
        self.raw4 = self.raw4.as_matrix()
        self.raw5 = self.raw5.as_matrix()
        self.raw6 = self.raw6.as_matrix()
        self.raw7 = self.raw7.as_matrix()
        self.raw8 = self.raw8.as_matrix()

        self.time1 = [j for j in range(1, 31)]
        self.time2 = [j for j in range(1, 41)]
        self.time3 = [j for j in range(1, 51)]
        self.time4 = [j for j in range(1, 61)]
        self.time5 = [j for j in range(1, 71)]
        self.time6 = [j for j in range(1, 81)]
        self.time7 = [j for j in range(1, 91)]
        self.time8 = [j for j in range(1, 101)]

        fig = plt.figure()
        plt.xlabel('Time')
        plt.ylabel('Position [mm]')
        plt.subplot(121)
        plt.plot(self.time1, self.raw1[0:30, self.ind1*3-3])
        plt.plot(self.time2, self.raw2[0:40, self.ind1*3-3])
        plt.plot(self.time3, self.raw3[0:50, self.ind1*3-3])
        plt.plot(self.time4, self.raw4[0:60, self.ind1*3-3])
        plt.plot(self.time5, self.raw5[0:70, self.ind1*3-3])
        plt.plot(self.time6, self.raw6[0:80, self.ind1*3-3])
        plt.plot(self.time7, self.raw7[0:90, self.ind1*3-3])
        plt.plot(self.time8, self.raw8[0:100, self.ind1*3-3])
        plt.plot(self.time1, self.raw1[0:30, self.ind1*3-3], label='30')
        plt.plot(self.time2, self.raw2[0:40, self.ind1*3-3], label='40')
        plt.plot(self.time3, self.raw3[0:50, self.ind1*3-3], label='50')
        plt.plot(self.time4, self.raw4[0:60, self.ind1*3-3], label='60')
        plt.plot(self.time5, self.raw5[0:70, self.ind1*3-3], label='70')
        plt.plot(self.time6, self.raw6[0:80, self.ind1*3-3], label='80')
        plt.plot(self.time7, self.raw7[0:90, self.ind1*3-3], label='90')
        plt.plot(self.time8, self.raw8[0:100, self.ind1*3-3], label='100')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode='expand')
        plt.xlim(0, 100)

        plt.subplot(122)
        plt.plot(self.time1, self.raw1[0:30, self.ind1*3-3+82])
        plt.plot(self.time2, self.raw2[0:40, self.ind1*3-3+82])
        plt.plot(self.time3, self.raw3[0:50, self.ind1*3-3+82])
        plt.plot(self.time4, self.raw4[0:60, self.ind1*3-3+82])
        plt.plot(self.time5, self.raw5[0:70, self.ind1*3-3+82])
        plt.plot(self.time6, self.raw6[0:80, self.ind1*3-3+82])
        plt.plot(self.time7, self.raw7[0:90, self.ind1*3-3+82])
        plt.plot(self.time8, self.raw8[0:100, self.ind1*3-3+82])
        plt.plot(self.time1, self.raw1[0:30, self.ind1*3-3+82], label='30')
        plt.plot(self.time2, self.raw2[0:40, self.ind1*3-3+82], label='40')
        plt.plot(self.time3, self.raw3[0:50, self.ind1*3-3+82], label='50')
        plt.plot(self.time4, self.raw4[0:60, self.ind1*3-3+82], label='60')
        plt.plot(self.time5, self.raw5[0:70, self.ind1*3-3+82], label='70')
        plt.plot(self.time6, self.raw6[0:80, self.ind1*3-3+82], label='80')
        plt.plot(self.time7, self.raw7[0:90, self.ind1*3-3+82], label='90')
        plt.plot(self.time8, self.raw8[0:100, self.ind1*3-3+82], label='100')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=8, mode='expand')
        plt.xlim(0, 100)

        plt.savefig(self.filename + '_all_splined.png')
        plt.show()

#1ST RUN FROM HERE TO TOP, DONT FORGET TO CHANGE THE DIRECTORY OF datapath, savesp, and train_path JUST LIKE IN THE data_operation_class.py!!!

# names = []

# f = open('config\\sign_list2.txt')
# names = f.read().splitlines()
#
# f = open('config\\Subjects_list2.txt')
# subjects = f.read().splitlines()

# subjects = ['Yokoyama']
# for subject in tqdm(subjects):
#     for name in tqdm(names):
#         class1 = Segment_data(subject, name)
#         class1.load_SP_list()
#         class1.cut_and_spline(70)
# print('-----end')

#2ND RUN HERE TO PRINT END
subject = 'Takuya'
name = 'notouch'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(150) #this is the number of frames for each cutted peak data (1 excel file)
print('-----end')

'''
for i in range(0, 80):
    class1.calc_distance(i, [])
    # class1.plot_raw_data()
    class1.plot_raw_distance()
'''
# class1.plot_temp('train\\20170620\\data\\', 'TurnInside', 7, 7)

self = class1


subject = 'Takuya'
frames = 80
name = 'notouch'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

name = 'gentlestroke'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

name = 'poke'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

name = 'press'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

name = 'scratch'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

frames = 100
name = 'notouch'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

name = 'gentlestroke'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

name = 'poke'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

name = 'press'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)

name = 'scratch'
class1 = Segment_data(subject, name)
class1.load_SP_list()
class1.cut_and_spline(frames) #this is the number of frames for each cutted peak data (1 excel file)
print('-----end')