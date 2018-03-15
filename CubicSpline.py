#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
class cubicSpline(object):
    def __init__(self):
        self.outputsize = 501
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.f = []
        self.bt = []
        self.gm = []
        self.h = []
        self.x_sample = []
        self.y_sample = []
        self.M = []
        self.sample_count = 0
        self.bound1 = 0
        self.bound2 = 0
        self.result = []

    def initParam(self, count):
        self.a = np.zeros(count, dtype="double")
        self.b = np.zeros(count, dtype="double")
        self.c = np.zeros(count, dtype="double")
        self.d = np.zeros(count, dtype="double")
        self.f = np.zeros(count, dtype="double")
        self.bt = np.zeros(count, dtype="double")
        self.gm = np.zeros(count, dtype="double")
        self.h = np.zeros(count, dtype="double")
        self.M = np.zeros(count, dtype="double")


    def loadData(self, x_data, y_data, count, bound1, bound2):
        if len(x_data) == 0 or len(y_data) == 0 or count < 3:
            return False

        self.initParam(count)

        self.x_sample = x_data
        self.y_sample = y_data
        self.sample_count = count
        self.bound1 = bound1
        self.bound2 = bound2

    def spline(self):
        f1 = self.bound1
        f2 = self.bound2

        for i in range(0, self.sample_count):
            self.b[i] = 2
        for i in range(0, self.sample_count - 1):
            self.h[i] = self.x_sample[i+1] - self.x_sample[i]
        for i in range(0, self.sample_count - 1):
            self.a[i] = self.h[i-1]/(self.h[i-1] + self.h[i])
        self.a[self.sample_count - 1] = 1

        self.c[0] = 1
        for i in range(1, self.sample_count-1):
            self.c[i] = self.h[i] / (self.h[i-1] + self.h[i])

        for i in range(0, self.sample_count-1):
            self.f[i] = (self.y_sample[i+1]-self.y_sample[i])/(self.x_sample[i+1]-self.x_sample[i])

        for i in range(1, self.sample_count-1):
            self.d[i] = 6*(self.f[i]-self.f[i-1])/(self.h[i-1]+self.h[i])

        """追赶法解方程"""
        self.d[0] = 6*(self.f[0] - f1)/self.h[0]
        self.d[self.sample_count - 1] = 6*(f2 - self.f[self.sample_count-2])/self.h[self.sample_count-2]

        self.bt[0] = self.c[0]/self.b[0]
        for i in range(1, self.sample_count-1):
            self.bt[i] = self.c[i]/(self.b[i] - self.a[i]*self.bt[i-1])

        self.gm[0] = self.d[0]/self.b[0]
        for i in range(1, self.sample_count):
            self.gm[i] = (self.d[i] - self.a[i]*self.gm[i-1])/(self.b[i]-self.a[i]*self.bt[i-1])
        self.M[self.sample_count-1] = self.gm[self.sample_count-1]
        temp = self.sample_count - 2
        for i in range(0, self.sample_count-1):
            self.M[temp] = self.gm[temp] - self.bt[temp]*self.M[temp+1]
            temp = temp - 1


    def getYbyX(self, x_in):
        klo = 0
        khi = self.sample_count - 1
        """二分法查找x所在的区间段"""
        while (khi - klo) >1:
            k = (khi+klo)/2
            if self.x_sample[k] > x_in:
                khi = k
            else:
                klo = k
        hh = self.x_sample[khi] - self.x_sample[klo]
        aa = (self.x_sample[khi] - x_in)/hh
        bb = (x_in - self.x_sample[klo])/hh

        y_out = aa * self.y_sample[klo] + bb * self.y_sample[khi] + \
                ((aa*aa*aa-aa)*self.M[klo] + (bb*bb*bb-bb)*self.M[khi])*hh*hh/6.0
        vel = self.M[khi] * (x_in - self.x_sample[klo]) * (x_in - self.x_sample[klo]) / (2 * hh)\
            - self.M[klo] * (self.x_sample[khi] - x_in) * (self.x_sample[khi] - x_in) / (2 * hh)\
            + (self.y_sample[khi] - self.y_sample[klo]) / hh - hh * (self.M[khi] - self.M[klo]) / 6

        return y_out, vel

    '''all_y_data 为得到角度后的转置，所以all_y_data[0]为第一个关节的所有的角度'''
    def caculate(self, all_x_data, all_y_data):
        length = len(all_x_data)
        dis = (all_x_data[length - 1] - all_x_data[0]) / (self.outputsize - 1)
        self.pos_result = np.zeros((self.outputsize, 7), dtype="double")
        self.vel_result = np.zeros((self.outputsize, 7), dtype="double")
        for ii in range(0, 7):
            self.loadData(all_x_data, all_y_data[ii], length, 0, 0)
            self.spline()
            x_out = -dis
            for i in range(0, self.outputsize):
                x_out = x_out + dis
                self.pos_result[i][ii], self.vel_result[i][ii] = self.getYbyX(x_out)



def main():
    o_x = [0,0.499875068670001,0.999899148950000,1.49996995926000,1.99996805191000,2.49999403954000,2.99991703034000,3.49990200997000,3.99996614458000,4.49994611738000,4.99992394448000]
    o_y = [[2.41601974092000,2.41601974092000,2.41410226493000,2.34507312948000,2.06128668372000,1.76791285804000,1.49409728740000,1.22296618314000,0.978296247474000,0.927291386277000,0.923839929504000],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [2.41601974092000, 2.41601974092000, 2.41410226493000, 2.34507312948000, 2.06128668372000, 1.76791285804000,
            1.49409728740000, 1.22296618314000, 0.978296247474000, 0.927291386277000, 0.923839929504000],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [2.41601974092000, 2.41601974092000, 2.41410226493000, 2.34507312948000, 2.06128668372000, 1.76791285804000,
            1.49409728740000, 1.22296618314000, 0.978296247474000, 0.927291386277000, 0.923839929504000]]
    test = cubicSpline()
    test.caculate(o_x, o_y)
    print "done"
if __name__ == '__main__':
    main()


