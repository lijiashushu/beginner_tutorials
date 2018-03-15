#!/usr/bin/env python
# coding=utf-8
import socket
import struct
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('10.1.1.20', 7777))
    data1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    while True:
        data,addr = s.recvfrom(1024)
        if not data:
            print("client hasn't exist")
            break
        dcd_data = struct.unpack('7H',data)
        for i in range(0, 7):
            data1[i] = (dcd_data[i] - 32678.0)/1000.0
        print data1

    s.close()

if __name__ == "__main__":
    main()

