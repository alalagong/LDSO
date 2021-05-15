#!/bin/bash
#usage: python xxx.py file_name
dataname="01"
# dataname="01"
# dataname="02"
# dataname="03"
# dataname="04"
# dataname="05"
# dataname="06"
# dataname="07"
# dataname="08"
# dataname="09"
# dataname="10"
    # run dso
     ./bin/run_dso_kitti \
 	  files=/media/gong/win_file/Dataset/KITTI/odometry/dataset/sequences/${dataname}/ \
 	  calib=./examples/Kitti/Kitti00-02.txt
