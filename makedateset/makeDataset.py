from __future__ import print_function
from collections import OrderedDict
from pcap2pkts import pcap2packets, pcap2sessions_statistic_with_pcapreader_scapy_improved
import tensorflow as tf
import numpy as np
from PIL import Image
import os, sys
import csv
import random
secssion_staticli= []
cnt = 0; delCnt = 0; solvCnt = 0;
testMat = [];
header2time = {}
time2header = {}
headerSet = set()
packetinfo = []#10* 16 = 160 list, every list is a list of packet number
packetthisget = [] # list size of 16, recording whether this session get i*100 length.
time2payload = OrderedDict()
timeInterval = 1000000 #seems like the max packets can be held in the buffer system
threshold = 20 # remember to change another one
upperbound = 10220
upperbound *= 2
lowerbound = 0
trainWriter = tf.python_io.TFRecordWriter('./train_1500.tfrecords')
testWriter = tf.python_io.TFRecordWriter('./test_1500.tfrecords')
csvtrain = np.zeros((1,2))
csvtest= np.zeros((1,2))

'''
files = [['SSL.Amazon.1','SSL.Amazon.2','SSL.Amazon.3','SSL.Amazon.4','SSL.Amazon.5','SSL.Amazon.6'],
         ['SSL.Cloudflare.1','SSL.Cloudflare.2','SSL.Cloudflare.3','SSL.Cloudflare.4','SSL.Cloudflare.5','SSL.Cloudflare.6'],
         ['SSL.Facebook.1','SSL.Facebook.2','SSL.Facebook.3','SSL.Facebook.4','SSL.Facebook.5','SSL.Facebook.6','SSL.Facebook.7','SSL.Facebook.8','SSL.Facebook.9'],
         ['SSL.Github.1','SSL.Github.2','SSL.Github.3','SSL.Github.4','SSL.Github.5','SSL.Github.6','SSL.Github.7','SSL.Github.8','SSL.Github.9','SSL.Github.10','SSL.Github.11','SSL.Github.12','SSL.Github.13','SSL.Github.14','SSL.Github.15','SSL.Github.16','SSL.Github.17','SSL.Github.18','SSL.Github.19'],
         ['SSL.Google.1','SSL.Google.2','SSL.Google.3','SSL.Google.4','SSL.Google.5','SSL.Google.6','SSL.Google.7','SSL.Google.8','SSL.Google.9'],
         ['SSL.Microsoft.1','SSL.Microsoft.2','SSL.Microsoft.3','SSL.Microsoft.4','SSL.Microsoft.5','SSL.Microsoft.6','SSL.Microsoft.7'],
         ['SSL.Skype.1','SSL.Skype.2','SSL.Skype.3','SSL.Skype.4','SSL.Skype.5'],
         ['SSL.Twitter.1','SSL.Twitter.2','SSL.Twitter.3','SSL.Twitter.4','SSL.Twitter.5','SSL.Twitter.6','SSL.Twitter.7','SSL.Twitter.8','SSL.Twitter.9','SSL.Twitter.10','SSL.Twitter.11','SSL.Twitter.12'],
         ['SSL.YouTube.1','SSL.YouTube.2','SSL.YouTube.3','SSL.YouTube.4','SSL.YouTube.5'],
         ['SSL.Slack.1','SSL.Slack.2','SSL.Slack.3','SSL.Slack.4','SSL.Slack.5','SSL.Slack.6','SSL.Slack.7','SSL.Slack.8','SSL.Slack.9']]
'''
#
dic10 = {0:'SSL.Amazon',1:'SSL.Cloudflare',2:'SSL.Facebook',3:'SSL.Github',4:'SSL.Google',5:'SSL.Microsoft',6:'SSL.Skype',7:'SSL.Twitter',8:'SSL.YouTube',9:'SSL.Slack'}
#generate files list:
files = []
for i in range(0,10):
    appname = dic10[i]
    #try:
    file_list=os.listdir("/scratch/lx643/makedata/10app/"+appname)
    files.append(file_list)
    #except:
    #    pass

def init():

    global cnt, delCnt, solvCnt, timeInterval, threshold,packetinfo
    global header2time, time2header, time2payload, headerSet, trainWriter, testWriter
    cnt = delCnt = solvCnt = 0;
    for i in range(0,10):
        appli = []
        for j in range(0,int(upperbound/100)+1):
            appli.append([])
        packetinfo.append(appli)
    for i in range(0,int(upperbound/100)+1):
        packetthisget.append(False)
    print ("timeInterval:",timeInterval)
    header2time.clear(); time2header.clear(); time2payload.clear(); headerSet.clear();

def emptyDict():# once per pcap file, flush all dics
    global header2time, time2header, time2payload, headerSet
    header2time = {}; time2header = {};
    time2payload = OrderedDict(); headerSet = set();
    
def getMatrixfrom_pcap(hexst, width):
    temp = []
    for i in range(0, len(hexst), 2):
        if hexst[i:i+2] == "xx":
            #temp.append(-1)
            pass
        else:
            temp.append(int(hexst[i:i+2], 16))
    #fh = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])
    #print(temp)
    fh = np.array(temp)
    width = int(width/2)
    if fh.shape[0] > width:
        fh = fh[:width] 
    if fh.shape[0] < width:
        fh = np.pad(fh, (0, width-len(fh)), 'constant')
    totallength = fh.shape[0]
    fh = np.reshape(fh, (-1, width))
    fh = np.uint8(fh)
    return fh, totallength

def formData(header, payloadList, label, session_max_packets,isTrain=True):
    global testMat, trainWriter, testWriter, csvtrain, csvtest
    payload = payloadList
    data, totallength = getMatrixfrom_pcap(payload, upperbound)
    secssion_staticli.append((label,session_max_packets,totallength))
    print("dataformed........................................................",(label,session_max_packets,totallength))
    data_raw = data.tobytes()
    example = tf.train.Example(
       features=tf.train.Features(feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'data_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_raw]))
       }))
    print("data...",data)
    if isTrain :
        trainWriter.write(example.SerializeToString())
        tremp=np.array([[label,data]],object)
        csvtrain=np.vstack((csvtrain,tremp))
    else :
        testWriter.write(example.SerializeToString())
        tremp=np.array([[label,data]],object)
        csvtest=np.vstack((csvtest,tremp))

def handlePkt(pkt, label,session_max_packets,isFinal): #called n time by one session
    global cnt, delCnt, solvCnt, timeInterval, upperbound
    global header2time, time2header, time2payload,headerSet, packetthisget,packetinfo 
    header, payload = pkt
    if payload[0:4]  =="1703":
        payload = "0000000000"+ payload[10:]
        #payload = payload[6:10]
    #else:
    #    return ;
    if payload[0:4] == "1403" or payload[0:4] == "1503" or payload[0:4] == "1603":
        return ;
    #payload = payload[2920:]
    #payload = payload + "xx" #showing the packet is ending
    #print("payload:..",payload)
    if header in headerSet: #print("packet auto discard for the whole number is upper than 1500......")
        return ;
    cnt += 1
    # update info and classify it
    if header not in header2time:   #this is the true buffer system. The socalled "time" is packet number
        header2time[header] = cnt
        time2header[cnt] = header
        time2payload[cnt] = ""
    tPrev = header2time[header]
    time2payload[tPrev] += payload #now payload is added in
    '''
    print("tPrev.......................",tPrev)
    print("cnt..............",cnt)
    print("header2time.....,,,,,,,,,,,,,,,,,,,,,,,.",header2time)
    print("time2payload[tPrev].......",(time2payload[tPrev]))
    print("len of time2payload[tPrev]./................",len(time2payload[tPrev]) )
    '''
    for i in range(0,int(upperbound/100)+1):
        #print("packet.false............................",packetthisget[i])
        if (packetthisget[i]== False and len(time2payload[tPrev]) > i*100):
            packetthisget[i] = True
            packetinfo[label][i].append(cnt)
    if len(time2payload[tPrev]) > upperbound or (isFinal == True and len(time2payload[tPrev]) > lowerbound ):
        headerSet.add(header)
        solvCnt += 1
        isTrain = True
        randomnumber = random.randint(0, 9)
        if randomnumber == -1: #general all data in one file
            isTrain = False
        formData(header, time2payload[tPrev], label,session_max_packets, isTrain=isTrain)
        del header2time[header]
        del time2header[tPrev]; del time2payload[tPrev]
    else :
        if tPrev != cnt:
            header2time[header] = cnt
            time2header[cnt] = header
            time2payload[cnt] = time2payload[tPrev]
            del time2header[tPrev]; del time2payload[tPrev]
    # delete info that is too old. here use packet numbers as thre
    for preTime, info in time2payload.items():
        if preTime < cnt - timeInterval:
            delCnt += 1
            preHeader = time2header[preTime]
            del header2time[preHeader]
            del time2header[preTime]
            del time2payload[preTime]
        break

def getLabel(targetFile):
    clss = -1
    for i in range(0,10):
        for fileName in files[i]:
            if targetFile == fileName:
                clss = i;
                break
        if clss != -1: break ;
    return clss

def main():
    global trainWriter, testWriter, csvtrain, csvtest,packetinfo,packetthisget,cnt
    init()
    dataFile = 0;
    counter = 0
    #for xl in range(0,10):
        #for fileName in os.listdir("/scratch/lx643/makedata/10app/"+dic10[xl]):
    with open("merged.txt","r") as rappfile:
        rapp = rappfile.read()
        rappli= rapp.split("\n")
        for rappitem in rappli:
            cateName = rappitem.split("\t")[0]
            fileName = rappitem.split("\t")[1]
            label = int(rappitem.split("\t")[3])
            counter += 1
            dataFile += 1
            input_file = os.path.join("/scratch/lx643/makedata/10app/"+cateName, fileName)
            #input_file = os.path.join("/home/xulu/test10app"+"/"+dic10[xl], fileName)
            #label = getLabel(fileName)
            if label == -1:
                print ("File %s has no label-----------------------------------------\n" % (fileName))
                continue
            #print ("Processing %s with label %d" % (fileName, label))
            sys.stdout.flush()
            all_stats_dict, pkts_dict, pkts_list, session_packets, session_max_packets,pkt_sizel = pcap2sessions_statistic_with_pcapreader_scapy_improved(input_file)
            for i in range(len(pkts_list)):#one packet a time
                if i == len(pkts_list) -1 :
                    handlePkt(pkts_list[i], label, session_packets,True)
                else:
                    handlePkt(pkts_list[i], label, session_packets,False)
            for i in range(0,len(packetthisget)):
                packetthisget[i] = False
            if (len(pkts_list) > 0):
                with open("pkt_size.txt", "a") as wpszfile:
                    wpszfile.write("{0}\t{1}\n".format(pkt_sizel,label))
            cnt = 0
            emptyDict()
            sys.stdout.flush()
    print(secssion_staticli)
    trainWriter.close(); testWriter.close();
    np.save('./newapp_10220_p',csvtrain)
    np.save('./tstdata',csvtest)
    avgpacketnum = []
    print(packetinfo)
    for i in range(0,10):
        appli = []
        for j in range(0,int(upperbound/100)+1):
            if (len(packetinfo[i][j]) != 0):
                appli.append(sum(packetinfo[i][j]) / float(len(packetinfo[i][j])))
            else:
                appli.append(0)
        avgpacketnum.append(appli)
    print(avgpacketnum )
    with open("thelist.txt","w") as wfile:
        wfile.write("{0}".format(avgpacketnum))
if __name__ == '__main__':
    main()
