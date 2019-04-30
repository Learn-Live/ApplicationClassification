"""
    Purpose:
        transform 'pcap or pcapng' to 'packets' (pkts) with scapy.

    Requirements:
        scapy
        python 3.x

    Created time:
        2018.10.10
    Version:
        0.0.1
    Author:
"""
from __future__ import print_function
import numpy as np
from scapy.all import rdpcap, PcapReader
from scapy.layers.inet import IP
import os, sys, time
from collections import OrderedDict
import time


#filterIP = ['0.0.0.0','255.255.255.255','224.0.0.252','131.202.243.255']
#filterPort = [53, 5355, 5353, 1900, 161, 137, 138, 123, 67, 68, 3478]

# filterIP = ['0.0.0.0','255.255.255.255','224.0.0.252']
# filterPort = [53, 67, 68, 137, 138, 123, 161]

filterIP = []
filterPort = []

# 'IPs':['0.0.0.0','255.255.255.255','224.0.0.252','131.202.243.255'],'ports':[53,5355,5353,1900,161,137,138,123,67,68,3478]

def getKeyValue(innerPkt):
    pass


def save_session_to_dict(k='five_tuple', v='pkt', sess_dict=OrderedDict()):
    if k not in sess_dict.keys():
        sess_dict[k] = []
    sess_dict[k].append(v)
    '''
    # swap src and dst
    tmp_lst = k.split('-')
    k_dst2src = tmp_lst[1] + '-' + tmp_lst[0] + '-' + tmp_lst[-1]
    if k_src2dst not in sess_dict.keys() and k_dst2src not in sess_dict.keys():
        sess_dict[k] = []
    if k_src2dst in sess_dict.keys():
        sess_dict[k].append(v)
    else:
        sess_dict[k_dst2src].append(v)
    '''

def count_protocls(sess_dict):
    """
        get TCP and UDP distribution
    :param sess_dict:
    :return:
    """
    res_dict = {'TCP': 0, 'UDP': 0}
    prtls_lst = []
    for key in sess_dict.keys():
        prtl = key.split('-')[-1]
        if prtl not in res_dict.keys():
            res_dict[prtl] = 1
        else:
            res_dict[prtl] += 1
        prtls_lst.append(prtl)

    return res_dict

def getPktList(sess_dict,ptimelist):
    pkt_list = [];
    pkt_size = [];

    for k, v in sess_dict.items():
        appbegin = False
        index = -1
        accumulatedTime = 0
        for pkt in v:
            index += 1
            
            if pkt.name == "Ethernet":                    
                thisPacketTime = ptimelist[index]
                pktlen= int(len(pkt.payload.payload.payload.original.hex())/2)
                if pkt.payload.payload.payload.original.hex()[0:2] == "17":
                    appbegin = True
                if appbegin == True and pktlen  > 6 and pktlen != 31:
                    P = pkt.payload.payload.payload.original.hex()
                    Htemp = pkt.payload.original.hex()
                    H = Htemp[0:24] + Htemp[48: len(Htemp) - len(P)]
                    thisPacketTime = ptimelist[index]
                    timeInterval = 0
                    if accumulatedTime == 0:
                        accumulatedTime = thisPacketTime
                    else:
                        timeInterval = thisPacketTime - accumulatedTime
                        accumulatedTime = thisPacketTime
                    timeInterval = timeInterval* 1000000
                    timeInterval = round(timeInterval % 100000000)
                    T = str(timeInterval).zfill(8)
                    '''
                    if len(P) < 2920:
                    	P = P+("0"*(2920 - len(P)))
                    else:
                    	P = P[0:2920]
                    '''
                    pkt_list.append([k, P])
                    pkt_size.append(pktlen)
            else:
                thisPacketTime = ptimelist[index]
                pktlen= int(len(pkt.payload.payload.original.hex())/2)
                if pkt.payload.payload.original.hex()[0:2] == "17":
                    appbegin = True
                if appbegin == True and pktlen  > 6 and pktlen != 31:
                    P = pkt.payload.payload.original.hex()
                    Htemp = pkt.original.hex()
                    H = Htemp[0:24] + Htemp[48: len(Htemp) - len(P)]
                    thisPacketTime = ptimelist[index]
                    timeInterval = 0
                    if accumulatedTime == 0:
                        accumulatedTime = thisPacketTime
                    else:
                        timeInterval = thisPacketTime - accumulatedTime
                        accumulatedTime = thisPacketTime
                    timeInterval = timeInterval* 1000000
                    timeInterval = round(timeInterval % 100000000)
                    T = str(timeInterval).zfill(8)
                    '''
                    if len(P) < 2920:
                    	P = P+("0"*(2920 - len(P)))
                    else:
                    	P = P[0:2920]
                    '''
                    pkt_list.append([k, P])
                    pkt_size.append(pktlen)
    return pkt_list,pkt_size


def count_sess_size(sess_dict):
    """
        get each sess size (sum of pkts_len in this sess), not flow.
    :param sess_dict:
    :return:
    """
    res_dict = {'TCP': [], 'UDP': []}
    for key in sess_dict.keys():
        prtl = key.split('-')[-1]
        if prtl not in res_dict.keys():
            res_dict[prtl] = [sum([len(p) for p in sess_dict[key]])]
        else:
            res_dict[prtl].append(sum([len(p) for p in sess_dict[key]]))
    return res_dict

def pcap2sessions_statistic_with_pcapreader_scapy_improved(input_f):
    """
        achieve the statistic of full sessions in pcap after removing uncompleted TCP sessions
        There is no process on UDP sessions
        Improved version : just use one for loop
        Note:
            pkts_lst = rdpcap(input_f)  # this will read all packets in memory at once.
            changed  to :
            There are 2 classes:
                PcapReader - decodes all packets immediately
                RawPcapReader - does not decode packets
                Both of them have iterator interface (which I fixed in latest commit). So you can write in your case:
                with PcapReader('file.pcap') as pr:
                  for p in pr:
                    ...do something with a packet p...
            reference:
                https://github.com/phaethon/kamene/issues/7
        flags in scapy
         flags = {
        'F': 'FIN',
        'S': 'SYN',
        'R': 'RST',
        'P': 'PSH',
        'A': 'ACK',
        'U': 'URG',
        'E': 'ECE',
        'C': 'CWR',
    }
    :param input_f:
    :return:
    """
    st = time.time()
    print('process ... \'%s\'' % input_f)
    # Step 1. read from pcap and do not return a list of packets
    try:
        # pkts_lst = rdpcap(input_f)  # this will read all packets in memory at once, please don't use it directly.
        # input_f  = '../1_pcaps_data/vpn_hangouts_audio2.pcap'  #
        # input_f = '/home/kun/PycharmProjects/Pcap2Sessions_Scapy/1_pcaps_data/aim_chat_3a.pcap'  #
        myreader = PcapReader(input_f)  # iterator, please use it to process large file, such as more than 4 GB
    except MemoryError as me:
        print('memory error ', me)
        return -1
    except FileNotFoundError as fnfe:
        print('file not found ', fnfe)
        return -2
    except:
        print('other exceptions')
        return -10

    # Step 2. achieve all the session in pcap.
    # data.stats
    pkts_stats = {'non_Ether_IPv4_pkts': 0, 'non_IPv4_pkts': 0, 'non_TCP_UDP_pkts': 0, 'TCP_pkts': 0,
                  'UDP_pkts': 0}
    cnt = 0
    sess_dict = OrderedDict()
    first_print_flg = True
    max_pkts_cnt = 1
    ptimelist = []
    while True:
        pkt = myreader.read_packet()
        
        if pkt is None:
            break
        ptimelist.append(pkt.time)
        if max_pkts_cnt >= 30000:
            print('\'%s\' includes more than %d packets and in this time just process the first %d packets. Please split it firstly and do again.' % (input_f, max_pkts_cnt,max_pkts_cnt))
            break
        max_pkts_cnt += 1
        # step 1. parse "Ethernet" firstly
        if pkt.name == "Ethernet":
            if first_print_flg:
                first_print_flg = False
                print('\'%s\' encapsulated by "Ethernet Header" directly' % input_f)
            if pkt.payload.name.upper() in ['IP', 'IPV4']:
                if pkt.payload.payload.name.upper() in ["TCP", "UDP"]:
                    # if cnt == 0:
                    #     print('packet[0] info: "%s:%d-%s:%d-%s"+%s' % (
                    #         pkt.payload.src, pkt.payload.payload.sport, pkt.payload.dst, pkt.payload.payload.dport,
                    #         pkt.payload.payload.name, pkt.payload.payload.payload))
                    # fix here by Site Li
                    src = pkt.payload.src; sport = pkt.payload.payload.sport;
                    dst = pkt.payload.dst; dport = pkt.payload.payload.dport;
                    if src > dst:# to generate full section
                        src, dst = dst, src
                        sport, dport = dport, sport
                    if src in filterIP or dst in filterIP or sport in filterPort or dport in filterPort:# banned ip
                        continue
                    five_tuple = src + ':' + str(sport) + '-' + dst + ':' + str(dport) + '-' + pkt.payload.payload.name.upper()
                    # save_session_to_dict(k=five_tuple, v=pkt,sess_dict=sess_dict)

                    save_session_to_dict(k=five_tuple, v=pkt.payload, sess_dict=sess_dict)  # only save Ethernet payload to sess_dict
                    cnt += 1
                    # pkts_lst.append(pkt.payload)  # only include "IPv4+IPv4_payload"
                    if pkt.payload.payload.name.upper() == "TCP":
                        pkts_stats['TCP_pkts'] += 1
                    else:
                        pkts_stats['UDP_pkts'] += 1
                else:
                    pkts_stats['non_TCP_UDP_pkts'] += 1
                    # pkts_stats['IPv4_pkts'] += 1
            else:
                pkts_stats['non_IPv4_pkts'] += 1
        else:  # step 2. if this pkt can not be recognized as "Ethernet", then try to parse it as (IP,IPv4)
            pkt = IP(pkt)  # without ethernet header,  then try to parse it as (IP,IPv4)
            if first_print_flg:
                first_print_flg = False
                print('\'%s\' encapsulated by "IP Header" directly, without "Ethernet Header"'%input_f)
            if pkt.name.upper() in ['IP', 'IPV4']:
                if pkt.payload.name.upper() in ["TCP", "UDP"]:
                    # if cnt == 0:
                    #     print('packet[0] info: "%s:%d-%s:%d-%s"+%s' % (
                    #         pkt.src, pkt.payload.sport, pkt.dst, pkt.payload.dport,
                    #         pkt.payload.name, pkt.payload.payload))
                    # fix here by Site Li
                    src = pkt.src; sport = pkt.payload.sport;
                    dst = pkt.dst; dport = pkt.payload.dport;
                    if src > dst:
                        src, dst = dst, src
                        sport, dport = dport, sport
                    if src in filterIP or dst in filterIP or sport in filterPort or dport in filterPort:
                        continue
                    five_tuple = src + ':' + str(sport) + '-' + dst + ':' + str(dport) + '-' + pkt.payload.name.upper()
                    save_session_to_dict(k=five_tuple, v=pkt, sess_dict=sess_dict)
                    cnt += 1
                    # pkts_lst.append(pkt.payload)  # only include "IPv4+IPv4_payload"
                    if pkt.payload.name.upper() == "TCP":
                        pkts_stats['TCP_pkts'] += 1
                    else:
                        pkts_stats['UDP_pkts'] += 1
                else:
                    pkts_stats['non_TCP_UDP_pkts'] += 1
                    # pkts_stats['IPv4_pkts'] += 1
            else:
                pkts_stats['non_IPv4_pkts'] += 1
                # print('unknown packets type!',pkt.name)
                pkts_stats['non_Ether_IPv4_pkts'] += 1

    # data.stats
    # print('%s info is %s' % (input_f, pkts_lst))
    print('packet info:"srcIP:srcPort-dstIP:dstPort-prtcl" + IP_payload')
    
    # Step 3. achieve all full session in sess_dict.
    full_sess_dict = OrderedDict()
    for k, v in sess_dict.items():   # all pkts in sess_dict without Ethernet headers and tails
        prtl = k.split('-')[-1]
        if prtl == "TCP":
            """
                only save the first full session in v (maybe there are more than one full session in v)
            """
            tcp_sess_list = []
            full_session_flg = False
            i = -1
            TCP_start_flg = False
            # if k == '131.202.240.87:64716-131.202.6.26:13111-TCP':
            #     print(f'k={k}, line_bytes={v}')
            while i < len(v):
            # # for pkt in v:
                i += 1
                if i >= len(v):
                    break
                if len(v) < 5:  # tcp start (3 packets) + tcp finish (at least 2 packets)
                    print('%s not full session, it only has %d packets' % (k, len(v)))
                    break
                S = str(v[i].payload.fields['flags'])
                # step 1. discern the begin of TCP session.
                if 'S' in S:
                    if 'A' not in S:  # the first SYN packet in TCP session.
                        # if flags[S] == "SYN":
                        TCP_start_flg = True
                        tcp_sess_list.append(v[i])
                        continue    # cannot ignore
                    else:  # the second SYN + ACK
                        tcp_sess_list.append(v[i])
                    continue
                # step 2. discern the transmitted data of TCP session
                if TCP_start_flg:  # TCP data transform.
                    while i < len(v):
                    # for pkt_t in v[i:]:
                        tcp_sess_list.append(v[i])
                        F = str(v[i].payload.fields['flags'])
                        if 'F' in F:  # if  flags[F]== "FIN":
                            full_session_flg = True
                        # step 3. discern the finish of TCP session.
                        if 'S' in str(v[i].payload.fields['flags']) and len(
                                tcp_sess_list) >= 5:  # the second session
                            print('the second session begins.')
                            break
                        i += 1
                else:
                    pass
                if full_session_flg:
                    full_sess_dict[k] = tcp_sess_list
                    break
        elif prtl == "UDP":
            full_sess_dict[k] = v  # do not do any process for UDP session.
        else:
            pass
    print('pkts_stats is ', pkts_stats)
    print('Number of sessions(TCP/UDP) in %s is %d, number of full session(TCP/UDP) is %d' % (
        input_f, len(sess_dict.keys()), len(full_sess_dict.keys())))
    print('all_sess_dict:', count_protocls(sess_dict), '\nfull_sess_dict:', count_protocls(full_sess_dict))

    all_stats_dict = OrderedDict()
    all_stats_dict['pkts_stats'] = pkts_stats
    all_stats_dict['all_sess'] = count_protocls(sess_dict)
    all_stats_dict['full_sess'] = count_protocls(full_sess_dict)
    all_stats_dict['full_sess_size_distribution'] = count_sess_size(full_sess_dict)

    # print(all_stats_dict)
    pkt_list, pkt_size = getPktList(full_sess_dict,ptimelist)
   
    return all_stats_dict, full_sess_dict, pkt_list, cnt,max_pkts_cnt,pkt_size


def pcap2packets(input_file='.pcap or pcapng', retDict=True):
    """
        "transform pcap to packets"
    :param input_file: pcap or pcapng
    :return: a list of packets.
    """
    pkts_dict = {}
    pkts_list = []
    try:
        myreader = PcapReader(input_file)
    except MemoryError as me:
        print('memory error ', me)
        return -1
    except FileNotFoundError as fnfe:
        print('file not found ', fnfe)
        return -2
    except:
        print('other exceptions')
        return -10
    # data = rdpcap(input_file)
    
    # print('%s info is ' % data)
    ab_pkts = {'non_Ether_pkts': 0, 'non_IPv4_pkts': 0, 'non_TCP_UDP_pkts': 0}
    print('packet info:"srcIP:srcPort-dstIP:dstPort-prtcl" + IP_payload')
    cnt = 0
    # for pkt in data:
    while True:
        pkt = myreader.read_packet()
        if pkt is None:
            break
        if pkt.name == "Ethernet":
            if pkt.payload.name.upper() in ['IP', 'IPV4']:
                if pkt.payload.payload.name.upper() in ["TCP", "UDP"]:
                # if pkt.payload.payload.name.upper() in ["TCP"]:
                    src = pkt.payload.src; sport = pkt.payload.payload.sport;
                    dst = pkt.payload.dst; dport = pkt.payload.payload.dport;
                    if src > dst:
                        src, dst = dst, src
                        sport, dport = dport, sport
                    if src in filterIP or dst in filterIP or sport in filterPort or dport in filterPort:
                        continue
                    name = pkt.payload.payload.name; payload = pkt.payload.payload.payload.original.hex();
                    curKey = (src, sport, dst, dport, name)
                    # update dict
                    if curKey not in pkts_dict:
                        pkts_dict[curKey] = list()
                    pkts_dict[curKey].append(payload)
                    # update list
                    pkts_list.append([curKey, payload])
                    cnt += 1
                else:
                    ab_pkts['non_TCP_UDP_pkts'] += 1
            else:
                ab_pkts['non_IPv4_pkts'] += 1
        # handle non Ether pkts
        elif pkt.name.upper() in ['IP', 'IPV4']:
            if pkt.payload.name.upper() in ["TCP", "UDP"]:
            # if pkt.payload.name.upper() in ["TCP"]:
                src = pkt.src; sport = pkt.payload.sport;
                dst = pkt.dst; dport = pkt.payload.dport;
                if src > dst:
                    src, dst = dst, src
                    sport, dport = dport, sport
                if src in filterIP or dst in filterIP or sport in filterPort or dport in filterPort:
                    continue
                name = pkt.payload.name; payload = pkt.payload.payload.original.hex();
                curKey = (src, sport, dst, dport, name)
                # update dict
                if curKey not in pkts_dict:
                    pkts_dict[curKey] = list()
                pkts_dict[curKey].append(payload)
                # update list
                pkts_list.append([curKey, payload])
                cnt += 1
            else:
                ab_pkts['non_TCP_UDP_pkts'] += 1
        else:
            ab_pkts['non_Ether_pkts'] += 1
    print('Number of packets in %s is %d.' % (str(input_file), cnt))
    print('Abnormal packets in %s is %s' % (str(input_file), np.array(ab_pkts.values()).sum()))
    if retDict :
        return pkts_dict
    print (len(pkts_list))
    return pkts_list

if __name__ == '__main__':
    input_file = '/scratch/sl6890/1_Pcap/vpn_skype_chat1a.pcap'
    pkts_dict = pcap2packets(input_file)
    cnt = 0
    for key, value in pkts_dict.iteritems():
        print (key)
        curLen = 0
        for x in value:
            curLen += len(x)
        print ("the length is %d" % curLen)
        cnt += 1
    print (cnt)
