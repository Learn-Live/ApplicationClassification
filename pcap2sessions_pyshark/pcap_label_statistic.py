# -*- coding: utf-8 -*-
"""
    find the url in pcap.

    out_put: five_tuple + url
-------------------
flow format: txt to flow
--------------------
0.000429 238.227.214.84 147.47.105.20 TCP 54 0.014504
<start time stamp> <src ip> <dst ip> <protocol> <flow size> <flow duration>
"""

from __future__ import print_function, division

import os
import time
from subprocess import check_call

import matplotlib.pyplot as plt

from scapy.layers.inet import IP
from scapy.main import load_layer
from scapy.utils import PcapReader


def save_to_file(output_file, input_f, res_dict):
    with open(output_file, 'a') as out:
        out.write(os.path.basename(input_f))
        for idx, (key, value) in enumerate(res_dict.items()):
            out.write("---" + str(value) + '\n')

    return output_file


def extract_url_from_pcap(input_f, output_file):
    st = time.time()
    print('process ... \'%s\'' % input_f, flush=True)
    # Step 1. read from pcap and do not return a list of packets
    try:

        load_layer("tls")  # enable scapy can parse tls
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

    res_dict = {}
    while True:
        pkt = myreader.read_packet()
        if pkt is None:
            break
        cnt = 0
        # step 1. parse "Ethernet" firstly
        if pkt.name == "Ethernet":
            if pkt.payload.name.upper() in ['IP', 'IPV4']:
                if pkt.payload.payload.name.upper() in ["TCP", "UDP"]:
                    if pkt.payload.payload.payload.name.upper() in 'TLS' and "Client Hello" in \
                            pkt.payload.payload.payload.msg[0].name:
                        print('packet[0] info: "%s:%d-%s:%d-%s"+%s' % (
                            pkt.payload.src, pkt.payload.payload.sport, pkt.payload.dst, pkt.payload.payload.dport,
                            pkt.payload.payload.name, pkt.payload.payload.payload.name))
                        five_tuple = pkt.payload.src + ':' + str(
                            pkt.payload.payload.sport) + '-' + pkt.payload.dst + ':' + str(
                            pkt.payload.payload.dport) + '-' + pkt.payload.payload.name.upper()
                        url_str = pkt.payload.payload.payload.msg[0].ext[00].fields['servernames'][0].fields[
                            'servername']
                        if five_tuple not in res_dict.keys():
                            res_dict[five_tuple] = url_str
                        else:
                            print(f'\'{url_str}\' appears')
                            res_dict[five_tuple] = url_str
        else:  # step 2. if this pkt can not be recognized as "Ethernet", then try to parse it as (IP,IPv4)
            pkt = IP(pkt)  # without ethernet header,  then try to parse it as (IP,IPv4)
            if pkt.payload.name.upper() in ["TCP", "UDP"]:
                if "Client Hello" in pkt.payload.payload.msg[0].name:
                    print('packet[0] info: "%s:%d-%s:%d-%s"+%s' % (
                        pkt.src, pkt.payload.sport, pkt.dst, pkt.payload.dport,
                        pkt.payload.name, pkt.payload.payload.name))
                    five_tuple = pkt.payload.src + ':' + str(
                        pkt.payload.payload.sport) + '-' + pkt.payload.dst + ':' + str(
                        pkt.payload.payload.dport) + '-' + pkt.payload.payload.name.upper()
                    pkt.payload.payload.msg[0].ext[00].fields['servernames'][0].fields['servername']
                    if five_tuple not in res_dict.keys():
                        res_dict[five_tuple] = url_str
                    else:
                        print(f'\'{url_str}\' appears')
                        res_dict[five_tuple] = url_str

    save_to_file(output_file, input_f, res_dict)


if __name__ == '__main__':
    root_dir = '../results'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    pcap_root_dir = '../input_data'
    os.listdir()
    file_lst = ['10.10.5.170_60219_52.202.180.164_443_SSL.Amazon.pcap',
                '10.10.5.170_60219_52.202.180.164_443_SSL.Amazon.pcap']

    output_file = os.path.join(root_dir, 'out_result.txt')
    if os.path.exists(output_file):
        os.remove(output_file)
    for file_tmp in file_lst:
        pcap_file_name = os.path.join(pcap_root_dir, file_tmp)
        print(pcap_file_name)
        # file_name_prefix = os.path.basename(pcap_file_name)
        extract_url_from_pcap(pcap_file_name, output_file)
