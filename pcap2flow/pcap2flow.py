# -*- coding: utf-8 -*-
"""
    @ Pcap Converter: convert pcap to text or flows.
    refer to:  https://github.com/hbhzwj/pcap-converter

txt format: pcap to txt
-------------------
1   0.000000 51.142.253.91 -> 15.236.229.88 TCP 54 2555 22746
<seq no> <time stamp> <src ip> -> <dst ip> <protocol> <pkt size> <src port> <dst port>

flow format: txt to flow
--------------------
0.000429 238.227.214.84 147.47.105.20 TCP 54 0.014504
<start time stamp> <src ip> <dst ip> <protocol> <flow size> <flow duration>
"""

from __future__ import print_function, division
from subprocess import check_call


# from CythonUtil import c_parse_records_tshark


def export_to_txt(f_name, txt_f_name):
    cmd = """tshark -o gui.column.format:'"No.", "%%m", "Time", "%%t", "Source", "%%s", "Destination", "%%d", "Protocol", "%%p", "len", "%%L", "srcport", "%%uS", "dstport", "%%uD"' -r %s > %s""" % (
        f_name, txt_f_name)

    print('--> ', cmd)
    check_call(cmd, shell=True)


def parse_records_tshark(f_name):
    records = []
    NAME = ['start_time', 'src_ip', 'dst_ip', 'protocol', 'length',
            'src_port', 'dst_port']
    with open(f_name, 'r') as infile:
        for line in infile:
            line = line.strip()
            items = line.split()
            rec = (float(items[1]), items[2], items[4], items[5], int(items[6]),
                   int(items[7]), int(items[8]))
            records.append(rec)
    return records, NAME


def change_to_flows(records, name, time_out):  # name=['start_time','src_ip',  'dst_ip',, 'protocol','length', 'src_port', 'dst_port']
    t_seq = name.index('start_time')
    length_seq = name.index('length')
    # five_tuple_seq = [name.index(k) for k in ['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol']]
    five_tuple_seq = [name.index(k) for k in ['src_ip', 'dst_ip', 'protocol','src_port', 'dst_port']]
    open_flows = dict()  # key: five_tuple, value:(start_time, end_time, length_lst, interval_time_diff_lst)
    res_flow = []
    for rec in records:
        # five_tuple = get_five_tuple(rec)
        five_tuple = tuple(rec[seq] for seq in five_tuple_seq)
        t = rec[t_seq]
        length = rec[length_seq]
        interval_time=0.0
        # check time out
        remove_flows = []
        for f_tuple, (st_time, last_time, fs, interval_diff) in open_flows.items():
            if t - last_time > time_out:  # time out : t-last_time
            # if t - st_time > time_out:  # flow_duration: t-start_time
                flow_duration = t - st_time
                res_flow.append((st_time,t) + f_tuple + (fs, flow_duration,interval_diff))
                print('f_tuple',f_tuple)
                remove_flows.append(f_tuple)
        for f_tuple in remove_flows:
            # print('---f_tuple:',five_tuple)
            del open_flows[f_tuple]

        stored_rec = open_flows.get(five_tuple)
        if stored_rec is not None:  # if already exists
            (st_time_old, last_time_old, fs_old, interval_old) = stored_rec
            # open_flows[five_tuple] = (st_time_old, t, fs_old + length)
            fs_old.append(length)   # return None
            interval_old.append(t - last_time_old)  # return None
            open_flows[five_tuple] = (st_time_old, t,fs_old , interval_old)
            # print(open_flows[five_tuple],fs_old.append(length),fs_old,length)
        else:  # not exisit
            open_flows[five_tuple] = (t, t, [length], [interval_time])


    print("""
Totoal Packets: [%i]
Exported Flows: [%i]
Open Flows: [%i]
            """ % (len(records), len(res_flow), len(open_flows)))

    return res_flow


def write_flow(flows, f_name):
    fid = open(f_name, 'w')
    for f in flows:
        # print(f)
        fid.write(' '.join([str(v) for v in f]) + '\n')
    fid.close()


def pcap2flow(pcap_file_name, flow_file_name, time_out):
    txt_f_name = pcap_file_name.rsplit('.pcap')[0] + '_tshark.txt'
    export_to_txt(pcap_file_name, txt_f_name)
    records, name = parse_records_tshark(txt_f_name)
    res_flows = change_to_flows(records, name, time_out)
    write_flow(res_flows, flow_file_name)


if __name__ == '__main__':
    input_file = '../data/WorldOfWarcraft.pcap'
    output_file = './flow.txt'
    pcap2flow(input_file, output_file, time_out=0.1)
