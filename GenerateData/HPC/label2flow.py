import sys
import time
import os


if __name__ == "__main__":
	statdic = {}
	if len(sys.argv) != 3:
		print("argv numbers:", len(sys.argv),sys.argv)
		print("label2pcap.py input.pcap label.txt\ngenerate flow pcap files in  ./result/protocols")
		exit(0)
	rawpcap= sys.argv[1]
	label =sys.argv[2]
	with open(label,"r") as rfile:
		s = rfile.readline()
		while (s):
			s = s.strip()
			ftuple = s.split("\t")#[sip,sport,dip,dport,protocol]
			sip = ftuple[0]
			sport = ftuple[1]
			dip = ftuple[2]
			dport = ftuple[3]
			protocol = ftuple[4]
			if protocol not in statdic:
				statdic.update({protocol:1})
				if not os.path.exists("./result/"+protocol):
					os.makedirs("./result/"+protocol)
			command  = 'tcpdump -r '+rawpcap+' -w '+'./result/'+protocol +'/'+ftuple[0]+'_'+ftuple[1]+'_'+ftuple[2]+'_'+ftuple[3]+'_'+protocol+'.pcap "((src host '+sip+' and src port '+sport+') and (dst host '+dip+' and dst port '+dport+')) or ((src host '+dip+' and src port '+ dport+') and (dst host '+sip+' and dst port '+sport+'))"'
			os.system(command)
			s = rfile.readline()

