import re
import sys
wantedProtocol = ["SSH","Bittorrent","RDP","FTP"] #all protocols with "." in it do not need to be filled in to it.

if __name__ == "__main__":
	statdic = {}
	
	if len(sys.argv) != 3:
		print("argv numbers:", len(sys.argv),sys.argv)
		print("usage: ndpi2lable.py ndpiresult.txt label.txt\ngetting statistics on the log and generate labels. Wanted protocol list can be changed\n* use ndpi -i pcap -v 1 to generate ndpiresult.txt")
		exit(0)
	ndpiResult = sys.argv[1]
	fout = open(sys.argv[2], "w")
	fout.close()
	fout = open(sys.argv[2], "a")
	with open(ndpiResult,"r",encoding = "UTF-8") as rfile:
		s = rfile.readline()
		while (s):
			s = s.strip()
			if(s == "Undetected flows:"):
				break			
			parselist = re.findall("TCP([\s\S]*?)proto: ([\s\S]*?)cat:",s)
			
			
			if (len(parselist) == 0):
				parselist = re.findall("UDP([\s\S]*?)proto: ([\s\S]*?)cat:",s)
				
			if (len(parselist) != 0):
				protocol = parselist[0][1].split("/")[1][:-2]
				if ("." not in protocol and protocol not in wantedProtocol):
					s = rfile.readline()
					continue
				if protocol not in statdic:
					statdic.update({protocol:1})
				else:
					statdic[protocol] += 1				
				ipline = parselist[0][0][:-1].strip()
				if '[' in ipline: #ipv6
					s = rfile.readline()
					continue
				sdlist = []
				if "<->" in ipline:
					sdlist = ipline.split("<->")
				else:
					sdlist = ipline.split("->")
				if (len(sdlist) != 2):
					print(s,"\n This line can not be parsed")
					s = rfile.readline()
					continue
				sip = sdlist[0].split(":")[0].strip()
				sport = sdlist[0].split(":")[1].strip()
				dip = sdlist[1].split(":")[0].strip()
				dport = sdlist[1].split(":")[1].strip()
				fout.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(sip,sport,dip,dport,protocol))
				
			s = rfile.readline()
	if (len(statdic) == 0):
		print("Finished, no desired flow found")	
	else:
		print("Finished:")
		with open("log.txt", "a",encoding="UTF-8") as logfile:
			for key, value in statdic.items():
				logfile.write(key+": "+str(value)+"\n")
			logfile.write("\n\n")
		for key, value in statdic.items():
			print(key,": ",value)
	fout.close()
			
