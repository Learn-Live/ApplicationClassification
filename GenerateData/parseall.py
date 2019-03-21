import os
import time
f = os.listdir("./")
sufix = str(int(time.time()))[5:10]
for doc in f:

	if doc.split(".")[-1].strip() == "pcapng" or doc.split(".")[-1].strip() == "pcap":
		os.system("./ndpiReader -i "+doc+" -v 1 > out"+sufix)
		os.system("python3 ndpi2label.py out"+sufix+" label2.txt"+sufix)
		os.system("python3 label2flow.py "+doc+" label2.txt"+sufix)
		print("Finished:"+doc)
