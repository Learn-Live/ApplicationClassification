# Data Generating Workflow
## 1. Install nDPI
Follow the instructions here:

```
https://github.com/ntop/nDPI
```

## 2. Use nDPI to process raw pcap file
Use the following command to use nDPI and save results into a file:

```
sudo ./ndpiReader -i YourPcapFile.pcap -v 1 > ndpiout.txt
```

## 3. Use ndpi2label.py to filter the flows we want
Use the following command to filter save labels into a file:

```
python ndpi2label.py ndpiout.txt label.txt
```

Statistics will be in the log.

Wanted protocol list can be changed

## 4. Use label2flow.py to generate pcap files we want from the labels
Use the following command: 

```
python label2flow.py YourPcapFile.pcap label.txt
```

generate pcap files in each flow in ./result/protocols

## Tips:Use parseall.py to parse all the pcap and pcapng files in the current folder
Use the following command: 

```
python parseall.py
```

## Online source to find pcap files:
https://www.netresec.com/?page=PcapFiles
https://toolbox.google.com/datasetsearch/search?query=
https://cybervan.appcomsci.com:9000/datasets    ---Lu Xu
DARPA_Scalable_Network_Monitoring-20091103 (11/03/2009 to 11/12/2009) --- Lu Xu
https://www.caida.org/data/
https://github.com/awesomedata/awesome-public-datasets
https://ant.isi.edu/datasets/all.html
https://wand.net.nz/wits/index.php
https://www.secrepo.com/
https://www.simpleweb.org/wiki/index.php/Traces
https://www.stratosphereips.org/datasets-overview/
https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/ --Lu Xu



