# Data Generating Workflow
## 1. Install nDPI
Follow the instructions here:

```
https://github.com/ntop/nDPI
```

## 2. Use nDPI to process raw pcap file
Use the following command to use nDPI and save results into a file:

```
sudo ./ndpiReader -i YourPcapFile.pcap -v 1> ndpiout.txt
```

## 3. Use ndpi2label.py to filter the flows we want
Use the following command to filter save labels into a file:

```
ndpi2label.py ndpiout.txt label.txt
```

Statistics will be in the log.

Wanted protocol list can be changed

## 4. Use label2flow.py to generate pcap files we want from the labels
Use the following command: 

```
label2flow.py YourPcapFile.pcap label.txt
```

generate pcap files in each flow in ./result/protocols
