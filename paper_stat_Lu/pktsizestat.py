s = ""
with open("pkt_size.txt", "r") as rfile:
    s = rfile.read()
res = {}
sres = {}
li = s.split("\n")
for line in li:
    tli = line.strip().split("\t")
    pkt = tli[0]
    label = int(tli[1])
    pktnum = len(pkt.split(","))
    if label in res:
        res[label] += pktnum
        sres[label] += 1
    else:
        res.update({label: pktnum})
        sres.update({label: 1})

print(res)
print(sres)
