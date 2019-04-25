import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

numdic = {}
sumdic = {}
for i in range(1, 9):
    numdic.update({i: 0})

for i in range(1, 9):
    sumdic.update({i: 0})

numdic.update({0: 1})
sumdic.update({0: 0})

with open("pkt_size.txt", "r") as rfile:
    s = rfile.read()

li = s.split("\n")
for line in li:
    numline = (line.split("\t")[0]).strip()[1:-1]
    label = int(line.split("\t")[1].strip())
    zili = numline.split(",")
    numli = []

    for i in zili:
        numli.append(int(i))
    current = 0
    sumlen = 0
    sumpkt = 0
    for item in numli:
        sumlen += item
        sumpkt += 1
        for i in range(current + 1, 9):
            if i * 1000 <= sumlen:
                numdic[i] = numdic[i] + 1
                sumdic[i] = sumdic[i] + sumpkt
        # print(sumdic[i]+sumpkt)
        current = sumlen // 1000
res = []
for i in range(0, 9):
    res.append(sumdic[i] / numdic[i])
print(res)

# session = [0,1000,2000,3000,4000,5000,6000,7000,8000]
# len = [0.0, 3.3820054628624083, 4.8554581426022185, 6.311031461569096, 7.189020486555698, 8.181256618425698, 9.121847891870082, 9.711817406143345, 10.453927161035542]

session = [0, 1000, 2000, 3000, 4000, 5000, 6000]
len = [0.0, 3.3820054628624083, 4.8554581426022185, 6.311031461569096, 7.189020486555698, 8.181256618425698,
       9.121847891870082]

plt.plot(session, len, 'b-^')
plt.ylabel("The number of packets needed")
plt.xlabel('Input size (Bytes) of the model')

# plt.savefig("pkt_len.jpg")
plt.savefig("relation_inputsize_pktsnum.pdf")
plt.show()
