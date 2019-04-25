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
res = {}
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
    thelen = sumlen // 1000
    if thelen > 8:
        thelen = 8
    if thelen in res:
        res[thelen] = res[thelen] + 1
    else:
        res.update({thelen: 1})
print(res)

session = []
len = []
for i in range(0, 9):
    session.append(i * 1000)
    len.append(res[i])
for i in range(0, 9):
    print("{0} ~ {1}: {2}".format(i * 1000, (i + 1) * 1000, res[i]))

plt.plot(session, len)
plt.ylabel("session numbers")
plt.xlabel('section size(Bytes)')

plt.savefig("sess_len.jpg")

plt.show()
