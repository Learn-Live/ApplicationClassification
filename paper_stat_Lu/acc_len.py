from matplotlib import style

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

# plt.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "pgf.preamble": [
#          r"\usepackage[utf8x]{inputenc}",
#          r"\usepackage[T1]{fontenc}",
#          r"\usepackage{cmbright}",
#          ]
# })

# style.use('seaborn-poster') #sets the size of the charts
# style.use('ggplot')
# style.use('seaborn-paper')


truncatelist = []
truncatelist.append(50)
for j in range(100, 1600, 100):
    truncatelist.append(j)
for j in range(1600, 3200, 200):
    truncatelist.append(j)
ide = []

# ide  = [50, 300,800,1500,3000,4000,6000,8000]
# test = [0.2875,0.5589,0.7893,0.8679,0.9161,0.9107,0.904,0.8968]

ide = [50, 300, 800, 1500, 3000, 4000, 6000]
test = [0.2875, 0.5589, 0.7893, 0.8679, 0.9161, 0.9107, 0.904]
test = np.asarray(test, dtype=float) * 100

# for i in range(1,len(train)+1):
#	ide.append(i)

# plt.plot(ide,train)

# plt.figure(figsize=(4.5, 2.5))

plt.plot(ide, test, 'b-o')
# plt.plot(ide, test, 'b-^', ide, test, 'g')
plt.ylabel("Classification accuracy (%)")
plt.xlabel('Input size (bytes) of the classification model')
plt.ylim(0, 100)
plt.tight_layout(.5)

plt.savefig("relation_inputsize_accuracy.pdf")  # should use before plt.show()

plt.show()

# plt.savefig('filename.png', dpi=300, format='.eps')
