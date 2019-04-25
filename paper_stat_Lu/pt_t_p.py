import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

#
# pseg  = [50, 300,800,1500,3000,4000,6000,8000]
# pt_test = [0.2875,0.5589,0.7893,0.8679,0.9161,0.9107,0.904,0.8968]
# p_test = [0.25357142857142856,0.55178571428571432,0.75178571428571428,0.8125,0.82857142857142863,0.83571428571428574,0.8607142857142857,0.81607142857142856]
#


pseg = [50, 300, 800, 1500, 3000, 4000, 6000]
pt_test = [0.2875, 0.5589, 0.7893, 0.8679, 0.9161, 0.9107, 0.904]
p_test = [0.25357142857142856, 0.55178571428571432, 0.75178571428571428, 0.8125, 0.82857142857142863,
          0.83571428571428574, 0.8607142857142857]

p_test = np.asarray(p_test, dtype=float) * 100
pt_test = np.asarray(pt_test, dtype=float) * 100

tx = [120]
ttest = [0.81428571428571428]

l1 = plt.plot(pseg, pt_test, 'ro-', label='type1')
l2 = plt.plot(pseg, p_test, 'g*-', label='type2')
l3 = plt.plot(tx, ttest, 'b^-', label='type3')
plt.plot(pseg, pt_test, 'ro-', pseg, p_test, 'g*-', tx, ttest, 'b^-')
plt.ylim(0, 100)
# plt.title('The Lasers in Three Conditions')
plt.xlabel('Input size (bytes) of the model')
plt.ylabel('Classification accuracy (%)')
plt.legend(['Payload + IAT', 'Payload', 'IAT'], loc='lower right')
# plt.savefig("pt_t_p"+".jpg", dpi = 400)
plt.savefig('comparison_pt_t_p.pdf')
plt.show()
