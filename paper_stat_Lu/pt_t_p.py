import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

#
# pseg  = [50, 300,800,1500,3000,4000,6000,8000]
# pt_test = [0.2875,0.5589,0.7893,0.8679,0.9161,0.9107,0.904,0.8968]
# p_test = [0.25357142857142856,0.55178571428571432,0.75178571428571428,0.8125,0.82857142857142863,0.83571428571428574,0.8607142857142857,0.81607142857142856]
#
def plot_p_t_pt():
    pseg = [50, 300, 800, 1500, 3000, 4000, 6000]
    pt_test = [0.2875, 0.5589, 0.7893, 0.8679, 0.9161, 0.9107, 0.904]
    p_test = [0.25357142857142856, 0.55178571428571432, 0.75178571428571428, 0.8125, 0.82857142857142863,
              0.83571428571428574, 0.8607142857142857]

    p_test = np.asarray(p_test, dtype=float) * 100
    pt_test = np.asarray(pt_test, dtype=float) * 100

    tx = [120, 100]
    ttest = [0.81428571428571428, 0.9]

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


def plot_p_t_pt_two_axes():
    pseg = [50, 300, 800, 1500, 3000, 4000, 6000]
    pt_test = [0.2875, 0.5589, 0.7893, 0.8679, 0.9161, 0.9107, 0.904]
    p_test = [0.25357142857142856, 0.55178571428571432, 0.75178571428571428, 0.8125, 0.82857142857142863,
              0.83571428571428574, 0.8607142857142857]

    p_test = np.asarray(p_test, dtype=float) * 100
    pt_test = np.asarray(pt_test, dtype=float) * 100

    # tx = [110, 130, 150]
    # ttest = [0.81428571428571428, 0.9,0.92]
    tx = [16, 20, 24, 28, 32, 36, 40, 44]
    tx = np.asarray(tx, dtype=float) / 4
    ttest = [0.6964285714285714, 0.79960317176485818, 0.78174603269213727, 0.82142857426688787, 0.79761905045736403,
             0.80952380857770401, 0.79960317649538559, 0.81746031935252839]
    ttest = np.asarray(ttest, dtype=float) * 100

    ax1 = plt.subplot(111)
    l1 = ax1.plot(pseg, pt_test, 'ro-', label='type1')
    l2 = ax1.plot(pseg, p_test, 'g*-', label='type2')
    l3 = ax1.plot([], [], 'b^-', label='type3')
    ax1.set_xlabel('Input size (bytes) of the model')

    ### set second x-axis
    ax2 = ax1.twiny()

    ax2_label = tx
    ax2.set_xticks(ax2_label)
    # ax2.set_xticklabels(tx)

    ax2.xaxis.set_ticks_position('bottom')  # set the ticks position of the second x-axis to bottom
    ax2.xaxis.set_label_position('bottom')  # set the label of the second x-axis to bottom
    ax2.spines['bottom'].set_position(('outward', 36))
    ax2.set_xlabel('Num. of IAT')
    ax2.set_xlim(tx[0] - 0.5, tx[-3] - 0.5)
    l3 = ax2.plot(tx[:-3], ttest[:-3], 'b^-', label='type3')

    # plt.plot(pseg, pt_test, 'ro-', pseg, p_test, 'g*-', tx, ttest, 'b^-')
    ax1.set_ylim(0, 100)
    # plt.title('The Lasers in Three Conditions')
    # plt.xlabel('Input size (bytes) of the model')
    ax1.set_ylabel('Classification accuracy (%)')
    ax1.legend(['Payload + IAT', 'Payload', 'IAT'], loc='lower right')
    # ax2.legend(['Payload + IAT', 'Payload', 'IAT'], loc='lower right')
    # plt.savefig("pt_t_p"+".jpg", dpi = 400)
    plt.savefig('comparison_pt_t_p.pdf')
    plt.show()


plot_p_t_pt_two_axes()
