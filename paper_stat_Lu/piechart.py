import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

# data to plot
recipe = ['Slack: 888', 'Google: 16149', 'Bing: 901', 'Twitter: 7613', 'Youtube: 2031', 'Outlook: 1661', 'Github:1320',
          'Facebook: 717', 'Others: 10899']

data = [888, 16149, 901, 7613, 2031, 1661, 1320, 717, 10899]

# colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
# explode = (0.1, 0, 0, 0)  # explode 1st slice

# Plot
# plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
# plt.pie(sizes,  labels=labels,  startangle=140)
# plt.axis('equal')


fig, ax = plt.subplots(figsize=(6, 2.8), subplot_kw=dict(aspect="equal"))

# fig, ax = plt.subplots()


wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=100)

bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.00)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1) / 2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.3 * np.sign(x), 1.3 * y),
                horizontalalignment=horizontalalignment, **kw)

plt.savefig('application_distribution.pdf')
# plt.savefig("distribution.jpg", dpi = 400)
plt.show()
