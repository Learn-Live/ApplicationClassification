import matplotlib.pyplot as plt


def plot_data(x_data, data, x_label, y_label, title=''):
    r"""

    :param data:
    :param x_label:
    :param y_label:
    :param title:
    :return:
    """
    # recommend to use, plt.subplots() default parameter is (111)
    fig, ax = plt.subplots()  # combine plt.figure and fig.add_subplots(111)
    ax.plot(x_data, data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.ylim(1,3)
    ax.set_title(title)
    # ax.set_ylabel()
    plt.show()


if __name__ == '__main__':
    # Accuracy of using only interval time:
    # number_of_IT = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    number_of_IAT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    acc = [0.27919911012235815, 0.61179087875417126, 0.71523915461624021, 0.7730812013348165, 0.7730812013348165,
           0.77864293659621797, 0.77419354838709675, 0.77641824249165736, 0.7730812013348165, 0.76307007786429371,
           0.76529477196885431, 0.77085650723025589, 0.77641824249165736, 0.77085650723025589, 0.76863181312569517]

    plot_data(number_of_IAT, acc, x_label='Number of IAT', y_label='Accuracy')
