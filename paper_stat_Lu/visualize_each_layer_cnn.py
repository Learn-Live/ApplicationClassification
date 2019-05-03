import matplotlib.pyplot as plt
import numpy as np


def load_data_from_txt(input_file):
    layer_data_dict = {}
    with open(input_file, 'r') as in_hdr:
        line = in_hdr.readline()
        key = ''
        while line:
            if line.startswith('###'):
                key = line.strip('###').strip('\n').strip()
                line = in_hdr.readline()
                continue
            if key != '':
                if line.startswith('X'):
                    X = eval(line.split('= ')[-1])
                    # X1=X
                elif line.startswith('Y'):
                    y = eval(line.split('= ')[-1])
                    layer_data_dict[key] = [X, y]
                    key = ''
                else:
                    print(line)
                line = in_hdr.readline()
                continue
            else:
                line = in_hdr.readline()

    return layer_data_dict


def draw_data(X, y, name='layer_5', title=''):
    plt.plot(X, label=name)
    plt.show()


if __name__ == '__main__':
    input_file = 'cnn_layer_data.txt'
    layer_data_dict = load_data_from_txt(input_file)

    for idx, (key, value) in enumerate(layer_data_dict.items()):
        X = value[0]
        Y = value[1]
        data_categories_dict = {}
        for (x, y) in zip(X, Y):
            if y not in data_categories_dict.keys():
                data_categories_dict[y] = [x]
            else:
                data_categories_dict[y].append(x)

        # fig, ax = plt.subplots(3, 2)
        fig = plt.figure()
        for i, (key_category, value_category) in enumerate(data_categories_dict.items()):
            X_category = np.mean(np.asarray(value_category, dtype=float), axis=0)

            name = key
            # >> > fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
            # >> > axes[0, 0].plot(x, y)
            # >> > axes[1, 1].scatter(x, y)
            ax1 = fig.add_subplot(4, 2, i + 1)
            if i % 2 == 0:
                ax1.plot(X_category, 'r-', label=name)
            else:
                ax1.plot(X_category, 'b-', label=name)
            ax1.set_title(key_category)
        plt.show()
        break
