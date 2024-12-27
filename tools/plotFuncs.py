import numpy as np
import matplotlib.pyplot as plt

def plot_funcs(q_arr_max, d_arr_max, x_l_max, x_u_max, data_points_num, linewidth, stem_spec, plot_stem_or_not):
    r_nums = len(q_arr_max)

    # get x data
    x = np.zeros((r_nums, data_points_num))
    for i in range(r_nums):
        x[i, :] = np.linspace(x_l_max[i], x_u_max[i], data_points_num)

    y = np.log((q_arr_max * (np.exp(x) - 1) + 1) / (d_arr_max * (np.exp(x) - 1) + 1))

    plt.plot(x.T, y.T, linewidth=linewidth)

    ax = plt.gca()
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('incremental privacy loss')

    if plot_stem_or_not:
        x_l_max_plot = x_l_max[x_l_max > np.min(x_l_max)]
        q_arr_max_plot_l = q_arr_max[x_l_max > np.min(x_l_max)]
        d_arr_max_plot_l = d_arr_max[x_l_max > np.min(x_l_max)]

        x_u_max_plot = x_u_max[x_u_max < np.max(x_u_max)]
        q_arr_max_plot_u = q_arr_max[x_u_max < np.max(x_u_max)]
        d_arr_max_plot_u = d_arr_max[x_u_max < np.max(x_u_max)]

        q_arr_max_plot = np.vstack((q_arr_max_plot_l, q_arr_max_plot_u))
        d_arr_max_plot = np.vstack((d_arr_max_plot_l, d_arr_max_plot_u))

        x_end = np.hstack((x_l_max_plot, x_u_max_plot))

        y_end = np.log((q_arr_max_plot * (np.exp(x_end) - 1) + 1) / (d_arr_max_plot * (np.exp(x_end) - 1) + 1))
        plt.stem(x_end, y_end, linefmt=stem_spec, markerfmt='o', basefmt=" ", use_line_collection=True)
        plt.setp(plt.gca().lines, linewidth=linewidth, markersize=8)

    plt.show()

