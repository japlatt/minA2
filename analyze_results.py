#!/usr/bin/env python
'''
Jason Platt (jplatt@ucsd.edu)
Department of Physics
University of California, San Diego
2021
'''

import numpy as np
from generate_data import RK4
from def_dyn import get_dynamics
import matplotlib.pyplot as plt
from run_da import path_to_specs, read_specs
import argparse
import os
import seaborn as sns

plt.style.use('seaborn')

######### May need to change to match your files #################
def get_data(data_folder, name, data_file, twin, pred_len):
    results = np.load(data_folder + name +'_results.npz', allow_pickle = True)
    start_data = results['start_data']
    end_data = start_data + results['num_data'] + pred_len

    true_data = np.load(data_folder + 'all_' + name + '.npy')[start_data:end_data] if twin else None
    obs_data = np.load(data_folder + data_file) if data_file.endswith('npy')\
               else np.loadtxt(data_folder + data_file)

    # make images folder if does not exist
    folder_name = data_folder + name + '_images'
    if not os.path.exists(folder_name): os.makedirs(folder_name)
    
    return results, true_data, obs_data[start_data:end_data]
##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-np', '--num_pred',
                         type = int,
                         help = '<Required> num time steps to predict forward',
                         required = True)
    parser.add_argument('-t', '--twin',
                         help = '<Optional> was data generated by generate_data.py. If so plot true data',
                         action="store_true")
    parser.add_argument('-vars', '--vars_to_plot',
                        nargs = '+',
                        help = '<Optional> List of vars to plot, default plot all. ex: -vars 0 3 4 5',
                        type = int)
    args = parser.parse_args()
    return args

def plot_action(action, folder_name):
    fig1 = plt.figure(figsize = (10, 7))
    min_action = (0, np.inf)
    for i, a in enumerate(action):
        if a[-1] < min_action[1]: min_action = (i, a[-1])
        plt.plot(np.log(a), lw = 3)
    print('min action is {:1.3f} with id {:d}'.format(min_action[1], min_action[0]))
    print('param estimates', params[min_action[0]])
    plt.title('Action Levels ' + name, fontsize = 22)
    plt.ylabel('Log Action', fontsize = 16)
    plt.xlabel('beta', fontsize = 16)
    fig1.savefig(folder_name +'/' + name + '_action_levels.pdf', bbox_inches = 'tight')
    return min_action

def get_prediction(f, x0, time_arr, p, stim = None):
    num_pred = len(time_arr)
    dt = time_arr[1] - time_arr[0]
    sol = np.zeros((num_pred+1, len(x0)))
    sol[0] = x0
    for i, t in enumerate(time_arr, 1):
        sol[i] = RK4(f, sol[i-1], t, dt, params = (p, stim[i-1]))
    return sol[:-1]


if __name__ == '__main__':
    args = get_args()
    specs = read_specs(path_to_specs)

    data_folder = specs['data_folder']
    name = specs['name']

    obs_vars = specs['obs_dim'] if specs['obs_dim'] != -1 else np.arange(specs['num_dims'])
    twin = args.twin # was data generated by generate_data.py
    vars_to_plot = np.array(args.vars_to_plot) if args.vars_to_plot is not None else np.arange(specs['num_dims'])
    num_pred = args.num_pred # number of time steps to predict forward

    results, true_data, obs_data = get_data(data_folder, name, specs['data_file'], twin, num_pred)

    paths = results['path']
    params = results['params']
    action = results['action']
    time_est = results['time']
    dt = time_est[1]-time_est[0]

    if specs.get('stim_file') is not None:
        stim = np.load(data_folder + specs['stim_file'])[:, 1]
        stim_time = np.load(data_folder + specs['stim_file'])[:, 0]
        end_estimation = results['start_data']+results['num_data']
        assert(len(stim) > end_estimation+num_pred+1), 'stim not long enough for num_pred'
        stim = np.interp(np.linspace(0, num_pred+1, results['time_step']*(num_pred+1)),
                         np.linspace(0, num_pred+1, num_pred+1),
                         stim[end_estimation:end_estimation+num_pred+1])
    else: stim = np.empty(num_pred)
    
    # fig 1 action plot
    image_folder = data_folder + name + '_images'
    min_action = plot_action(action, image_folder)

    # fig 2 estimation and prediction for obs variables
    estimation = paths[min_action[0]].reshape(results['num_data'], results['num_dims'])
    dyn, _, _ = get_dynamics(specs)
    f = lambda r, t, params: np.array(dyn(r, t, *params))
    time_pred = np.linspace(time_est[-1], time_est[-1]+num_pred*dt, num_pred)
    prediction = get_prediction(f, estimation[-1], time_pred, params[min_action[0]], stim)

    set_vtp = set(vars_to_plot)
    set_ov = set(obs_vars)

    obs_plots = list(set_vtp.intersection(set_ov)); obs_plots.sort()
    unobs_plots = list(set_vtp.difference(set_ov)); unobs_plots.sort()

    colors = sns.color_palette('dark')
    fig_obs, ax_obs = plt.subplots(len(obs_plots), 1, sharex = True)
    if len(obs_plots): ax_obs = [ax_obs]
    for i, ax in enumerate(ax_obs):
        ax.plot(time_est, estimation[:, obs_plots[i]], color = colors[0], label = 'DA')
        ax.plot(time_pred, prediction[:, obs_plots[i]], color = colors[0])
        ax.plot(obs_data[:, 0], obs_data[:, i+1], color = colors[1], label = 'Obs')
        if twin: ax.plot(true_data[:, 0], true_data[:, obs_plots[i]+1],
                         color = colors[3], ls = '--', label = 'True')
        ax.axvline(time_est[-1], color = 'k')
        ax.set_ylabel(r'$x_{:d}$'.format(obs_plots[i]), fontsize = 16)
    ax_obs[-1].set_xlabel('Time', fontsize = 16)
    ax_obs[0].legend(loc = 'best')
    ax_obs[0].set_title(name + ' Observed Vars', fontsize = 22)
    fig_obs.savefig(image_folder +'/' + name + '_obs_vars.pdf', bbox_inches = 'tight')

    # fig 3 estimation and prediction for unobserved variables
    if twin:
        fig_unobs, ax_unobs = plt.subplots(len(unobs_plots), 1)
        for i, ax in enumerate(ax_unobs):
            ax.plot(time_est, estimation[:, unobs_plots[i]], color = colors[0], label = 'DA')
            ax.plot(time_pred, prediction[:, unobs_plots[i]], color = colors[0])
            ax.plot(true_data[:, 0], true_data[:, unobs_plots[i]+1],
                    color = colors[3], ls = '--', label = 'True')
            ax.axvline(time_est[-1], color = 'k')
            ax.set_ylabel(r'$x_{:d}$'.format(unobs_plots[i]), fontsize = 16)
        ax.set_xlabel('Time', fontsize = 16)
        ax_unobs[0].legend(loc = 'best')
        ax_unobs[0].set_title(name + ' Unobserved Vars', fontsize = 22)
        fig_unobs.savefig(image_folder +'/' + name + '_unobs_vars.pdf', bbox_inches = 'tight')

    plt.show()





