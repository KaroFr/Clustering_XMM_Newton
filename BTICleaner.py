# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:44:58 2022

@author: Karolin.Frohnapfel
"""
import numpy as np
from sklearn.cluster import DBSCAN
from Helpers import pick_sample
from astropy.table import vstack
import matplotlib.pyplot as plt
from astropy.table import Table
import os, time

from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from Multithreading import Multithreading


"""
Class to detect and clean the bad time intervals in one observation
"""
class BTICleaner:
    n_btis = 0
    time_btis = 0.0
    time_gtis = 0.0
    plot_identifier = 'before'
    step_counter = 0
    eps_bti_det = 0.0
    events_after_cleaning = Table()
    
    def __init__(self, events, header, ID, n_bins_PI = 10, n_bins_DETX = 5, n_bins_DETY = 5, ms_bti_det = 100):
        self.events_before_cleaning = events
        self.obs_time = np.ptp(events['TIME'])
        observation_ID = header['OBS_ID']
        self.observation_ID = observation_ID
        self.ID = ID
        self.n_bins_PI = n_bins_PI
        self.n_bins_DETX = n_bins_DETX
        self.n_bins_DETY = n_bins_DETY
        self.ms_bti_det = ms_bti_det
        # make a directory for the upcoming results, if not already existend
        if not os.path.isdir(os.path.join('results', "results_ID_" + str(ID) + '_' + str(observation_ID) + '/')):
            os.mkdir(os.path.join('results', "results_ID_" + str(ID) + '_' + str(observation_ID) + '/'))
    
    """
    Method to get the important variables of this class to asses
    Output: dictionary of the main variables
    """
    def get_values(self):
        events_bc = self.events_before_cleaning
        events_ac = self.events_after_cleaning
        obs_time = self.obs_time
        var_dict = {'ID': [self.ID],
                    'obs_ID': [self.observation_ID],
                    'obs_time': [obs_time],
                    'range_DETX': [max(events_bc['DETX_norm']) - min(events_bc['DETX_norm'])],
                    'range_DETY': [max(events_bc['DETY_norm']) - min(events_bc['DETY_norm'])],
                    'range_PI': [max(events_bc['PI_norm']) - min(events_bc['PI_norm'])],
                    'range_TIME': [max(events_bc['TIME_norm']) - min(events_bc['TIME_norm'])],
                    'ms_bti_det': [self.ms_bti_det],
                    'eps_bti_det': [self.eps_bti_det],
                    'n_events_bc': [len(events_bc)],
                    'rate_bc': [len(events_bc)/obs_time],
                    'n_events_ac': [len(events_ac)],
                    'rate_ac': [len(events_ac)/obs_time],
                    }
        return var_dict
    
    """
    Method to perform DBSCAN on the TIME feature
    Input:  the events file and the hyper parameters
    Output: the clustered events file
    """
    def perform_DBSCAN_on_TIME(self, events, eps, min_pts):
        time_column = events["TIME"].reshape(-1, 1) 
        clustering = DBSCAN(eps = eps, min_samples = min_pts, n_jobs = -1).fit(time_column)
        events["flare_label"] = clustering.labels_
        return events
    
    """
    Method to detect the bad time intervals (btis)
    If the first run of DBSCAN doesnt give good results, 
    a second one is run
    """
    def detect_btis(self):
        self.step_counter = 1
        events = self.events_before_cleaning
        obs_time = self.obs_time
        
        # pick a sample (predefined method)
        events_thined, events_oob = pick_sample(events, 700000)
        thin_out = len(events_oob) > 0
        if thin_out:
            events_oob['flare_label'] = -1
          
        ################################################################################################
        ############################## Run the first round of DBSCAN
        ################################################################################################
        # set the hyperparameters
        ms_bti_det = self.ms_bti_det
        new_rate = len(events_thined)/obs_time
        eps_bti_det = ms_bti_det/(0.8*new_rate)
        self.eps_bti_det = eps_bti_det

        events_thined = self.perform_DBSCAN_on_TIME(events_thined, eps_bti_det, ms_bti_det)

        # calculate the number of bad time intervals (+1 because 0 is also a cluster)
        n_btis = max(events_thined["flare_label"]) + 1

        # calculate the time duration of the btis together
        time_btis = 0
        for cluster in range(0,n_btis):
            cluster_mask = (events_thined["flare_label"] == cluster)
            cluster_start = min(events_thined[cluster_mask]["TIME"])
            cluster_end = max(events_thined[cluster_mask]["TIME"])
            time_cluster = cluster_end - cluster_start
            time_btis = time_btis + time_cluster
           
        ################################################################################################
        ############################## Run the second round of DBSCAN
        ################################################################################################
        # compare the time of the good and the bad time intervals
        if time_btis/obs_time >= 0.75:
            eps_bti_det = ms_bti_det/(2.5 * new_rate)
            self.eps_bti_det = eps_bti_det
            events_thined = self.perform_DBSCAN_on_TIME(events_thined, eps_bti_det, ms_bti_det)
            # calculate the number of bad time intervals (+1 because 0 is also a cluster)
            n_btis = max(events_thined["flare_label"]) + 1
            
        ################################################################################################
        ############################## Apply the labels to the rest of the data
        ################################################################################################
        # calculate the flare_label for the rest of the data
        time_btis = 0
        for cluster in range(-1,n_btis):
            # calculate start and end of that specific bti
            cluster_mask = (events_thined["flare_label"] == cluster)
            cluster_start = min(events_thined[cluster_mask]["TIME"])
            cluster_end = max(events_thined[cluster_mask]["TIME"])
            # add the flare_labels in the oob events list
            if thin_out:
                events_oob["flare_label"][np.where((events_oob["TIME"] >= cluster_start) & (events_oob["TIME"] <= cluster_end))] = cluster
            
            # calculate the time of that bti
            time_cluster = cluster_end - cluster_start
                
            # add up the bad time
            if cluster != -1:
                time_btis = time_btis + time_cluster
         
        # calculate the length of the gtis in seconds
        self.time_gtis = obs_time - time_btis
        self.time_btis = time_btis
        self.n_btis = n_btis
        # put together the oob and thined events
        if thin_out:
            self.events_before_cleaning = vstack([events_thined, events_oob])
        else:
            self.events_before_cleaning = events_thined.copy()
        
        print('  - Finished the bti detection.')

    """
    Method to clean the bad time intervals (btis)
    Input:  the number of bins for the cleaning
    From each bin we will randomly remove points to match the btis to the gtis
    """
    def clean_btis(self):
        n_bins_PI = self.n_bins_PI
        n_bins_DETX = self.n_bins_DETX
        n_bins_DETY = self.n_bins_DETY
        
        if self.step_counter == 0:
            self.detect_btis()
        self.step_counter = 2
        
        events = self.events_before_cleaning
        n_btis = self.n_btis
        time_gtis = self.time_gtis
        
        # set the detector boundaries
        self.min_DETX = min(events['DETX'])
        self.min_DETY = min(events['DETY'])
        self.max_DETX = max(events['DETX'])
        self.max_DETY = max(events['DETY'])
        
        # get the quantiles of PI
        quant_prob = np.round(np.linspace(0.0, 1.0, num = n_bins_PI+1), 3)
        quantiles = np.quantile(events["PI"], quant_prob)

        # divide the events in different energy bins
        energy_events_list = []
        for i in range(n_bins_PI):
            start = quantiles[i]
            end = quantiles[i + 1]
            if i == 0:
                energy_mask = (events['PI'] >= start) & (events['PI'] <= end)
            else:
                energy_mask = (events['PI'] > start) & (events['PI'] <= end)
            energy_events = events[energy_mask]
            energy_events_list.append(energy_events)

        # initiate a table for the cleaned events
        gtis_mask = (events['flare_label'] == -1)
        events_PN_btis_cleaned = events[gtis_mask]

        # need it for the btis_table
        n_rows = n_bins_DETX*n_bins_DETY

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #
        #   Sequential job
        #
        # print('-' * 100)
        # print('  - Start clean_btis in a row ...')
        # t_0 = time.time()
        # for energy_events in energy_events_list:
        #     self.clean_btis_thread(input=energy_events,
        #                            events_PN_btis_cleaned=events_PN_btis_cleaned, n_rows=n_rows)
        # t_1 = time.time() - t_0
        # print(f'\t** Time to run all threads in sequence : {t_1:.2f} seconds')
        # print('-' * 100)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #
        #   Parallel job
        #
        print('  - Start clean_btis in parallel ...')
        t_4 = time.time()
        energy_events_list_len = len(energy_events_list)
        threads = energy_events_list_len
        m = Multithreading(threads=threads, function=self.clean_btis_thread, input_list=energy_events_list,
                           events_PN_btis_cleaned=events_PN_btis_cleaned, n_rows=n_rows)
        m.multi()
        t_5 = time.time()
        print(f'\t** Time to run all threads in parallel : {t_5 - t_4:.4f} seconds')
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.events_after_cleaning = Table(events_PN_btis_cleaned)
        self.plot_identifier = 'after'
        
        print('  - Finished the bti cleaning.')


    """
    Parallel thread for cleaning BTIs
    """
    def clean_btis_thread(self, input=[], events_PN_btis_cleaned=[], n_rows=1):
        energy_events = input

        n_bins_DETY = self.n_bins_DETY
        min_DETX = self.min_DETX
        min_DETY = self.min_DETY
        max_DETX = self.max_DETX
        max_DETY = self.max_DETY
        n_bins_DETX = self.n_bins_DETX
        n_btis = self.n_btis
        time_gtis = self.time_gtis
        events = self.events_before_cleaning

        l = len(energy_events)
        print(f'\t- starting with energy bin no # energy_events len = {l}')

        energy_events_gtis = energy_events[(energy_events['flare_label'] == -1)]
        energy_events_btis = energy_events[(energy_events['flare_label'] != -1)]

        # set the bins
        quant_prob_DETY = np.round(np.linspace(0.0, 1.0, num=n_bins_DETY + 1), 3)
        quant_prob_DETX = np.round(np.linspace(0.0, 1.0, num=n_bins_DETY + 1), 3)
        quantiles_DETY = np.quantile(events["DETY"], quant_prob_DETY)
        quantiles_DETX = np.quantile(events["DETX"], quant_prob_DETX)

        # calculate the histogram for all the good time intervals
        H_gtis, xedges, yedges = np.histogram2d(energy_events_gtis["DETX"], energy_events_gtis["DETY"],
                                                bins=(quantiles_DETX, quantiles_DETY))
        xedges[0] = min_DETX - 1
        yedges[0] = min_DETY - 1
        xedges[-1] = max_DETX + 1
        yedges[-1] = max_DETY + 1

        # start a table for the bad time intervals and how they should look
        btis_table = Table([np.zeros(n_rows), np.zeros(n_rows), np.zeros(n_rows), np.zeros(n_rows), np.zeros(n_rows)],
                           names=('DETX_bin_start', 'DETX_bin_end', 'DETY_bin_start', 'DETY_bin_end', 'counts_gtis'))

        for i in range(n_bins_DETX):
            for j in range(n_bins_DETY):
                btis_table['DETX_bin_start'][i * n_bins_DETX + j] = xedges[i]
                btis_table['DETX_bin_end'][i * n_bins_DETX + j] = xedges[i + 1]
                btis_table['DETY_bin_start'][i * n_bins_DETX + j] = yedges[j]
                btis_table['DETY_bin_end'][i * n_bins_DETX + j] = yedges[j + 1]
                btis_table['counts_gtis'][i * n_bins_DETX + j] = H_gtis[i][j]

        for bti_counter in range(n_btis):
            # extract the events of one specific cluster of flaring background found by DBSCAN
            cluster_mask = (energy_events_btis['flare_label'] == bti_counter)
            events_PN_cluster = energy_events_btis[cluster_mask]

            if len(events_PN_cluster) > 0:
                # get the time of the specific cluster
                time_cluster = max(events_PN_cluster["TIME"]) - min(events_PN_cluster["TIME"])

                btis_table['bti_' + str(bti_counter) + '_keep'] = (
                            (btis_table['counts_gtis'] / time_gtis) * time_cluster)

                # now in one cluster look at each position bin
                for bin_counter in range(len(btis_table)):
                    # start and end of the bin
                    DETY_start = btis_table['DETY_bin_start'][bin_counter]
                    DETY_end = btis_table['DETY_bin_end'][bin_counter]
                    DETX_start = btis_table['DETX_bin_start'][bin_counter]
                    DETX_end = btis_table['DETX_bin_end'][bin_counter]

                    # extract the number of events to be kept
                    n_keep = btis_table['bti_' + str(bti_counter) + '_keep'][bin_counter]

                    # get only those events within the position interval
                    remove_mask = (events_PN_cluster['DETY'] > DETY_start) & (events_PN_cluster['DETY'] <= DETY_end) & (
                                events_PN_cluster['DETX'] > DETX_start) & (events_PN_cluster['DETX'] <= DETX_end)
                    events_for_remove = events_PN_cluster[remove_mask]

                    # randomly remove n_remove events
                    events_cleaned_interval = np.random.choice(events_for_remove,
                                                               size=min(round(n_keep), len(events_for_remove)),
                                                               replace=False)

                    # append the to a cleaned data set
                    events_PN_btis_cleaned = np.append(events_PN_btis_cleaned, events_cleaned_interval)





    """
    Plot the distributions according to position, energy and time
    The plot identifier changes after the cleaning and determines, 
    if the events before or after cleaning are used
    """
    def plot_distribution(self):
        plot_identifier = self.plot_identifier
        if plot_identifier == 'before':
            events = self.events_before_cleaning
        else:
            events = self.events_after_cleaning
        ID = self.ID
        observation_ID = self.observation_ID
        step_counter = self.step_counter
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        plt.rcParams['font.size'] = '12'
        plt.suptitle("Distributions of observation " + str(observation_ID) + " " + plot_identifier + " cleaning", fontsize=16)

        ax[0,0].hist(events['TIME_norm'], bins = 60, edgecolor = 'BLACK')
        ax[0,1].hist(events['PI_norm'], bins = 60, edgecolor = 'BLACK')
        ax[1,0].hist(events['DETX_norm'], bins = 60, edgecolor = 'BLACK')
        ax[1,1].hist(events['DETY_norm'], bins = 60, edgecolor = 'BLACK')
        ax[0,0].set_xlabel('TIME_norm')
        ax[0,1].set_xlabel('PI_norm')
        ax[1,0].set_xlabel('DETX_norm')
        ax[1,1].set_xlabel('DETY_norm')       
        ax[0,0].set_ylabel("Counts")
        ax[1,0].set_ylabel("Counts")

        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_distributions_" + plot_identifier + "_cleaning.jpg", dpi = 150)
        plt.close()
    
    """
    Compare the distributions of position, energy and time for good and bad time intervals
    The plot identifier changes after the cleaning and determines, 
    if the events before or after cleaning are used
    """
    def plot_bti_comparison(self):
        plot_identifier = self.plot_identifier
        if plot_identifier == 'before':
            events = self.events_before_cleaning
        else:
            events = self.events_after_cleaning
        ID = self.ID
        observation_ID = self.observation_ID
        step_counter = self.step_counter
        time_btis = self.time_btis
        time_gtis = self.time_gtis
        obs_time = self.obs_time
        events_gtis = events[(events['flare_label'] == -1)]
        events_btis = events[(events['flare_label'] != -1)]
        
        ###### Set up for the plot
        fig, ax = plt.subplots(3, 3, figsize=(24, 24))
        plt.rcParams['font.size'] = '12'
        plt.suptitle("Comparison of the bad and good time intervals " + plot_identifier + " cleaning", fontsize=20)
        
        ##############################################################################
        ###### Plot of the bad and good time intervals looking at DETX (first row)
        ##############################################################################
        n_DETX, bins_DETX, _ = ax[0,0].hist(events["DETX"], bins = 60, edgecolor = 'BLACK')
        n_gtis_DETX,_,_ = ax[0,1].hist(events_gtis["DETX"], bins = 60, edgecolor = 'BLACK')
        n_btis_DETX,_,_ = ax[0,2].hist(events_btis["DETX"], bins = 60, edgecolor = 'BLACK')
        
        #calculate table for the bad time intervals and how it should look
        btis_table_DETX = Table([bins_DETX[0:len(bins_DETX)-1],bins_DETX[1:len(bins_DETX)],n_DETX,n_gtis_DETX, n_btis_DETX],
                                names=('start of bin','end of bin', 'total counts', 'counts in gtis', 'counts in btis'))
        btis_table_DETX['should be'] = btis_table_DETX['counts in gtis']/time_gtis * time_btis
        btis_table_DETX['to be removed'] = btis_table_DETX['counts in btis'] - btis_table_DETX['should be'] 
        
        # plot a red line for all the events that should be removed
        ax[0,2].plot((btis_table_DETX['start of bin'] + btis_table_DETX['end of bin'])/2, btis_table_DETX['to be removed'], color="red")
        # plot a green line to see, how the distribution in the bad time interval should look like
        ax[0,2].plot((btis_table_DETX['start of bin'] + btis_table_DETX['end of bin'])/2, btis_table_DETX['should be'], color="lime")
        
        ax[0,0].axis([min(events["DETX"]), max(events["DETX"]), -1000, max(n_DETX)])
        ax[0,1].axis([min(events["DETX"]), max(events["DETX"]), -1000, max(n_DETX)])
        ax[0,2].axis([min(events["DETX"]), max(events["DETX"]), -1000, max(n_DETX)])
        ax[0,0].set_xlabel("DETX")
        ax[0,1].set_xlabel("DETX")
        ax[0,2].set_xlabel("DETX")
        ax[0,0].set_ylabel("Counts")
        ax[0,0].set_title("All the events: {:.2f} s".format(obs_time))
        ax[0,1].set_title("Good time intervals only: {:.2f} s".format(time_gtis))
        ax[0,2].set_title("Bad time intervals only: {:.2f} s".format(time_btis))
        
        ##############################################################################
        ###### Plot of the bad and good time intervals looking at DETY
        ##############################################################################
        
        n_DETY, bins_DETY, _ = ax[1,0].hist(events["DETY"], bins = 60, edgecolor = 'BLACK')
        n_gtis_DETY,_,_ = ax[1,1].hist(events_gtis["DETY"], bins = 60, edgecolor = 'BLACK')
        n_btis_DETY,_,_ = ax[1,2].hist(events_btis["DETY"], bins = 60, edgecolor = 'BLACK')
        
        #calculate table for the bad time intervals and how it should look
        btis_table_DETY = Table([bins_DETY[0:len(bins_DETY)-1],bins_DETY[1:len(bins_DETY)],n_DETY,n_gtis_DETY, n_btis_DETY],
                                names=('start of bin','end of bin', 'total counts', 'counts in gtis', 'counts in btis'))
        btis_table_DETY['should be'] = btis_table_DETY['counts in gtis']/time_gtis * time_btis
        btis_table_DETY['to be removed'] = btis_table_DETY['counts in btis'] - btis_table_DETY['should be'] 
        
        # plot a red line for all the events that should be removed
        ax[1,2].plot((btis_table_DETY['start of bin'] + btis_table_DETY['end of bin'])/2, btis_table_DETY['to be removed'], color="red")
        # plot a green line to see, how the distribution in the bad time interval should look like
        ax[1,2].plot((btis_table_DETY['start of bin'] + btis_table_DETY['end of bin'])/2, btis_table_DETY['should be'], color="lime")
        
        ax[1,0].axis([min(events["DETY"]), max(events["DETY"]), -1000, max(n_DETY)])
        ax[1,1].axis([min(events["DETY"]), max(events["DETY"]), -1000, max(n_DETY)])
        ax[1,2].axis([min(events["DETY"]), max(events["DETY"]), -1000, max(n_DETY)])
        ax[1,0].set_ylabel("Counts")
        ax[1,0].set_xlabel("DETY")
        ax[1,1].set_xlabel("DETY")
        ax[1,2].set_xlabel("DETY")
        
        ##############################################################################
        ###### Plot of the bad and good time intervals looking at the Energy (PI)
        ##############################################################################
        
        n_PI, bins_PI, _ = ax[2,0].hist(events["PI"], bins = 60, edgecolor = 'BLACK')
        n_gtis_PI,_,_ = ax[2,1].hist(events_gtis["PI"], bins = 60, edgecolor = 'BLACK')
        n_btis_PI,_,_ = ax[2,2].hist(events_btis["PI"], bins = 60, edgecolor = 'BLACK')
        
        #calculate table for the bad time intervals and how it should look
        btis_table_PI = Table([bins_PI[0:len(bins_PI)-1],bins_PI[1:len(bins_PI)],n_PI,n_gtis_PI, n_btis_PI],
                              names=('start of bin','end of bin', 'total counts', 'counts in gtis', 'counts in btis'))
        btis_table_PI['should be'] = btis_table_PI['counts in gtis']/time_gtis * time_btis
        btis_table_PI['to be removed'] = btis_table_PI['counts in btis'] - btis_table_PI['should be'] 
        
        # plot a red line for all the events that should be removed
        ax[2,2].plot((btis_table_PI['start of bin'] + btis_table_PI['end of bin'])/2, btis_table_PI['to be removed'], color="red")
        # plot a green line to see, how the distribution in the bad time interval should look like
        ax[2,2].plot((btis_table_PI['start of bin'] + btis_table_PI['end of bin'])/2, btis_table_PI['should be'], color="lime")
        
        ax[2,0].axis([min(events["PI"]), max(events["PI"]), -1000, max(n_PI)])
        ax[2,1].axis([min(events["PI"]), max(events["PI"]), -1000, max(n_PI)])
        ax[2,2].axis([min(events["PI"]), max(events["PI"]), -1000, max(n_PI)])
        ax[2,0].set_xlabel("PI")
        ax[2,1].set_xlabel("PI")
        ax[2,2].set_xlabel("PI")
        ax[2,0].set_ylabel("Counts")
        
        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_compare_btis_" + plot_identifier + "_cleaning.jpg", dpi = 150)
        plt.close()
    
    """
    3 dimensional scatter plot of position and time.
    The color is determined by the energy.
    The plot identifier changes after the cleaning and determines, 
    if the events before or after cleaning are used.
    """
    def plot_DETX_DETY_TIME(self):
        plot_identifier = self.plot_identifier
        if plot_identifier == 'before':
            events = self.events_before_cleaning
        else:
            events = self.events_after_cleaning
        observation_ID = self.observation_ID
        step_counter = self.step_counter
        ID = self.ID
        
        fig = plt.figure(figsize=(16,12))
        plt.suptitle("3D Plot of observation " + str(observation_ID) + " " + plot_identifier + " cleaning", fontsize=16)
        ax = fig.add_subplot(projection='3d')

        #to use ENERGY_BAND as color in the plots you need this colormap
        colormap = np.array(['r', 'g', 'b'])
        ax.scatter(xs = events['TIME_norm'], ys = events['DETY_norm'], zs = events['DETX_norm'], c=colormap[events["ENERGY_BAND"].astype(int)], s = 0.04)

        # set the size of the axis
        ax.set_box_aspect(aspect = (4,1,1))
        ax.set_xlabel('TIME_norm')
        ax.set_ylabel('DETY_norm')
        ax.set_zlabel('DETX_norm')

        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_3D_Plot_" + plot_identifier + "_cleaning.jpg", dpi = 150)
        plt.close()
    
    """
    2 dimensional scatter plots of the position before and after cleaning.
    The color is determined by the energy.
    The plot identifier changes after the cleaning and determines, 
    if the events before or after cleaning are used.
    """
    def plot_DETX_DETY(self):
        plot_identifier = self.plot_identifier
        if plot_identifier == 'before':
            events = self.events_before_cleaning
        else:
            events = self.events_after_cleaning
        observation_ID = self.observation_ID
        step_counter = self.step_counter
        ID = self.ID
        events_before_cleaning = self.events_before_cleaning
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        plt.rcParams['font.size'] = '12'
        
        colormap = np.array(['r', 'g', 'b'])
        ax[0].scatter(events_before_cleaning['DETX_norm'], events_before_cleaning['DETY_norm'], c = colormap[events_before_cleaning["ENERGY_BAND"].astype(int)], s = 0.004)
        ax[1].scatter(events['DETX_norm'], events['DETY_norm'], c = colormap[events["ENERGY_BAND"].astype(int)], s = 0.004)

        ax[0].set_title("before cleaning")
        ax[1].set_title("after cleaning")
        ax[0].set_xlabel("DETX_norm")
        ax[1].set_xlabel("DETX_norm")
        ax[0].set_ylabel("DETY_norm")

        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_compare_image_" + plot_identifier + "_cleaning.jpg", dpi = 150)
        plt.close()
    
    """
    2 dimensional scatter plots of DETY and TIME before and after cleaning.
    The color is determined by the energy.
    The plot identifier changes after the cleaning and determines, 
    if the events before or after cleaning are used.
    """
    def plot_DETY_TIME(self):
        plot_identifier = self.plot_identifier
        if plot_identifier == 'before':
            events = self.events_before_cleaning
        else:
            events = self.events_after_cleaning
        observation_ID = self.observation_ID
        step_counter = self.step_counter
        ID = self.ID
        events_before_cleaning = self.events_before_cleaning
        
        remove_events = events[events['flare_label'] == -1]

        fig, ax = plt.subplots(3, 1, figsize=(12, 9))
        colormap = np.array(['r', 'g', 'b'])
        ax[0].scatter(events_before_cleaning['TIME_norm'], events_before_cleaning['DETY_norm'], c = colormap[events_before_cleaning['ENERGY_BAND'].astype(int)], s = 0.004)
        ax[1].scatter(remove_events['TIME_norm'], remove_events['DETY_norm'], c = colormap[remove_events['ENERGY_BAND'].astype(int)], s = 0.004)
        ax[2].scatter(events['TIME_norm'], events['DETY_norm'], c = colormap[events['ENERGY_BAND'].astype(int)], s = 0.004)

        ax[0].set_title("before cleaning")
        ax[1].set_title("detected btis")
        ax[2].set_title("after cleaning")
        ax[2].set_xlabel("TIME_norm")
        ax[0].set_ylabel("DETY_norm")
        ax[1].set_ylabel("DETY_norm")
        ax[2].set_ylabel("DETY_norm")
        ax[0].axis([min(events_before_cleaning['TIME_norm']), max(events_before_cleaning['TIME_norm']), min(events_before_cleaning['DETY_norm']), max(events_before_cleaning['DETY_norm'])])
        ax[1].axis([min(events_before_cleaning['TIME_norm']), max(events_before_cleaning['TIME_norm']), min(events_before_cleaning['DETY_norm']), max(events_before_cleaning['DETY_norm'])])
        ax[2].axis([min(events_before_cleaning['TIME_norm']), max(events_before_cleaning['TIME_norm']), min(events_before_cleaning['DETY_norm']), max(events_before_cleaning['DETY_norm'])])

        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_compare_DETY_TIME_" + plot_identifier + "_cleaning.jpg", dpi = 150)
        plt.close()