# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:45:40 2022

@author: Karolin.Frohnapfel
"""
import numpy as np
from astropy.table import Table
from astropy.table import vstack
from sklearn.cluster import DBSCAN
import hdbscan
from Helpers import pick_sample
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import itertools
import os

from Multithreading import Multiprocessing

"""
Class to use clustering (DBSCAN and HDBSCAN) to identify sources
"""
class SourceDetector:
    step_counter = 2
    eps_noise_det = 0.0
    n_events_and = 0
    n_noise_and = 0
    n_events_acc = 0
    n_noise_acc = 0
    clusters_HDBSCAN = Table()
    final_clusters_HDBSCAN = Table()
    
    def __init__(self, events, header, ID, ms_noise_det = 20, mcs_HDBSCAN = [10,10,15,20], 
                 ms_HDBSCAN = [20,40,30,60], cse_HDBSCAN = [0.005,0.005,0.005,0.005], not_noise_threshold = 2):
        self.events_sources = events
        self.n_events_bnd = len(events)
        self.range_DETX = np.ptp(events['DETX_norm'])
        self.range_DETY = np.ptp(events['DETY_norm'])
        self.events_noise = Table()
        observation_ID = header['OBS_ID']
        self.observation_ID = observation_ID
        self.ID = ID
        self.ms_noise_det = ms_noise_det
        self.mcs_HDBSCAN = mcs_HDBSCAN
        self.ms_HDBSCAN = ms_HDBSCAN
        self.cse_HDBSCAN = cse_HDBSCAN
        self.not_noise_threshold = not_noise_threshold
        # make a directory for the upcoming results, if not already existend
        if not os.path.isdir(os.path.join('results', "results_ID_" + str(ID) + '_' + str(observation_ID) + '/')):
            os.mkdir(os.path.join('results', "results_ID_" + str(ID) + '_' + str(observation_ID) + '/'))
    
    """
    Method to get the important variables of this class to asses
    Output: dictionary of the main variables
    """
    def get_values(self):
        mcs_HDBSCAN = self.mcs_HDBSCAN
        ms_HDBSCAN = self.ms_HDBSCAN
        cse_HDBSCAN = self.cse_HDBSCAN
        var_dict = {'ms_noise_det': [self.ms_noise_det],
                    'eps_noise_det': [self.eps_noise_det],
                    'n_events_and': [self.n_events_and],
                    'n_noise_and': [self.n_noise_and]}
        
        for i in range(len(mcs_HDBSCAN)):
            var_dict_2 = {'mcs_' + str(i+1): [mcs_HDBSCAN[i]],
                          'ms_' + str(i+1): [ms_HDBSCAN[i]],
                          'cse_' + str(i+1): [cse_HDBSCAN[i]]}
            var_dict.update(var_dict_2)
        
        var_dict_3 ={'not_noise_threshold': [self.not_noise_threshold],
                    'n_events_acc': [self.n_events_acc],
                    'n_noise_acc': [self.n_noise_acc],
                    'n_clusters': [len(self.final_clusters_HDBSCAN)],
                    }
        var_dict.update(var_dict_3)
        return var_dict
    
    """
    Noise detection with DBSCAN
    """
    def noise_detection(self):
        self.step_counter = 3
        
        events = self.events_sources
        n_events = self.n_events_bnd
        ms_noise_det = self.ms_noise_det

        range_DETX = self.range_DETX 
        range_DETY = self.range_DETY 
        # number of bins of 150.000 to 300.000 events possible
        n_bins = int(np.floor(n_events/100000))

        ##################################################################################
        ################### running DBSCAN on all the events could take to long
        ################### split into bins if there are to many events
        ##################################################################################
        if n_bins > 1:
            # sort for time so the bins will be equally distributed in position and energy
            events.sort('TIME_norm')
            
            # calculate the events per bin
            events_per_bin = int(np.ceil(n_events/n_bins))
            
            # split into bins and store all of them in a list
            events_for_noise_detection = []
            for bin_counter in range(n_bins):
                start_index = bin_counter * events_per_bin
                end_index = (bin_counter + 1) * events_per_bin
                events_for_noise_detection.append(events[start_index: end_index])
            
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #
            #   Multi Processing
            #
            # events_for_noise_detection_len = len(events_for_noise_detection)
            # threads = events_for_noise_detection_len
            # m = Multiprocessing(threads=threads, function=self.noise_detection_thread, input_list=events_for_noise_detection)
            # noise_detection_result = m.multi()
            # events = noise_detection_result.copy()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #
            #   No parallelization
            #
            events_table = Table()
            for inx,event_bin in enumerate(events_for_noise_detection):
                print("   -> " + str(inx + 1) + ". run of DBSCAN started.")
                
                # get events into right shape for the DBSCAN
                events_for_DBSCAN = event_bin['DETX_norm', 'DETY_norm']
                event_list = []
                for i in range(len(events_for_DBSCAN)):
                    event_list.append(list(events_for_DBSCAN[i]))   
                
                # set the hyper parameters
                n_events_bin = len(event_bin)
                eps_noise_det = np.sqrt((range_DETX*range_DETY*ms_noise_det)/(np.pi*n_events_bin))
                self.eps_noise_det = eps_noise_det
                
                # perform the clustering
                clustering = DBSCAN(min_samples = ms_noise_det, eps = eps_noise_det, n_jobs = -1)
                clustering.fit(event_list)
                
                # save the labels
                event_bin["DBSCAN_cluster"] = clustering.labels_
                
                events_table = vstack([events_table, event_bin])
                
                print("   -> " + str(inx + 1) + ". run of DBSCAN finished.")
            events = events_table.copy()
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        ##################################################################################
        ################### if the number of events is low
        ################### no splitting is needed
        ##################################################################################
        else:
            # get events into right shape for the DBSCAN
            events_for_DBSCAN = events['DETX_norm', 'DETY_norm']
            event_list = []
            for i in range(len(events_for_DBSCAN)):
                event_list.append(list(events_for_DBSCAN[i]))   
            
            # set the hyper parameters
            eps_noise_det = np.sqrt((range_DETX*range_DETY*ms_noise_det)/(np.pi*n_events))
            self.eps_noise_det = eps_noise_det
            
            # perform the clustering
            clustering = DBSCAN(min_samples = ms_noise_det, eps = eps_noise_det, n_jobs = -1)
            clustering.fit(event_list)
            
            # save the labels
            events["DBSCAN_cluster"] = clustering.labels_
        
        # divide into noise and source events
        noise_mask = (events["DBSCAN_cluster"] == -1)
        self.events_noise = events[noise_mask]
        self.events_sources = events[~noise_mask]
        self.n_events_and = len(self.events_sources)
        self.n_noise_and = len(self.events_noise)
        
        print("  - Finished the Noise Detection.")
    
    """
    Parallel thread for noise detection
    """
    def noise_detection_thread(self, input=[]):
        event_bin = input
        
        range_DETX = self.range_DETX 
        range_DETY = self.range_DETY 
        ms_noise_det = self.ms_noise_det
        
        l = len(event_bin)
        print(f'\t- starting with event bin no # event_bin len = {l}')
        
        # get events into right shape for the DBSCAN
        events_for_DBSCAN = event_bin['DETX_norm', 'DETY_norm']
        event_list = []
        for i in range(len(events_for_DBSCAN)):
            event_list.append(list(events_for_DBSCAN[i]))   
        
        # set the hyper parameters
        n_events_bin = len(event_bin)
        eps_noise_det = np.sqrt((range_DETX*range_DETY*ms_noise_det)/(np.pi*n_events_bin))
        self.eps_noise_det = eps_noise_det
        
        # perform the clustering
        clustering = DBSCAN(min_samples = ms_noise_det, eps = eps_noise_det, n_jobs = -1)
        clustering.fit(event_list)
        
        # save the labels
        event_bin["DBSCAN_cluster"] = clustering.labels_
        
        return event_bin
        
    
    """
    Helping method for the ensemble.
    Input:  events_samples = events that have to be clustered
            events_oop = events that are not used for clustering
            HDBSCAN_counter = run no. of HDBSCAN
    Output: events_samples = combination of events_samples and events_oob, all clustered
    """
    def perform_HDBSCAN(self, events_samples, events_oob, HDBSCAN_counter):
        mcs = int(self.mcs_HDBSCAN[HDBSCAN_counter - 1])
        ms = int(self.ms_HDBSCAN[HDBSCAN_counter - 1])
        cse = int(self.cse_HDBSCAN[HDBSCAN_counter - 1])
        cluster_str = "cluster_" + str(HDBSCAN_counter)
        probs_str = "probs_" + str(HDBSCAN_counter)
        
        thin_out = len(events_oob) > 0
        
        events_for_HDBSCAN = events_samples['TIME_norm','DETX_norm','DETY_norm']
        event_list = []
        for i in range(len(events_for_HDBSCAN)):
            event_list.append(list(events_for_HDBSCAN[i]))  
         
        # perform the clustering
        clustering = hdbscan.HDBSCAN(min_samples = ms, 
                                     min_cluster_size = mcs, 
                                     cluster_selection_epsilon = cse,
                                     metric = 'euclidean',
                                     prediction_data = True, # soft clustering
                                     cluster_selection_method = 'leaf' 
                                     )
        clustering.fit(event_list)
        
        # apply labels to data
        events_samples[cluster_str] = clustering.labels_
        events_samples[probs_str] = clustering.probabilities_
        
        if thin_out:
            # transform the oob events into the right format
            events_for_oob = events_oob['TIME_norm','DETX_norm', 'DETY_norm']
            event_list_oob = []
            for i in range(len(events_for_oob)):
                event_list_oob.append(list(events_for_oob[i]))
            
            labels, probs = hdbscan.approximate_predict(clustering, event_list_oob)
            events_oob[cluster_str] = labels
            events_oob[probs_str] = probs
        
            # put together the HDBSCAN samples and the oob samples
            events_samples = vstack([events_samples, events_oob])
        
        return events_samples
    
    """
    Ensemble of multiple HDBSCAN++
    """
    def ensemble(self):
        self.step_counter = 4
        print("  - Starting the Ensemble.")
        events_sources = self.events_sources
        events_noise = self.events_noise
        mcs_HDBSCAN = self.mcs_HDBSCAN
        not_noise_threshold = self.not_noise_threshold

        # set up some columns
        events_sources['count_not_noise'] = 0.0
        events_noise['count_not_noise'] = 0.0
        events_sources['sum_probs'] = 0 # will be deleted at the end

        ##################################################################################
        ################### run the clustering multiple times
        ##################################################################################

        n_HDBSCAN = len(mcs_HDBSCAN)

        for HDBSCAN_counter in range(1, n_HDBSCAN + 1):
            cluster_str = "cluster_" + str(HDBSCAN_counter)
            probs_str = "probs_" + str(HDBSCAN_counter)
            events_noise[cluster_str] = -1
            events_noise[probs_str] = 0.0

            # pick a random sample
            events_samples, events_oob = pick_sample(events_sources, 200000)
            
            # perform HDBSCAN
            events_sources = self.perform_HDBSCAN(events_samples, events_oob, HDBSCAN_counter)
            
            print("    -> Finished " + str(HDBSCAN_counter) + ". run of HDBSCAN++.")
            
            ##################################################################################
            ########## columns to calculate the final probability of importance
            ##################################################################################
            events_sources['count_not_noise'] = events_sources['count_not_noise'] + np.ceil(events_sources[probs_str])
            events_sources['sum_probs'] = events_sources['sum_probs'] + events_sources[probs_str]

            ##################################################################################
            ########## generate a plot for each run
            ##################################################################################
            self.events_sources = events_sources
            # self.plot_HDBSCAN(HDBSCAN_counter)
            
        ##################################################################################
        ########## calculate the final probability of importance
        ##################################################################################

        # SC for soft clustering -> needed for the pairing
        events_sources['probs_SC_total'] = np.divide(events_sources['sum_probs'], events_sources['count_not_noise'], where = (events_sources['count_not_noise']!=0))
        events_noise['probs_SC_total'] = 0.0
        del events_sources['sum_probs']

        events_sources['probs_not_noise'] = events_sources['count_not_noise']/n_HDBSCAN
        events_noise['probs_not_noise'] = 0.0

        ##################################################################################
        ################### update the cluster and noise events
        ##################################################################################
        consensus_threshold = not_noise_threshold/n_HDBSCAN

        consensus_mask = (events_sources["probs_not_noise"] > consensus_threshold)
        events_HDBSCAN_sources = events_sources[consensus_mask]
        events_HDBSCAN_noise = events_sources[~consensus_mask]

        self.events_sources = events_HDBSCAN_sources.copy()
        self.events_noise = vstack([events_noise, events_HDBSCAN_noise])

    """
    Consensus clustering
    Combining the different runs of HDBSCAN in the ensemble.
    """
    def consensus_clustering(self):
        if self.step_counter <= 3:
            self.ensemble()
        self.step_counter = 5

        events_sources = self.events_sources 
        events_noise = self.events_noise

        n_HDBSCAN = len(self.mcs_HDBSCAN)

        ##################################################################################
        ################### get the events into the right shape
        ##################################################################################
        clusterings = []
        for HDBSCAN_counter in range(1, n_HDBSCAN + 1):
            cluster_str = "cluster_" + str(HDBSCAN_counter)
            clusterings.append(cluster_str)

        # change -1 cluster into numbers for single clusters, so that two events clustered as noise have different labels and not all -1
        # so that the hamming distance will be = 1 for two noise points
        for cluster_str in clusterings:
            min_noise_cluster = max(events_sources[cluster_str]) + 1
            counter = min_noise_cluster
            for event in events_sources:
                if event[cluster_str] == -1:
                    event[cluster_str] = counter
                    counter = counter + 1

        events_for_consensus = events_sources[clusterings]
        event_list = []
        for i in range(len(events_for_consensus)):
            event_list.append(list(events_for_consensus[i])) 

        ##################################################################################
        ################### perform HDBSCAN with hamming distance
        ##################################################################################

        mcs_consensus = int(min(self.mcs_HDBSCAN))
        clustering = hdbscan.HDBSCAN(min_cluster_size = mcs_consensus, 
                                     metric = 'hamming',
                                     prediction_data = True)
        clustering.fit(event_list)

        events_sources['final_cluster'] = clustering.labels_
        events_sources['final_probs'] = clustering.probabilities_
        events_noise['final_cluster'] = -1
        events_noise['final_probs'] = 0.0

        ##################################################################################
        ################### change the noise labels back to -1
        ##################################################################################
        for inx, cluster_str in enumerate(clusterings):
            probs_str = 'probs_' + str(inx + 1)
            for event in events_noise:
                if event[probs_str] == 0.0:
                    event[cluster_str] = -1

        #################################################################################
        ######### calculate some statistics of the clusters(needed for the next steps)
        #################################################################################
        clusters_HDBSCAN = Table(names = ['cluster', 'min_TIME', 'max_TIME', 'cluster_range_TIME', 
                                          'min_DETX', 'max_DETX', 'mean_DETX', 'mean_DETX_scaled', 
                                          'min_DETY', 'max_DETY', 'mean_DETY', 'mean_DETY_scaled'])
        
        # number of clusters in the consenus clustering
        n_cons_clusters = max(events_sources["final_cluster"]) + 1
        for cluster in range(-1, n_cons_clusters):
            # extract the clustered events
            cluster_mask = (events_sources["final_cluster"] == cluster)
            events_cluster = events_sources[cluster_mask]
            
            # calculate the variables of interest
            min_TIME = min(events_cluster["TIME"])
            max_TIME = max(events_cluster["TIME"])
            cluster_range_TIME = max_TIME - min_TIME
            min_DETX = min(events_cluster["DETX"])
            max_DETX = max(events_cluster["DETX"])
            mean_DETX = np.mean(events_cluster["DETX"])
            mean_DETX_scaled = np.mean(events_cluster["DETX_norm"])
            min_DETY = min(events_cluster["DETY"])
            max_DETY = max(events_cluster["DETY"])
            mean_DETY = np.mean(events_cluster["DETY"])
            mean_DETY_scaled = np.mean(events_cluster["DETY_norm"])
            
            # add to the table
            clusters_HDBSCAN.add_row((cluster, min_TIME, max_TIME, cluster_range_TIME,
                                      min_DETX, max_DETX, mean_DETX, mean_DETX_scaled, 
                                      min_DETY, max_DETY, mean_DETY, mean_DETY_scaled))

        ##################################################################################
        ################### update the cluster and noise events
        ##################################################################################
        noise_mask = (events_sources["final_cluster"] == -1)
        events_consensus_sources = events_sources[~noise_mask]
        events_consensus_noise = events_sources[noise_mask]

        self.events_sources = events_consensus_sources.copy()
        self.events_noise = vstack([events_noise, events_consensus_noise])
        self.clusters_HDBSCAN = clusters_HDBSCAN
        self.n_events_acc = len(self.events_sources)
        self.n_noise_acc = len(self.events_noise)
        
        print("  - Finished the Consensus Clustering.")
    
    """
    Pairing of the clusters after the consensus clustering.
    If the consensus clustering is not yet performed, it will be performed first.
    Pair the clusters accoring to their position using nearest neighbor criteria.
    """
    def pairing(self):
        if self.step_counter <= 4:
            self.consensus_clustering()
        self.step_counter = 6
        
        events_sources = self.events_sources
        events_noise = self.events_noise
        clusters_HDBSCAN = self.clusters_HDBSCAN
        
        #################################################################################
        ######### threshold to disregard events that are undecided for a cluster
        #################################################################################
        probs = events_sources['probs_SC_total']
        median_probs = np.median(probs)
        knn_probability_threshold = median_probs

        #################################################################################
        ######### get the non noise events and put it in right shape
        #################################################################################
        # noise points and points that are very likely to be near another cluster are not used for the pairing
        knn_mask = events_sources['final_probs'] >= knn_probability_threshold
        events_knn = events_sources[knn_mask]['DETX', 'DETY', 'probs_SC_total', 'final_cluster']

        # modify the table
        events_knn.rename_column('probs_SC_total', 'orig_probs')
        events_knn.rename_column('final_cluster', 'orig_cluster')
        events_knn['knn_cluster'] = -1
        events_knn['knn_probs'] = 0.0

        # put events into right shape
        events_for_knn = events_knn['DETX', 'DETY']
        event_list = []
        for i in range(len(events_for_knn)):
            event_list.append(list(events_for_knn['DETX', 'DETY'][i]))

        #################################################################################
        ######### perform Nearest Neighbor (get the first nearest neighbor of each event)
        #################################################################################

        n_knn = 2 #1st nearest neighbor is the element itself
        neigh = NearestNeighbors(n_neighbors = n_knn, metric = 'euclidean')
        nbrs = neigh.fit(event_list)
        distances, indices = nbrs.kneighbors(event_list)

        # get the indices and the distances of the first nearest neighbor of each event
        events_knn['knn_index'] = indices[:,1]
        events_knn['knn_distance'] = distances[:,1]

        # the position is not needed anymore
        events_knn.remove_column('DETX')
        events_knn.remove_column('DETY')

        # get the cluster and the probability of the first KNN
        for j in range(len(events_knn)):
            event_index = events_knn['knn_index'][j]
            events_knn['knn_cluster'][j] = events_knn['orig_cluster'][event_index]
            events_knn['knn_probs'][j] = events_knn['orig_probs'][event_index]
        #################################################################################
        ######### create a dictionary of all the knn clusters associated to a original cluster
        #################################################################################

        # group by the original cluster
        grouping = events_knn.group_by('orig_cluster')
        
        pairing_dict = {}
        n_orig_clusters = max(events_knn['orig_cluster']) + 1
        for cluster in range(n_orig_clusters):
            # group by the knn clusters
            grouping2 = grouping.groups[cluster].group_by('knn_cluster')
            keys = grouping2.groups.keys
            indices = grouping2.groups.indices
            
            pairing_dict[cluster] = list(keys['knn_cluster'])
            
        #################################################################################
        ######### get a list of lists of the clusters belonging together
        #################################################################################

        # create a list of original clusters 
        # we will delete from that list the clusters, that are part of another cluster
        key_list = list(range(n_orig_clusters))

        # start a list of lists for the paired clusters
        result = []

        # go through the list of keys and iteratively put the clusters together that belong together
        for key in key_list:
            # if the cluster is not assigned to itself, add it to the list
            if key not in pairing_dict[key]:
                pairing_dict[key].append(key)
            # go through every value in the according dictionary and assign all assigned clusters to this one as well
            for next_key in pairing_dict[key]:
                if next_key != key:
                    for value in pairing_dict[next_key]:
                        if value not in pairing_dict[key]:
                            pairing_dict[key].append(value)
                    # key_list.remove(next_key)
            result_cluster = [int(x) for x in pairing_dict[key]]
            result.append(result_cluster)
            
        # put together the subsets
        for m in result[:]:
            for n in result[:]:
                if set(m).issubset(set(n)) and m != n:
                    result.remove(m)
                    break

        # remove duplicates
        result.sort()
        final_result = list(result for result,_ in itertools.groupby(result))
            
        #################################################################################
        ######### modify clusters_HDBSCAN
        ######### assign a new cluster 
        #################################################################################  
            
        # assign the pairs to the same cluster
        clusters_HDBSCAN.sort('cluster')
        clusters_HDBSCAN['new_cluster'] = -1

        # the index of the list will be the new cluster (starting with 0)
        for index, clusters in enumerate(final_result):
            for cluster in clusters:
                clusters_HDBSCAN['new_cluster'][cluster + 1] = index    

        #################################################################################
        ######### assign the new clusters to the list of events (events_sources)
        #################################################################################  

        events_sources['new_cluster'] = -1
        events_noise['new_cluster'] = -1

        # leave out the noise (they are already set to -1)
        for i in range(1, len(clusters_HDBSCAN)):
            old_cluster = clusters_HDBSCAN['cluster'][i]
            new_cluster = clusters_HDBSCAN['new_cluster'][i]
            for j in range(len(events_sources)):
                if events_sources['final_cluster'][j] == old_cluster:
                    events_sources['new_cluster'][j] = new_cluster

        #################################################################################
        ######### get a table of the final sources
        ################################################################################# 
        final_clusters_HDBSCAN = Table(names = ['cluster', 'min_TIME', 'max_TIME', 'cluster_range_TIME',
                                                'min_X', 'max_X', 'mean_X', 'median_X', 
                                                'min_Y', 'max_Y', 'mean_Y', 'median_Y', 'count_events'])

        n_new_clusters = max(events_sources["new_cluster"]) + 1

        for cluster in range(n_new_clusters):
            cluster_mask = (events_sources["new_cluster"] == cluster)
            events_cluster = events_sources[cluster_mask]
            
            min_TIME = min(events_cluster["TIME"])
            max_TIME = max(events_cluster["TIME"])
            cluster_range_TIME = max_TIME - min_TIME
            min_X = min(events_cluster["X"])
            max_X = max(events_cluster["X"])
            mean_X = np.mean(events_cluster["X"])
            median_X = np.median(events_cluster["X"])
            min_Y = min(events_cluster["Y"])
            max_Y = max(events_cluster["Y"])
            mean_Y = np.mean(events_cluster["Y"])
            median_Y = np.median(events_cluster["Y"])
            count = len(events_cluster)
            
            final_clusters_HDBSCAN.add_row((cluster, min_TIME, max_TIME, cluster_range_TIME,
                                           min_X, max_X, mean_X, median_X, 
                                           min_Y, max_Y, mean_Y, median_Y, count))
        
        self.clusters_HDBSCAN = clusters_HDBSCAN
        self.final_clusters_HDBSCAN = final_clusters_HDBSCAN
        self.events_sources = events_sources
        self.events_noise = events_noise
        
        print("  - Finished the Pairing of the final clusters.")
     
    """
    Plot of the noise detection results
    First plots: 2D plot (clusters in random color)
    Second plot: 2D plot (oonly noise)
    """
    def plot_noise_detection(self):
        observation_ID = self.observation_ID
        step_counter = self.step_counter
        ID = self.ID
        events_sources = self.events_sources
        events_noise = self.events_noise
        
        fig = plt.figure(figsize=[12,16])
        # =============
        # First subplot
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title("Found clusters of observation " + str(observation_ID) + " by 2D DBSCAN", fontsize = 14)
        ax.set_box_aspect(aspect = (1))
        ax.scatter(events_noise["DETX_norm"], events_noise["DETY_norm"], c = 'grey', s = 0.004)
        ax.scatter(events_sources["DETX_norm"], events_sources["DETY_norm"], c = events_sources["DBSCAN_cluster"], cmap = 'prism', s = 0.004)
        ax.set_xlabel("DETY_norm")
        ax.set_ylabel("DETX_norm")
        # ==============
        # Second subplot
        ax = fig.add_subplot(2, 1, 2)
        ax.set_title("Found noise of observation " + str(observation_ID) + " by 2D DBSCAN", fontsize = 14)
        ax.scatter(events_noise["DETX_norm"], events_noise["DETY_norm"], c = 'grey', s = 0.004)
        ax.set_box_aspect(aspect = (1))
        ax.set_xlabel("DETX_norm")
        ax.set_ylabel("DETY_norm")

        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_2DBSCAN_Noise_Detection.jpg", dpi = 150)
        plt.close()
    
    """
    Plot of the results of one individual HDBSCAN in the ensemble
    First plots: 3D plot
    Second plot: 2D plot
    Input: number of the run no. in the ensemble
    """
    def plot_HDBSCAN(self, HDBSCAN_counter):
        mcs_HDBSCAN = self.mcs_HDBSCAN
        if HDBSCAN_counter > len(mcs_HDBSCAN) or HDBSCAN_counter < 1:
            print("Invalid input: HDBSCAN_counter has to be smaller or equal to " + str(len(mcs_HDBSCAN)) + " and greater than 0.")
            return
        
        mcs = mcs_HDBSCAN[HDBSCAN_counter - 1]
        ms = self.ms_HDBSCAN[HDBSCAN_counter - 1]
        cse = self.ms_HDBSCAN[HDBSCAN_counter - 1]
        events_sources = self.events_sources
        ID = self.ID
        observation_ID = self.observation_ID
        step_counter = self.step_counter
        cluster_str = "cluster_" + str(HDBSCAN_counter)
        
        # extract the clustered events
        cluster_mask = (events_sources[cluster_str] != -1)
        events_cluster = events_sources[cluster_mask]
        events_background = events_sources[~cluster_mask]
        
        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=[12,16])
        
        # =============
        # First subplot
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.set_title("Found sources over Time by 3HDBSCAN++ in run no. " + str(HDBSCAN_counter) + "\n mcs = " + str(mcs) + ", ms = " + str(ms) + ", cse = " + str(cse), fontsize = 18)
        ax.scatter(xs = events_cluster["TIME_norm"], 
                    ys = events_cluster["DETY_norm"], 
                    zs = events_cluster["DETX_norm"], 
                    c = events_cluster[cluster_str], 
                    s = 0.5,
                    cmap = 'prism')
        ax.set_xlim3d(min(events_sources["TIME_norm"]), max(events_sources["TIME_norm"]))
        ax.set_ylim3d(min(events_sources["DETY_norm"]), max(events_sources["DETY_norm"]))
        ax.set_zlim3d(min(events_sources["DETX_norm"]), max(events_sources["DETX_norm"]))
        ax.set_box_aspect(aspect = (4,1,1))
        ax.set_xlabel("TIME_norm")
        ax.set_ylabel("DETY_norm")
        ax.set_zlabel("DETX_norm")
        
        # ==============
        # Second subplot
        ax = fig.add_subplot(2, 1, 2)
        ax.scatter(events_background["DETX_norm"], events_background["DETY_norm"], c = 'grey', s = 0.004)
        ax.scatter(events_cluster["DETX_norm"], events_cluster["DETY_norm"], c = events_cluster[cluster_str], cmap = 'prism', s = 0.004)
        ax.set_box_aspect(aspect = (1))
        ax.set_xlabel("DETX_norm")
        ax.set_ylabel("DETY_norm")

        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_3HDBSCAN++_run" + str(HDBSCAN_counter) + ".jpg", dpi = 150)
        plt.close()
    
    """
    Plot of the final clustering after pairing
    First plots: 3D plot
    Second plot: 2D plot
    """
    def plot_pairing(self):
        observation_ID = self.observation_ID
        step_counter = self.step_counter
        ID = self.ID
        events_sources = self.events_sources
        events_noise = self.events_noise

        fig = plt.figure(figsize=[12,16])
        # =============
        # First subplot
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        ax.set_title("Found Sources with 3HDBSCAN++ after pairing", fontsize = 18)
        ax.scatter(xs = events_sources["TIME_norm"], 
                    ys = events_sources["DETY_norm"], 
                    zs = events_sources["DETX_norm"], 
                    c = events_sources["new_cluster"], 
                    s = 0.5,
                    cmap = 'prism')

        ax.set_xlim3d(min(events_sources["TIME_norm"]), max(events_sources["TIME_norm"]))
        ax.set_ylim3d(min(events_sources["DETY_norm"]), max(events_sources["DETY_norm"]))
        ax.set_zlim3d(min(events_sources["DETX_norm"]), max(events_sources["DETX_norm"]))
        ax.set_box_aspect(aspect = (4,1,1))
        ax.set_xlabel("TIME_norm")
        ax.set_ylabel("DETY_norm")
        ax.set_zlabel("DETX_norm")

        # ==============
        # Second subplot
        ax = fig.add_subplot(2, 1, 2)
        plt.scatter(events_noise["DETX_norm"], events_noise["DETY_norm"], c = 'grey', s = 0.004)
        plt.scatter(events_sources["DETX_norm"], events_sources["DETY_norm"], c = events_sources["new_cluster"], cmap = 'prism', s = 0.004)

        ax.set_box_aspect(aspect = (1))
        ax.set_xlabel("DETX_norm")
        ax.set_ylabel("DETY_norm")

        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_clustering_after_pairing.jpg", dpi = 150)
        plt.close()