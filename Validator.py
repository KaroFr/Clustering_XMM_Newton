# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:52:01 2022

@author: Karolin.Frohnapfel
"""
from Helpers import load_sources

from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.spatial import distance
import os

  
"""
Class for the validation of the clustering.
"""      
class Validator:
    step_counter = 7
    n_SOC_sources = 0
    n_matches = 0
    match_rate = 0.0
    detection_rate = 0.0
    F_score = 0.0
    src_SOC_median = Table()
    src_clus_median = Table()
    
    def __init__(self, events, header, final_clusters_HDBSCAN, ID, distance_threshold = 300):
        self.events = events
        self.header = header
        observation_ID = header['OBS_ID']
        self.observation_ID = observation_ID
        self.ID = ID
        self.final_clusters_HDBSCAN = final_clusters_HDBSCAN
        cluster_cols = [c for c in events.colnames if 'cluster_' in c]
        self.n_HDBSCAN = len(cluster_cols)
        self.distance_threshold = distance_threshold
        # make a directory for the upcoming results, if not already existend
        if not os.path.isdir(os.path.join('results', "results_ID_" + str(ID) + '_' + str(observation_ID) + '/')):
            os.mkdir(os.path.join('results', "results_ID_" + str(ID) + '_' + str(observation_ID) + '/'))
    
    """
    Method to get the important variables of this class to asses
    Output: dictionary of the main variables
    """
    def get_values(self):
        var_dict = {'n_SOC_sources': [self.n_SOC_sources],
                    'n_matches': [self.n_matches],
                    'match_rate': [self.match_rate],
                    'detection_rate': [self.detection_rate],
                    'F_score': [self.F_score]}
        return var_dict
     
    """
    Cross match validation.
    Check for matches with the sources found by the SOC pipeline.
    Calculate match-rate, detection-rate and F-Score
    """
    def cross_match(self):
        observation_ID = self.observation_ID
        events = self.events
        header = self.header
        final_clusters_HDBSCAN = self.final_clusters_HDBSCAN
        n_HDBSCAN = self.n_HDBSCAN
        distance_threshold = self.distance_threshold
        
        srclist = load_sources(observation_ID)
        #################################################################################
        ######### preperation for the transformation into Sky Coordinates
        ################################################################################# 
        w = WCS(header)
        w.wcs.ctype = [header['REFXCTYP'], header['REFYCTYP']]
        w.wcs.crval = [header['REFXCRVL'], header['REFYCRVL']]
        w.wcs.cdelt = np.array([header['REFXCDLT'], header['REFYCDLT']])
        w.wcs.crpix = [header['REFXCRPX'], header['REFYCRPX']]
        w.wcs.cunit = [header['REFXCUNI'], header['REFYCUNI']]

        #################################################################################
        ######### get the detector coordinates of the SOC and clustering sources 
        ######### use the median to define a source
        ################################################################################# 
        final_clusters_HDBSCAN.sort('cluster')
        src_SOC_median = Table(w.world_to_pixel(SkyCoord(ra=srclist.RA *u.degree, dec=srclist.DEC *u.degree)), names = ['SOC_X', 'SOC_Y'])
        src_clus_median = final_clusters_HDBSCAN['median_X','median_Y']

        # number of sources detected by SOC and clustering pipeline 
        n_SOC = len(src_SOC_median)
        self.n_SOC_sources = n_SOC
        n_clus = len(src_clus_median) 

        #################################################################################
        ######### get the matches
        ################################################################################# 

        # get the SOC and clustering sources into right shape for distance matrix
        src_SOC_list = []
        for i in range(n_SOC):
            src_SOC_list.append([src_SOC_median['SOC_X'][i],src_SOC_median['SOC_Y'][i]])

        src_clus_list = []
        for i in range(n_clus):
            src_clus_list.append([src_clus_median['median_X'][i], src_clus_median['median_Y'][i]])

        # euclidean distance matrix of the SOC and clustering sources
        dist = distance.cdist(src_SOC_list, src_clus_list, metric = 'euclidean')

        # find the clusters below a threshold
        cluster_SOC, cluster_clus = np.where(np.logical_and(dist < distance_threshold, dist > 0))
        pairs_table = Table([cluster_SOC + 1, cluster_clus], names = ['SOC_cluster', 'clus_cluster'])
        pairs_table['distance'] = 0
        for index,pair in enumerate(pairs_table):
            SOC_cluster = pair['SOC_cluster'] - 1
            clus_cluster = pair['clus_cluster']
            pairs_table['distance'][index] = dist[SOC_cluster][clus_cluster]

        # number of matches
        n_matches = len(np.unique(pairs_table['clus_cluster']))
        self.n_matches = n_matches

        #################################################################################
        ######### add the matches to the final_clusters_HDBSCAN
        ######### calculate the inter cluster distance for each cluster
        ################################################################################# 
        final_clusters_HDBSCAN['match'] = 0
        final_clusters_HDBSCAN['inter_distance'] = 0.0
        final_clusters_HDBSCAN.sort('cluster')

        for index, cluster in enumerate(list(final_clusters_HDBSCAN['cluster'])):
            if cluster in list(pairs_table['clus_cluster']):
                final_clusters_HDBSCAN['match'][index] = 1
            if cluster != -1:
                dis_mask = (events['new_cluster'] == cluster)
                events_for_dis = events[dis_mask]['DETX','DETY']

                dis_list = []
                for i in range(len(events_for_dis)):
                    dis_list.append([events_for_dis['DETX'][i], events_for_dis['DETY'][i]])

                inter_distances = distance.pdist(dis_list)
                mean_dist = np.mean(inter_distances)
                final_clusters_HDBSCAN['inter_distance'][index] = mean_dist

        #################################################################################
        ######### count how many times a source has been detected
        ################################################################################# 
        final_clusters_HDBSCAN['count_detected'] = 0

        for HDBSCAN_counter in range(1, n_HDBSCAN + 1):
            cluster_str = "cluster_" + str(HDBSCAN_counter)
            
            final_clusters_HDBSCAN[cluster_str] = 0

            # get the not noise events
            cluster_mask = (events[cluster_str] != -1)
            events_cluster = events[cluster_str,'X','Y'][cluster_mask]
            
            # group by cluster and calculate the median
            cluster_groups = events_cluster.group_by(cluster_str)
            source_centers = cluster_groups.groups.aggregate(np.median)

            # get into right shape for distance calculations
            source_centers_list = []
            for i in range(len(source_centers)):
                source_centers_list.append([source_centers['X'][i], source_centers['Y'][i]])
            
            # euclidean distance matrix of the final and step sources
            dist = distance.cdist(src_clus_list, source_centers_list, metric = 'euclidean')
            
            # find the sources at the same position using a threshold
            found_sources, _ = np.where(np.logical_and(dist < 300, dist > 0))

            # 1 = source found; 0 = source not found
            for index, cluster in enumerate(final_clusters_HDBSCAN['cluster']):
                if cluster in found_sources:
                    final_clusters_HDBSCAN[cluster_str][index] = 1
            
            # set the count up for those sources detected in this step
            final_clusters_HDBSCAN['count_detected'] = final_clusters_HDBSCAN['count_detected'] + final_clusters_HDBSCAN[cluster_str]

        # calculate the probability that a source is detected
        final_clusters_HDBSCAN['detected_prob'] = final_clusters_HDBSCAN['count_detected']/n_HDBSCAN
                
        #################################################################################
        ######### calculate some other probabilities of the found clusters to be clusters
        ################################################################################# 
        final_clusters_HDBSCAN.sort('cluster')
        # a low count indicates a wrong cluster
        max_count = max(final_clusters_HDBSCAN['count_events'][1:])
        final_clusters_HDBSCAN['count_prob'] = final_clusters_HDBSCAN['count_events']/max_count
        final_clusters_HDBSCAN['count_prob'][0] = 0.0

        # a high mean inter cluster distance indicates a wrong cluster
        max_inter_dis = max(final_clusters_HDBSCAN['inter_distance'][1:])
        final_clusters_HDBSCAN['inter_distance_prob'] = 1 - final_clusters_HDBSCAN['inter_distance']/max_inter_dis
        final_clusters_HDBSCAN['inter_distance_prob'][0] = 0.0

        # mean probability to be a source
        final_clusters_HDBSCAN['source_likelihood'] = (final_clusters_HDBSCAN['count_prob'] + final_clusters_HDBSCAN['inter_distance_prob'] + final_clusters_HDBSCAN['detected_prob'])/3

        self.final_clusters_HDBSCAN = final_clusters_HDBSCAN
        #################################################################################
        ######### get the validation rates
        ################################################################################# 
        match_rate = n_matches/n_clus
        detection_rate = n_matches/n_SOC
        self.F_score = 2*match_rate*detection_rate/(match_rate + detection_rate)
        self.match_rate = match_rate
        self.detection_rate = detection_rate
        
        self.src_SOC_median = src_SOC_median
        self.src_clus_median = src_clus_median
        self.pairs_table = pairs_table
        
        print("  - Finished the Cross Match Validation.")

    """
    Plot of the found sources by this pipeline and by the SOC pipeline detection results
    Top right plot:     all sources
    Top left plot:      Falsly detected sources
    Bottom right plot:  not detected sources
    Bottom left plot:   matches
    """
    def plot_cross_match(self):
        match_rate = self.match_rate
        detection_rate = self.detection_rate
        F_score = self.F_score
        
        src_SOC_median = self.src_SOC_median
        pairs_table = self.pairs_table
        
        ID = self.ID
        step_counter = self.step_counter
        observation_ID = self.observation_ID
        
        final_clusters_HDBSCAN = self.final_clusters_HDBSCAN
        
        events = self.events
        events_noise = events[events['new_cluster'] == -1]
        events_sources = events[events['new_cluster'] != -1]

        #################################################################################
        ######### plot the sources
        ################################################################################# 

        # divide clusters into matches and no matches
        match_mask = (final_clusters_HDBSCAN['match'] == 1)
        matches = final_clusters_HDBSCAN[match_mask]
        not_matches = final_clusters_HDBSCAN[~match_mask]

        fig, ax = plt.subplots(2,2, figsize=(12, 12))
        plt.suptitle("Clusters found by 3HDBSCAN++\n Detection Rate = " + str(round(detection_rate,4)) + ", Match Rate = " + str(round(match_rate,4)) + ", F-Score = " + str(round(F_score,4)), fontsize = 14)

        # top left plot: all matches and mismatches
        ax[0,0].scatter(events_noise["X"], events_noise["Y"], c = 'grey', s = 0.004)
        ax[0,0].scatter(events_sources["X"], events_sources["Y"], c = 'red', s = 0.04)
        ax[0,0].scatter(matches["median_X"], matches["median_Y"], s = 300, facecolors = 'none', edgecolors = 'g')
        ax[0,0].scatter(not_matches["median_X"], not_matches["median_Y"], s = 300, facecolors = 'none', edgecolors = 'r')
        ax[0,0].set_title("Green circles = matches and red circles = mismatches", fontsize = 10)

        # top right plot: wrong detected sources
        ax[0,1].scatter(events_noise["X"], events_noise["Y"], c = 'grey', s = 0.004)
        wrong_counter = 0
        for cluster in list(not_matches['cluster']):
            if cluster != -1:
                wrong_counter = wrong_counter + 1
                cluster_mask = (events_sources['new_cluster'] == cluster)
                events_cluster = events_sources[cluster_mask]
                ax[0,1].scatter(events_cluster["X"], events_cluster["Y"], c = 'blue', s = 0.04)  
        ax[0,1].scatter(not_matches["median_X"], not_matches["median_Y"], s = 300, facecolors = 'none', edgecolors = 'r')
        ax[0,1].set_title("Falsly detected sources 'False positive' (" + str(wrong_counter) + ")", fontsize = 10)

        # bottom left plot: not detected sources
        ax[1,0].scatter(events_noise["X"], events_noise["Y"], c = 'grey', s = 0.004)
        not_counter = 0
        for cluster in range(len(src_SOC_median)):
            if cluster not in list(pairs_table['SOC_cluster']):
                not_counter = not_counter + 1
                ax[1,0].scatter(src_SOC_median['SOC_X'][cluster - 1], src_SOC_median['SOC_Y'][cluster - 1], s = 300, facecolors = 'none', edgecolors = 'b')
        ax[1,0].set_title("Not detected Sources 'False negative' (" + str(not_counter) + ")", fontsize = 10)

        # bottom right plot: matches
        ax[1,1].scatter(events_noise["X"], events_noise["Y"], c = 'grey', s = 0.004)
        for cluster in list(matches['cluster']):
            cluster_mask = (events_sources['new_cluster'] == cluster)
            events_cluster = events_sources[cluster_mask]
            ax[1,1].scatter(events_cluster["X"], events_cluster["Y"], c = 'blue', s = 0.04)
        ax[1,1].scatter(matches["median_X"], matches["median_Y"], s = 300, facecolors = 'none', edgecolors = 'g')
        n_matches = len(np.unique(pairs_table['clus_cluster']))
        ax[1,1].set_title("Matches 'True positive' (" + str(n_matches) + ")", fontsize = 10)

        # labels
        ax[0,0].set_ylabel("Y")
        ax[1,0].set_ylabel("Y")
        ax[1,0].set_xlabel("X")
        ax[1,1].set_xlabel("X")

        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_' + str(step_counter) + "_crossMatch_seperate.jpg", dpi = 150)
        plt.close()