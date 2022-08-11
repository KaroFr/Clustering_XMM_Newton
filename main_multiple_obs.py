# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:47:49 2022

@author: Karolin.Frohnapfel
"""

from Helpers import load_events
from Helpers import linear_normalization
from Helpers import save_table_to_csv
from Helpers import save_events_to_fits
from BTICleaner import BTICleaner
from SourceDetector import SourceDetector
from Validator import Validator

from astropy.table import vstack
import os
import time
from astropy.table import Table

# make a directory for the upcoming results, if not already existend
if not os.path.isdir('evFiles'):
    os.mkdir('evFiles')
if not os.path.isdir('results'):
    os.mkdir('results')

try:
    results_table = Table.read('results/results_csv.csv')
    ID = max(results_table['ID'])
except:
    ID = 0

##########################################################################
############# choose the observation
##########################################################################
observations = ['0827350201','0860260601','0860302501','0861610201','0862730101',
                '0862920201','0863560201','0864050401','0864052301','0864110101',
                '0864110201','0864950201','0865350101','0865350201','0865350301',
                '0870830201','0870920101','0164560701','0208000101','0405320801',
                '0651870401','0673851601','0694170101','0744414101']

##########################################################################
############# set the parameters
##########################################################################
# scaling factor for the linear normalization
# scale_factor = 3                                   # default = 3

# min_samples for the DBSCAN for the bti detection
# ms_bti_det = 100                                   # default = 100

# intervals for the bti cleaning
# n_bins_PI = 10                                     # default = 10
# n_bins_DETX = 5                                    # default = 5
# n_bins_DETY = 5                                    # default = 5

# min_samples for the DBSCAN for the noise detection
# ms_noise_det = 20                                  # default = 20

# hyper parameters for the HDBSCAN for the ensemble
# mcs_HDBSCAN = [10,10,15,20]                        # default = [10,10,15,20]
# ms_HDBSCAN = [20,40,30,60]                         # default = [20,40,30,60]
# cse_HDBSCAN = [0.005,0.005,0.005,0.005]            # default = [0.005,0.005,0.005,0.005]

# threshold to disregard noise before the consensus clustering
# not_noise_threshold = 2                            # default = 2

# threshold for two sources to be a match in the validation
# distance_threshold = 300                           # default = 300

for observation_ID in observations:
    ##########################################################################
    ############# loading the data
    ##########################################################################
    
    time_start = time.time()
    ID = ID + 1
    print("Observation ID: " + observation_ID + ", ID: " + str(ID))
    events, header = load_events(observation_ID)
    # events = linear_normalization(events)
    events = linear_normalization(events, scale_factor)
    
    ##########################################################################
    ############# ceaning the bad time intervals
    ##########################################################################
    
    # initalize a cleaner
    bti_cleaner = BTICleaner(events, header, ID)
    # bti_cleaner = BTICleaner(events, header, ID, n_bins_PI, n_bins_DETX, n_bins_DETY)
    
    # plots before cleaning
    bti_cleaner.plot_distribution()
    bti_cleaner.plot_DETX_DETY_TIME()
    
    # bti detection
    time_bbd = time.time()
    bti_cleaner.detect_btis()
    time_abd = time.time()
    bti_cleaner.plot_bti_comparison()
    
    # cleaning
    time_bc = time.time()
    bti_cleaner.clean_btis()
    time_ac = time.time()
    
    # plots after cleaning
    bti_cleaner.plot_distribution()
    bti_cleaner.plot_bti_comparison()
    bti_cleaner.plot_DETX_DETY_TIME()
    bti_cleaner.plot_DETX_DETY()
    bti_cleaner.plot_DETY_TIME()
    
    # get values and cleaned events
    results_dict = bti_cleaner.get_values()
    events_cleaned = bti_cleaner.events_after_cleaning        
    
    ##########################################################################
    ############# source detection
    ##########################################################################
    time_ac = time.time()
    # initialize a source detector
    source_detector = SourceDetector(events_cleaned, header, ID)
    # source_detector = SourceDetector(events_cleaned, header, ID, ms_noise_det, mcs_HDBSCAN, ms_HDBSCAN, cse_HDBSCAN, not_noise_threshold)
    
    # noise detection + plot
    time_bnd = time.time()
    source_detector.noise_detection()
    time_and = time.time()
    source_detector.plot_noise_detection()
    
    # ensemble
    time_be = time.time()
    source_detector.ensemble()
    time_ae = time.time()
    
    # consensus_clustering
    time_bcc = time.time()
    source_detector.consensus_clustering()
    time_acc = time.time()
    
    # pairing + plot
    source_detector.pairing()
    source_detector.plot_pairing()
    
    # get the values of the sources detection
    results_dict_source_det = source_detector.get_values()
    results_dict.update(results_dict_source_det)
    
    ##########################################################################
    ############# save the clusters tables
    ##########################################################################
    final_clusters_HDBSCAN = source_detector.final_clusters_HDBSCAN
    save_table_to_csv(final_clusters_HDBSCAN, observation_ID, ID, 'final_clusters')
    
    clusters_HDBSCAN = source_detector.clusters_HDBSCAN
    save_table_to_csv(clusters_HDBSCAN, observation_ID, ID, 'cluster_pairing')
    
    ##########################################################################
    ############# cross match validation
    ##########################################################################
    
    # get events and co for validation
    events_sources = source_detector.events_sources
    events_noise = source_detector.events_noise
    events_final = vstack([events_sources, events_noise])
    
    # initialize a validator
    validator = Validator(events_final, header, final_clusters_HDBSCAN, ID)
    # validator = Validator(events_final, header, final_clusters_HDBSCAN, ID, ditance_threshold)
    
    # cross match validation + plot
    validator.cross_match()
    validator.plot_cross_match()
    
    # get the values of the sources detection
    results_dict_validation = validator.get_values()
    results_dict.update(results_dict_validation)
    
    ##########################################################################
    ############# save the final event file and the results
    ##########################################################################
    save_events_to_fits(events_final, header, ID, 'clustered')
    
    time_end = time.time()
    
    time_dict = {'time_bti_detection': [time_abd - time_bbd],
                 'time_bti_cleaning': [time_ac - time_bc],
                 'time_denoising': [time_and - time_bnd],
                 'time_ensemble': [time_ae - time_be],
                 'time_consensus_clus': [time_acc - time_bcc],
                 'time_total_without_plots': [time_abd - time_bbd + 
                                             time_ac - time_bc + 
                                             time_and - time_bnd + 
                                             time_ae - time_be + 
                                             time_acc - time_bcc],
                 'time_total': [time_end - time_start]}
    results_dict.update(time_dict)
    
    try:
        results_table.add_row(results_dict)
    except:
        results_table = Table(results_dict)
    
results_table.write('results/results_csv.csv', overwrite = True)