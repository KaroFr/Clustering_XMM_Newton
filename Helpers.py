# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:44:01 2022

@author: Karolin.Frohnapfel
"""
from astropy.io import fits
import os
import requests
import tarfile
import shutil
from astropy.table import Table
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import pandas as pd
import matplotlib.pyplot as plt

"""
Method to download the important files from the XMM Newton Archive
Input:  the observation ID, that should be loaded
"""
def download_data(observation_ID):
    if len(observation_ID) != 10 or type(observation_ID) != str:
        print("Invalid observation ID: It must be a 10 digits String.")
        return
    
    matches_FTZ = []
    matches_OBSMLI = []

    # Search for existing files in the observation 
    for file in os.listdir("evFiles"):
        if (file.find(observation_ID) != -1) and (file.find("PIEVLI") != -1):
            matches_FTZ.append(file)
        if (file.find(observation_ID) != -1) and (file.find("OBSMLI") != -1):
            matches_OBSMLI.append(file)

    if (len(matches_FTZ) == 0) and (len(matches_OBSMLI) == 0):
        
        ######################################################################
        ########## download EPIC PN event list FITS file
        ######################################################################
        print("  - Downloading EPIC-PN FITS file from XMM-Newton Science Archive.")
        
        # URL that gets the file from XMM archive:
        url_FTZ_1 = "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=" + observation_ID + "&name=PIEVLI&level=PPS&instname=PN&extension=FTZ"
        r1 = requests.get(url_FTZ_1, allow_redirects = True)
        
        # Store the content of this URL query in a new file inside "evFiles/" directory:
        with open("evFiles/P" + observation_ID + "PNPIEVLI0000.FTZ", "wb") as file:
            file.write(r1.content) 

        ######################################################################
        ########## download source list
        ######################################################################
        print("  - Downloading source list FITS file from XMM-Newton Science Archive.")
        
        # URL that gets the list of sources in FITS format:
        url_FTZ_2 = "http://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno=" + observation_ID + "&name=OBSMLI&level=PPS&extension=FTZ&entity=EPIC_PPS_SOURCES"
        r2 = requests.get(url_FTZ_2, allow_redirects = True)
        
        # Store the content of the URL query as if it was a FITS file:
        with open("evFiles/P" + observation_ID + "PNEPX000OBSMLI0000.FTZ", "wb") as file:
            file.write(r2.content)
            
        fits_path = "evFiles/P" + observation_ID + "PNEPX000OBSMLI0000.FTZ"
        
        ######################################################################
        ########## open the source list as a FITS file
        ######################################################################
        try:
            fits.open(fits_path)
            could_open = True
            # If it couldnt be opened, it means that the URL query returns more than one file in a TAR file
        except:
            # Create a TAR file and store there the content returned by the URL query:
            with open("evFiles/P" + observation_ID + "PNEPX000OBSMLI0000.TAR", "wb") as file:
                file.write(r2.content)
            could_open = False
        
        # If the file is a TAR:
        if could_open == False:
            # Remove the previously created FITS, since it is corrupted:
            os.remove("evFiles/P" + observation_ID + "PNEPX000OBSMLI0000.FTZ")
            
            # Open the TAR file:
            tar_path = "evFiles/P" + observation_ID + "PNEPX000OBSMLI0000.TAR"
            tar = tarfile.open(tar_path)
            
            # Create a directory to store the files inside the TAR:
            os.mkdir("evFiles/P" + observation_ID + "_TAR")
            tar.extractall("evFiles/P" + observation_ID + "_TAR")
            
            # Close the TAR file:
            tar.close()
            
            # From the new directory, select the EPIC source list FITS file and copy to "evFiles/" directory:
            files_in_tar = os.listdir("evFiles/P" + observation_ID + "_TAR/" + observation_ID + "/pps/")
            shutil.copyfile("evFiles/P" + observation_ID + "_TAR/" + observation_ID + "/pps/" + files_in_tar[0], 
                            "evFiles/P" + observation_ID + "PNEPX000OBSMLI0000.FTZ")
            
            # Delete the new directory, since it is no longer useful:
            shutil.rmtree("evFiles/P" + observation_ID + "_TAR/")
            os.remove(tar_path)
        
        # If the file could be opened as a FITS file, there no need to do more tasks:
        print("  - Downloaded the PN events list and the source list of observation " + str(observation_ID))

    else:
        print("  - The files of observation " + str(observation_ID) + " have already been downloaded")
        
"""
Method to open the PN event file from the local directory /evfiles
Input:  the observation ID, that should be opened
Output: header and data of the events file
"""
def load_events(observation_ID):
    if len(observation_ID) != 10 or type(observation_ID) != str:
        print("Invalid observation ID: It must be a 10 digits String.")
        return
    
    # look for files in local directory
    files = os.listdir("evFiles")
    selected_files = []
    for file in files:
        if file.find(observation_ID) != -1:
            selected_files.append(file)

    # if there are no fitting files in the local directory, need to download them first
    if len(selected_files) == 0:
        print("  (!) No local files matched with this observation ID.")
        download_data(observation_ID)
        # look for files in local directory
        files = os.listdir("evFiles")
        selected_files = []
        for file in files:
            if file.find(observation_ID) != -1:
                selected_files.append(file)
    
    for file in selected_files:
        # Find the FITS file corresponding to the uncleaned EPIC PN event list:
        if file.find("PIEVLI") != -1:
            selected_PN = file
            cleaned = False
            break

    fits_path_PN = "evFiles/" + selected_PN
    
    # Get header and data of he EPIC PN file
    with fits.open(fits_path_PN) as hdul:
        header = hdul["EVENTS"].header
        events = Table(hdul["EVENTS"].data)
        
        if cleaned == False:
            # Filter some events on the total EPIC PN event list:
            mask = (events['FLAG'] == 0) & (events['PI'] > 300) & (events['PI'] < 7000) & (events['PATTERN'] <= 4)
            events = events[mask]
            
            # Classify the ENERGY_BAND according to the Energy level (PI)
            events['ENERGY_BAND'] = 0
            events["ENERGY_BAND"][np.where(events["PI"] <= 700)] = 0 #RED
            events["ENERGY_BAND"][np.where((events["PI"] > 700) & (events["PI"] <= 1200))] = 1 #GREEN
            events["ENERGY_BAND"][np.where(events["PI"] > 1200)] = 2 #BLUE
    
    print("  - Loaded the EPIC-PN FITS file from local directory.")
    return events, header

"""
Load the sources file from the local directory.
It has been downloaded together with the events file.
"""
def load_sources(observation_ID):
    if len(observation_ID) != 10 or type(observation_ID) != str:
        print("Invalid observation ID: It must be a 10 digits String.")
        return
    
    srcFile = 'P' + str(observation_ID) + 'PNEPX000OBSMLI0000.FTZ'
    
    try:
        with fits.open("evFiles/" + srcFile) as sourceHDU:
            srclist = sourceHDU['SRCLIST'].data
        return srclist
    except:
        download_data(observation_ID)
        with fits.open("evFiles/" + srcFile) as sourceHDU:
            srclist = sourceHDU['SRCLIST'].data
        return srclist

"""
Save an astropy table to a csv file
Input: table, observation ID, ID and the title of the file
"""
def save_table_to_csv(table, observation_ID, ID, title = 'MyTable'):
    array_table = np.asarray(table)
    df_table = pd.DataFrame(array_table)
    df_table.to_csv(r'results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/' + title + '.csv', index = False)

"""
Save an events file to a fits file
Input:  the observation ID, events file, header and identifier for the file name
"""
def save_events_to_fits(events, header, ID, identifier = ''):
    observation_ID = header['OBS_ID']
    myHDU = fits.BinTableHDU(data = events, header = header)
    myHDU.writeto("evFiles/P" + observation_ID + "_ID_" + str(ID) + "_" + identifier + ".FTZ", overwrite = True)

"""
Method for the linear normalization
Input:  the events file, that should be normalized and the scale factor (deault = 3)
The higher the scale facter, the less importance gets the time compared to the position
"""
def linear_normalization(events, scale_factor = 3):
    # linear scaling
    data_TIME = events['TIME']
    data_DETX = events['DETX']
    data_DETY = events['DETY']
    data_PI = events['PI']

    # calculate the intervals for scaling
    min_DETX_scaled = min(data_DETX)/10000
    max_DETX_scaled = max(data_DETX)/10000
    min_DETY_scaled = min(data_DETY)/10000
    max_DETY_scaled = max(data_DETY)/10000
    min_TIME_scaled = 0
    max_TIME_scaled = np.ptp(data_TIME)/(10000 * scale_factor)
    min_PI_scaled = 0
    max_PI_scaled = 1

    # set the scalers
    scaler_TIME = MinMaxScaler(feature_range = (min_TIME_scaled, max_TIME_scaled))
    scaler_DETX = MinMaxScaler(feature_range = (min_DETX_scaled, max_DETX_scaled))
    scaler_DETY = MinMaxScaler(feature_range = (min_DETY_scaled, max_DETY_scaled))
    scaler_PI = MinMaxScaler(feature_range = (min_PI_scaled, max_PI_scaled))

    # transform data
    scaled_TIME = scaler_TIME.fit_transform(data_TIME.reshape(-1,1))
    scaled_DETX = scaler_DETX.fit_transform(data_DETX.reshape(-1,1))
    scaled_DETY = scaler_DETY.fit_transform(data_DETY.reshape(-1,1))
    scaled_PI = scaler_PI.fit_transform(data_PI.reshape(-1,1))

    events['TIME_norm'] = scaled_TIME.reshape(-1)
    events['DETX_norm'] = scaled_DETX.reshape(-1)
    events['DETY_norm'] = scaled_DETY.reshape(-1)
    events['PI_norm'] = scaled_PI.reshape(-1)
    
    print("  - Finished the linear scaling of the events file.")
    return events

"""
Method to pick a sample
Input:  the events file, from which to pick and the size of the sample
Output: the sample, the oob events and a boolean saying if a sample was picked
If the events file is smaller than the prefered sample size, then no sample will be picked
"""
def pick_sample(events_table, max_n_events):
    # number of events in the original data
    n_events = len(events_table)
    # amount of events to be kept
    n_keep = min(max_n_events, n_events)

    thin_out = n_keep < n_events
    if thin_out:
        
        # create random indices to pick the sample
        indices = np.arange(0,n_events)
        random.shuffle(indices)
        indices_thined = indices[:n_keep]
        indices_oob = indices[n_keep:]
        
        # save the samples for HDBSCAN and the oob samples
        events_thined = events_table[indices_thined]
        events_oob = events_table[indices_oob]
    
    else:
        events_thined = events_table.copy()
        events_oob = Table(names = events_table.colnames)
        
    return events_thined, events_oob

"""
Generates plots of each cluster individually.
Can be used after the consensus clustering is performed.
"""
def individual_cluster_plots(events_table, final_clusters_HDBSCAN, ID, observation_ID):
    print("  - Started plotting the individual clusters.")
    cluster_str = "new_cluster"

    n_clusters = max(events_table[cluster_str]) + 1

    for index, cluster in enumerate(range(n_clusters)):
        if cluster == -1:
            continue
        # extract the clustered events
        cluster_mask = (events_table[cluster_str] == cluster)
        events_cluster = events_table[cluster_mask]

        # extract the background events
        events_background = events_table[~cluster_mask]
        
        ######################## total plot
        fig = plt.figure(figsize=[12,16])
        
        # =============
        # First subplot
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        if final_clusters_HDBSCAN['match'][index] == 1:
            ax.set_title("Cluster " + str(int(cluster)) + " (Match; Source Probability = " + str(round(final_clusters_HDBSCAN['source_likelihood'][index],4)) + ")", fontsize = 18)
        else:
            ax.set_title("Cluster " + str(int(cluster)) + " (not a Match; Source Probability = " + str(round(final_clusters_HDBSCAN['source_likelihood'][index],4)) + ")", fontsize = 18)
        ax.scatter(xs = events_cluster["TIME"], 
                        ys = events_cluster["DETY"].astype(float), 
                        zs = events_cluster["DETX"].astype(float), 
                        c = events_cluster[cluster_str], 
                        s = 0.5,
                        cmap = 'prism')
        
        ax.set_xlim3d(min(events_table["TIME"]), max(events_table["TIME"]))
        ax.set_ylim3d(min(events_table["DETY"]), max(events_table["DETY"]))
        ax.set_zlim3d(min(events_table["DETX"]), max(events_table["DETX"]))
        
        ax.set_box_aspect(aspect = (4,1,1))
        
        ax.set_xlabel("TIME")
        ax.set_ylabel("DETY")
        ax.set_zlabel("DETX")
        
        # ==============
        # Second subplot
        ax = fig.add_subplot(2, 1, 2)
        
        # ax.text(x = -18000, y = 17000, s = "mean intra cluster distance = " + str(round(mean_dist, 5 )), c = 'red')
        
        ax.scatter(events_background["DETX"], events_background["DETY"], c = 'grey', s = 0.004)
        ax.scatter(events_cluster["DETX"], events_cluster["DETY"], c = events_cluster[cluster_str], cmap = 'prism', s = 0.04)
        ax.scatter(x = np.mean(events_cluster["DETX"]), y = np.mean(events_cluster["DETY"]), s = 300, facecolors = 'none', edgecolors = 'r')
        
        ax.set_box_aspect(aspect = (1))

        ax.set_xlabel("DETX")
        ax.set_ylabel("DETY")
        
        plt.savefig('results/results_ID_' + str(ID) + '_' + str(observation_ID) + '/step_8_3HDBSCAN++_cluster_' + str(int(cluster)) + ".jpg", dpi = 150)
        plt.close()
    
    print("  - Finished plotting the individual clusters.")
    