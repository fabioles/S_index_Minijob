#!/usr/bin/env python
# coding: utf-8

# ### Routines adapted from git project

# In[1]:


import pandas as pd
import os
import numpy as np
from glob import glob
import filepaths

def clean_objectlists(objectlist,path_to_outfile):
    '''

    :param objectlist: list with object names
    :param path_to_outfile: pathname where list should be stored
    :return: list with object names according to SIMBAD
    '''
    objectlist_clean = []
    for source in objectlist:
        if source[:2] == 'Cl' or source[:2] == 'EM':
            obj_class = source[:2]
            source_clean = obj_class+'* '+source[2:]
        elif source[:2] == 'EQ' :
            source_clean = ''
        elif source == 'FHya':
            source_clean = 'F Hya'
        elif source == 'f02cyg':
            source_clean = '*f02 Cyg'
        elif source == 'fPer':
            source_clean = 'f Per'
        elif source == 'GJ3404A':
            source_clean = 'GJ3404'
        elif source == 'GJ3193B':
            source_clean = 'GJ3193'        
        elif source[:2] == 'HD':
            if source[-1].isalpha():
                source_clean = source[:-1]
            else:
                source_clean = source
        elif source == 'KOCH':
            source_clean = ''
        elif source[:3]=='NGC':
            source_clean = source[:7]+' '+source[7:]
        elif source[:4] == 'NOVA':
            source_clean = source[:4] + ' ' + source[4:]
        elif source[0]=='V' and source != 'VB10' and source != 'VAnd' and source != 'VepsEri' and source[1] != '*':
            source_clean = 'V*'+source[1:]
        elif source =='VAnd':
            source_clean = 'V And'
        elif source == 'VepsEri':
            source_clean = 'V*eps Eri'
        elif source[0].islower() and source[:4] != 'kelt':
            source_clean= '*'+source
        elif source[:3]=='ADS':
            source_clean = ''
        elif source =='Sol':
            source_clean = ''
        elif source == 'CD_44836B':
            source_clean == ''
        elif source == 'CD_44836A':
            source_clean == ''
        # Noch nicht bekannt: Form "Cl* Melotte 25 VA AGe m"
        else:
            source_clean = source
        objectlist_clean.append(source_clean.strip())
    object_frame = pd.DataFrame({'object':objectlist,'object_simbad':objectlist_clean})
    object_frame.to_csv(path_to_outfile,index=False)
    return objectlist_clean

def generate_objectlists_for_gaia(objectlist,folder,extra_name=''):
    '''
    Generates objectlists as chunks of 200 objects in order to downlad gaia data from http://gaia.ari.uni-heidelberg.de/singlesource.html
    (since the maximum number of single sources is 200)
    :param objectlist: list with single sources
    :param folder: folder where the lists should be generated
    :return: Nothing, lists are directly generated as files in the folder
    '''
    num_of_lists = int(np.ceil(len(objectlist)/200))
    for i in range(num_of_lists-1):
        with open(folder+f'sources_{extra_name}_{i}.csv','w+') as f:
            f.writelines('\n'.join(objectlist[i*200:i*200+200]))
    with open(folder+f'sources_{extra_name}_{num_of_lists-1}.csv', 'w+') as f:
        f.writelines('\n'.join(objectlist[(num_of_lists-1) * 200:]))
    return

def generate_gaia_frame(folder,columns ='../column_names_subset.txt',sourcefile_pattern = None):
    '''
    Generates a frame with gaia data from all sources lists, sources lists should be in the the form 'SingleSource*'
    :param folder: folder where the single gaia tables are
    :param sourcefile_pattern: pattern which is looked for in the singlesource files. If None, every file with pattern 'SingleSource' is taken
    :param columns: either str or list, gives the columns subset. If str, the parameter is interpreted as filename,
    which is looked for in folder
    :return: pandas  DataFrame with the gaia data for all downloaded single sources
    '''
    origin = os.getcwd()
    os.chdir(folder)
    if sourcefile_pattern is not None:
        file_list= glob('SingleSource*'+sourcefile_pattern+'*')
    else:
        file_list = glob('SingleSource*')
    frames = []
    for file in file_list:
        frames.append(pd.read_csv(file))
    gaia_frame = pd.concat(frames)
    column_names = []
    if type(columns)==str:
        with open(columns) as f:
            for line in f:
                column_names.append(line.rstrip('\n'))
    else:
        column_names = columns
    os.chdir(origin)
    return gaia_frame[column_names]


def find_gaia_errors(gaia_frame):
    """
    calculates errorbars from gaia percentiles
    :param gaia_frame: data frame with gaia params
    :return: the dataframe, enriched by the errorbars
    """
    df = gaia_frame
    df['teff_err_lower'] = df['teff_val'] - df['teff_percentile_lower']
    df['teff_err_upper'] = df['teff_percentile_upper'] - df['teff_val']
    df['lum_err_lower'] = df['lum_val'] - df['lum_percentile_lower']
    df['lum_err_upper'] = df['lum_percentile_upper'] - df['lum_val']
    return df



def create_database(polar_frame, gaia_frame):
    '''
    joins the polarbase data and the gaia data
    :param polar_frame: the data from polarbase
    :param gaia_frame: the respective gaia data for single sources
    :return: pandas DataFrame with polarbase data where gaia data are available
    '''
    together_frame = pd.merge(polar_frame, gaia_frame, left_on='objet', right_on='input_position', how='outer',
                              suffixes=('_polarbase', '_gaia'))
    return together_frame


# ### Load and clean catalog file

# In[2]:


def load_catalog(path):
    '''
    loads the catalog from catalog.dat, which has space seperated table with some entries missing.
    '''
    catalog_list = []
    with open(path) as f:
        for line in f:
            Seq = int(line[:4])
            Name = line[4:15].strip()
            SpType = line[15:37].strip()
            BV = line[37:51]
            RAdeg = line[51:64]
            DEdeg = line[64:77]
            VMAG = line[77:93]
            Plx = line[93:100]
            Smean = line[100:118]
            Smed = line[118:136]
            logRpHK = line[136:-2]

            #convert these entries to floats
            line_list = [Seq, Name, SpType]
            for parameter in [BV, RAdeg, DEdeg, VMAG, Plx, Smean, Smed, logRpHK]:
                if parameter.strip() == '':
                    parameter = None
                else:
                    parameter = float(parameter)
                line_list.extend([parameter])
            catalog_list.append(line_list)

    catalog_dict = {'Seq': [entry[0] for entry in catalog_list],
                   'Name': [entry[1] for entry in catalog_list],
                   'SpType': [entry[2] for entry in catalog_list],
                   'BV': [entry[3] for entry in catalog_list],
                   'RAdeg': [entry[4] for entry in catalog_list],
                   'DEdeg': [entry[5] for entry in catalog_list],
                   'VMAG': [entry[6] for entry in catalog_list],
                   'Plx': [entry[7] for entry in catalog_list],
                   'Smean': [entry[8] for entry in catalog_list],
                   'Smed': [entry[9] for entry in catalog_list],
                   'logRpHK': [entry[10] for entry in catalog_list]}

    catalog = pd.DataFrame(catalog_dict)
    
    return catalog

def clean_catalog(catalog):
    '''
    Removes duplicate entries in catalog. The resulting catalog contains median values of parameters
    if there were multiple entries for a star.
    '''
    new_cataloglist = []
    parameter_list = ['Seq', 'Name', 'SpType', 'BV', 'RAdeg', 'DEdeg',
                      'VMAG', 'Plx', 'Smean', 'Smed', 'logRpHK']
    for name in set(catalog['Name']):
        one_entry = []
        ind = np.where(catalog['Name'] == name)[0]
        if ind.size == 1:
            #if there is only one entry, take all of these values
            for parameter in parameter_list:
                one_entry.extend([catalog[parameter][ind[0]]])
        else:
            #if there are multiple entries, take median of float parameters
            for parameter in parameter_list[:3]:
                one_entry.extend([catalog[parameter][ind[0]]])
            for parameter in parameter_list[3:]:
                ind_notnan = np.where(pd.notnull(catalog[parameter][ind]))[0]
                if ind_notnan.size == 0:
                     one_entry.extend([np.nan])
                else:
                    median_value = np.median(catalog[parameter][ind[ind_notnan]])
                    one_entry.extend([median_value])
        new_cataloglist.append(one_entry)

    catalog_clean = pd.DataFrame(new_cataloglist, columns = ['Seq', 'Name', 'SpType', 'BV', 'RAdeg', 'DEdeg', 'VMAG', 'Plx', 'Smean', 'Smed', 'logRpHK'])
    
    return catalog_clean


# In[3]:


#load and clean S-index catalog
catalog_path = filepaths.base_path+'catalog_original.dat'
catalog = load_catalog(catalog_path)
catalog_clean = clean_catalog(catalog)


# In[53]:


#automatically request gaia data, excluding all stars not found in simbad
notinSimbad = []
for name in catalog_clean['Name']:
    filename = filepaths.base_path+'GaiaDownload/'+name+'.csv'
    wget.download('http://gaia.ari.uni-heidelberg.de/singlesource/search?obj='+name, out=filename)
    dataframe = pd.read_csv(filename)
    if dataframe.columns[0][:5] == 'ERROR':
        notinSimbad.append(name)


# In[65]:


#Save list of stars not in the simbad catalog
pd.DataFrame(notinSimbad, columns = ['Name']).to_csv(filepaths.base_path+'NotInSimbad.csv')


# In[80]:


#Combine all single files into one large dataframe
df_list = []
for name in catalog_clean['Name']:
    if not name in notinSimbad:
        filename = filepaths.base_path+'CheckSource/'+name+'.csv'
#         with open(filename) as file:
#             print(file.read().split(','))
        df = pd.read_csv(filename)
        df.insert(0, 'Name', name)
        df_list.append(df)
df_combined = pd.concat(df_list, ignore_index=True)


# In[91]:


#Find errors of gaia parameters
gf = find_gaia_errors(df_combined)


# In[100]:


#merge with s-index catalog and save
data = pd.merge(gf, catalog_clean,on='Name')
data.to_csv(filepaths.base_path+'CombinedCatalog.csv',index=False)


# In[348]:





# In[ ]:




