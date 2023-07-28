#!/usr/bin/env python
import sys
import os
import glob
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import astropy.table as at
from numpy.lib.recfunctions import append_fields
from scipy.stats import linregress
import pymc as pm
import pandas as pd
from collections import OrderedDict

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from jax import random
import jax

    


def get_data():
    suffix = '_abmag'
    ilaph_version = '5'
    magsys = 'ABmag'
    ref = 'FMAG'  # which photometry package should be used to compute zeropoints
    mintime = 0.7 # mininum exposure length to consider for computing zeropoints

    stars     = ['GD-153', 'GD-71', 'G191B2B'] # what stars are standards
    marker    = ['o',      'd',     '*']       # markers to use for each standard in plots
    use_stars = ['GD-153', 'G191B2B', 'GD-71'] # what stars are to be used to get zeropoints

    standard_mags_file = f'/data/bmb41/WD_data/photometry/20190215/calspec_standards_WFC3_UVIS2_IR_{magsys}.txt' # standard's apparent magnitudes in each band
    smags = at.Table.read(standard_mags_file, format='ascii')
    smags = smags.to_pandas()
    smags.set_index('objID', inplace=True)


    # cut out some fields we do not need to make indexing the data frame a little easier
    dref = 'ERRMAG'
    drop_fields = ['X', 'Y', 'BCKGRMS', 'SKY', 'FITS-FILE']

    mag_table = OrderedDict() # stores combined magnitudes and zeropoints in each passband
    all_mags   = at.Table.read('/data/bmb41/WD_data/photometry/20190215/src/AS/all+standardmeasures_C25_ILAPHv{}_AS.txt'.format(ilaph_version), format='ascii')
    all_mags[ref] -= 30.
    mask = (all_mags[dref] < 0.5) & (np.abs(all_mags[ref]) < 50) & (all_mags['EXPTIME'] >= mintime)
    nbad = len(all_mags[~mask])
    print(all_mags[~mask])
    print("Trimming {:n} bad observations".format(nbad))
    all_mags = all_mags[mask]
    all_mags.rename_column('OBJECT-NAME','objID')
    all_mags.rename_column('FILTER','pb')

    #cycle_flag = [ 1 if x <= 56700 else 0 for x in all_mags['MJD'] ]
    #cycle_flag = np.array(cycle_flag)
    #all_mags['cycle'] = cycle_flag

    for pb in np.unique(all_mags['pb']):
        mask = (all_mags['pb'] == pb)
        mag_table[pb] = all_mags[mask].to_pandas()

   
    # init some structure to store the results for each passband
    return mag_table,smags



def phot_cal_model(standard_mask,zpt_est):
     
     sig_int =  numpyro.sample("sig_int", dist.HalfCauchy(1))
     zpt = numpyro.sample("zpt",dist.Normal(mu=zpt_est,sgima=1))


     

     mag_inst_i = mag_app_i[standard_idx] - zpt


     m_




mag_table,smags=get_data()

result_table = OrderedDict()
    # drop some fields we do not need from the results
derop_fields = ['hdi_3%','hdi_97%','mcse_mean','mcse_sd','ess_bulk','ess_tail','r_hat']
    # keep a track of all_objects
all_objects = set()
    # and variable names
    #var_names = ['zeropoint', 'c20_offset', 'sig_intrinsic', 'nu']
var_names = ['zeropoint', 'sig_intrinsic', 'nu']
nvar = len(var_names)
use_stars = ['GD-153', 'G191B2B', 'GD-71']
ref = 'FMAG'  # which photometry package should be used to compute zeropoints


print(mag_table)


for i, pb in enumerate(mag_table):

        all_mags = mag_table[pb]
        mask = all_mags['objID'].isin(use_stars)

        sample_mags   = all_mags.copy()

        # the standards are "special" - they have apparent magnitudes from a
        # model and are used to set the zeropoint for everything else
        standard_mags = all_mags[mask].copy()

        # what are the unique stars
        standards  = standard_mags['objID'].unique()

        nstandards = len(standards)
        # map each standard star to an integer
        standard_ind = list(range(nstandards))
        standard_map = dict(zip(standards, standard_ind))
        index_standards = {value:key for key, value in standard_map.items()}
        print(index_standards, "Standards")

        # construct an index with the integer mapping for each star in the table
        standard_idx = [standard_map[x] for x in standard_mags['objID']]
        standard_mags['idx'] = standard_idx
        standard_idx = standard_mags['idx'].values
        #standard_cycle_idx = standard_mags['cycle'].values

        # get the apparent magnitude corresponding to each standard measurement
        mag_app_i  = np.array([smags.loc[x.replace('-','').lower(), pb] for x in standards])
        # the zeropoint guess is just the average difference of the apparent mags and the instrumental mags
        zpt_est = np.average(mag_app_i[standard_idx] - standard_mags[ref])
        print('Initial Guess Zeropoint {} : {:.4f}'.format(pb, zpt_est))

        samples  = sample_mags['objID'].unique()
        all_objects.update(samples)
        nsamples = len(samples)
        sample_ind = range(nsamples)
        sample_map = dict(zip(samples, sample_ind))
        print(sample_map)
        index_sample = {value:key for key, value in sample_map.items()}
        print(index_sample, "sample")
        sample_idx = [sample_map[x] for x in sample_mags['objID']]
        sample_mags['idx'] = sample_idx
        sample_idx = sample_mags['idx'].values






