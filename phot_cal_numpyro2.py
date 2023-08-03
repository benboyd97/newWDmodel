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
import pandas as pd
from collections import OrderedDict

import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from jax import random
import jax
import jax.numpy as jnp
from numpyro.infer import init_to_median, init_to_sample

numpyro.set_host_device_count(4)
jax.config.update('jax_enable_x64',True)


def get_data():
    suffix = '_abmag'
    ilaph_version = '5'
    magsys = 'ABmag'
    ref = 'FMAG'  # which photometry package should be used to compute zeropoints
    mintime = 0.7 # mininum exposure length to consider for computing zeropoints

    stars     = ['GD-153', 'GD-71', 'G191B2B'] # what stars are standards
    marker    = ['o',      'd',     '*']       # markers to use for each standard in plots
    use_stars = ['GD-153', 'G191B2B', 'GD-71'] # what stars are to be used to get zeropoints

    standard_mags_file = f'/Users/bboyd/Documents/work/wd/WD_data/photometry/20190215/calspec_standards_WFC3_UVIS2_IR_{magsys}.txt' # standard's apparent magnitudes in each band
    smags = at.Table.read(standard_mags_file, format='ascii')
    smags = smags.to_pandas()
    smags.set_index('objID', inplace=True)


    # cut out some fields we do not need to make indexing the data frame a little easier
    dref = 'ERRMAG'
    drop_fields = ['X', 'Y', 'BCKGRMS', 'SKY', 'FITS-FILE']

    mag_table = OrderedDict() # stores combined magnitudes and zeropoints in each passband
    all_mags   = at.Table.read('/Users/bboyd/Documents/work/wd/WD_data/photometry/20190215/src/AS/all+standardmeasures_C25_ILAPHv{}_AS.txt'.format(ilaph_version), format='ascii')
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



def phot_cal_model(sample_idx,standard_idx,mag_app_i,sig_i,sig_j,sample_mags=None,standard_mags=None,zpt_est=24,f160w=False):
    sig_int =  numpyro.sample("sig_intrinsic", dist.HalfCauchy(1))
    zpt = numpyro.sample("zeropoint",dist.Normal(loc=zpt_est,scale=1))


    if f160w != True:
        nu = numpyro.sample("nu", dist.HalfCauchy(3))

    

    n_sample_obj= len(np.unique(sample_idx))

    with numpyro.plate("plate_j", n_sample_obj):
        mag_app_j = numpyro.sample("mag_app_j", dist.Uniform(8,25))
        mag_inst_j = mag_app_j - zpt

    
    n_standard_obj= len(np.unique(standard_idx))

    with numpyro.plate("plate_i", n_standard_obj):

        mag_inst_i = mag_app_i - zpt



    n_sample_obs= len(sample_idx)

    m_j = mag_inst_j[sample_idx]

    
    with numpyro.plate("data_j", n_sample_obs):
        full_var_j = (sig_int**2. + sig_j**2.)**0.5
        if f160w:
            numpyro.sample("m_j", dist.Normal(loc=m_j,scale=full_var_j), obs=sample_mags)

        else:
            numpyro.sample("m_j", dist.StudentT(loc=m_j,df=nu,scale=full_var_j), obs=sample_mags)

    n_standard_obs= len(standard_idx)

    m_i = mag_inst_i[standard_idx]
    with numpyro.plate("data_i", n_standard_obs):
        full_var_i = (sig_int**2. + sig_i**2.)**0.5
        if f160w:
            numpyro.sample("m_i", dist.Normal(loc=m_i,scale=full_var_i), obs=standard_mags)

        else:
            numpyro.sample("m_i", dist.StudentT(loc=m_i,df=nu,scale=full_var_i), obs=standard_mags)







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
dref = 'ERRMAG'




blank_table = at.Table(names=( 'obj_ID','F275W','dF275W','F336W','dF336W','F475W','dF475W','F625W','dF625W','F775W','dF775W','F160W','dF160W'),dtype=('S20','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8','f8'))

blank_table.add_row({'obj_ID': 'zeropoint'})
blank_table.add_row({'obj_ID': 'sig_intrinsic'})
blank_table.add_row({'obj_ID': 'nu'})

row_index =blank_table['obj_ID'] == 'nu'
blank_table['F160W'][row_index]=np.nan
blank_table['dF160W'][row_index]=np.nan


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
        print(standard_mags[ref])
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


        init_strategy=init_to_sample()
        nuts_kernel = NUTS(phot_cal_model,adapt_step_size=True,init_strategy=init_strategy)
        mcmc = MCMC(nuts_kernel, num_samples=5000, num_warmup=5000,num_chains=4)
        rng_key = random.PRNGKey(0)

        


        if pb == 'F160W':

            mcmc.run(rng_key,jnp.asarray(sample_idx),jnp.asarray(standard_idx),mag_app_i=jnp.asarray(mag_app_i),sig_i=jnp.asarray(standard_mags[dref]),sig_j=jnp.asarray(sample_mags[dref]),sample_mags=jnp.asarray(sample_mags[ref]),standard_mags=jnp.asarray(standard_mags[ref]),zpt_est=zpt_est,f160w=True)

        else:
            mcmc.run(rng_key,jnp.asarray(sample_idx),jnp.asarray(standard_idx),mag_app_i=jnp.asarray(mag_app_i),sig_i=jnp.asarray(standard_mags[dref]),sig_j=jnp.asarray(sample_mags[dref]),sample_mags=jnp.asarray(sample_mags[ref]),standard_mags=jnp.asarray(standard_mags[ref]),zpt_est=zpt_est,f160w=False)

        mcmc.print_summary()
        samps=mcmc.get_samples()


        keys = samps.keys()

        print(keys)

        for key in keys:

            shape = samps[key].shape
            if len(shape)>1:
                for i in range(shape[1]):

                    star_name=index_sample[i]
                    if star_name[0]=='W':
                        star_name=star_name[:star_name.rfind('-')]

                    star_name = star_name.replace('-', '').lower()
                    if star_name.rfind('.') !=-1:
                        star_name=star_name[:star_name.rfind('.')]
                    

                    if star_name not in list(blank_table['obj_ID']):
                        blank_table.add_row({'obj_ID': star_name})

                    row_index =blank_table['obj_ID'] == star_name
                    blank_table[pb][row_index] = np.mean(samps[key][:,i])
                    blank_table['d'+pb][row_index] = np.std(samps[key][:,i])

            else:
                row_index =blank_table['obj_ID'] == key
                blank_table[pb][row_index] = np.mean(samps[key])
                blank_table['d'+pb][row_index] = np.std(samps[key])
            

cols = blank_table.colnames
for c in cols:
    if blank_table[c].dtype == np.float64:
        blank_table[c].format = '%.6f'




blank_table.write('new_cal.dat',format='ascii.fixed_width',delimiter='  ',overwrite=True)

