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

numpyro.set_host_device_count(4)
jax.config.update('jax_enable_x64',True)


def get_data(cycles=False):
    suffix = '_abmag'
    ilaph_version = '5'
    magsys = 'ABmag'
    ref = 'FMAG'  # which photometry package should be used to compute zeropoints
    mintime = 0.7 # mininum exposure length to consider for computing zeropoints

    stars     = ['GD-153', 'GD-71', 'G191B2B'] # what stars are standards
    marker    = ['o',      'd',     '*']       # markers to use for each standard in plots
    use_stars = ['GD-153', 'G191B2B', 'GD-71'] # what stars are to be used to get zeropoints
    if cycles:
        standard_mags_file = f'/Users/bboyd/Documents/work/wd/WD_data/photometry/20181102/calspec_standards_WFC3_UVIS2_IR_{magsys}.txt' # standard's apparent magnitudes in each band
    else:
        standard_mags_file = f'/Users/bboyd/Documents/work/wd/WD_data/photometry/20190215/calspec_standards_WFC3_UVIS2_IR_{magsys}.txt' # standard's apparent magnitudes in each band
    smags = at.Table.read(standard_mags_file, format='ascii')
    smags = smags.to_pandas()
    smags.set_index('objID', inplace=True)


    # cut out some fields we do not need to make indexing the data frame a little easier
    dref = 'ERRMAG'
    drop_fields = ['X', 'Y', 'BCKGRMS', 'SKY', 'FITS-FILE']

    mag_table = OrderedDict() # stores combined magnitudes and zeropoints in each passband
    if cycles:
        all_mags   = at.Table.read('/Users/bboyd/Documents/work/wd/WD_data/photometry/20181102/src/AS/all+standardmeasures_C20_C22_ILAPHv{}_AS.txt'.format(ilaph_version), format='ascii')
    else:
        all_mags   = at.Table.read('/Users/bboyd/Documents/work/wd/WD_data/photometry/20190215/src/AS/all+standardmeasures_C25_ILAPHv{}_AS.txt'.format(ilaph_version), format='ascii')
    all_mags[ref] -= 30.
    mask = (all_mags[dref] < 0.5) & (np.abs(all_mags[ref]) < 50) & (all_mags['EXPTIME'] >= mintime)
    nbad = len(all_mags[~mask])

    all_mags = all_mags[mask]
    all_mags.rename_column('OBJECT-NAME','objID')
    all_mags.rename_column('FILTER','pb')

    if cycles:
        cycle_flag = [ 1 if x <= 56700 else 0 for x in all_mags['MJD'] ]
        cycle_flag = np.array(cycle_flag)
        all_mags['cycle'] = cycle_flag
    
    for pb in np.unique(all_mags['pb']):
        mask = (all_mags['pb'] == pb)
        mag_table[pb] = all_mags[mask].to_pandas()


   
    # init some structure to store the results for each passband
    return mag_table,smags



def phot_cal_model_all_student(sample_idx,standard_idx,mag_app_i,sig_i,sig_j,sample_mags=None,standard_mags=None,zpt_est=24):
    
    
    n_bands= mag_app_i.shape[0]


    with numpyro.plate("plate_b", n_bands):

        sig_int_b =  numpyro.sample("sig_intrinsic", dist.HalfCauchy(1))
        zpt_b = numpyro.sample("zeropoint",dist.Normal(loc=zpt_est,scale=1))
        nu_b = numpyro.sample("nu", dist.HalfCauchy(3))

    
    n_sample_obj= len(np.unique(sample_idx))


    with numpyro.plate("plate_j", n_sample_obj*n_bands):
        mag_app_j = numpyro.sample("mag_app_j", dist.Uniform(8,25)).reshape(n_bands,n_sample_obj)
        mag_inst_j = mag_app_j - zpt_b.reshape(n_bands,1)


    n_standard_obj= len(np.unique(standard_idx))

    with numpyro.plate("plate_i", n_standard_obj):

        mag_inst_i = mag_app_i - zpt_b.reshape(n_bands,1)


    n_sample_obs= sample_mags.shape[1]



    m_j = mag_inst_j[jnp.arange(n_bands).astype(int)[:,None], sample_idx]
    mask = sample_idx!=1000
    with numpyro.plate("data_k",n_sample_obs):
        with numpyro.plate("data_j", n_bands):
       
            full_var_j = (sig_int_b.reshape(n_bands,1)**2.+sig_j**2.)**0.5

            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("m_j", dist.StudentT(loc=m_j,df=nu_b.reshape(n_bands,1),scale=full_var_j), obs=sample_mags)

    n_standard_obs= standard_mags.shape[1]

    m_i = mag_inst_i[jnp.arange(n_bands).astype(int)[:,None], standard_idx]


    mask = standard_idx!=1000
    with numpyro.plate("data_l",n_standard_obs):
        with numpyro.plate("data_i", n_bands):
            full_var_i = (sig_int_b.reshape(n_bands,1)**2. + sig_i**2.)**0.5
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("m_i", dist.StudentT(loc=m_i,df=nu_b.reshape(n_bands,1),scale=full_var_i), obs=standard_mags)

def phot_cal_model_all_normal(sample_idx,standard_idx,mag_app_i,sig_i,sig_j,sample_mags=None,standard_mags=None,zpt_est=24):
    
    
    n_bands= mag_app_i.shape[0]

    alpha = numpyro.sample("alpha",dist.Normal(loc=0,scale=0.01))
    beta = 15

    alpha_arr = jnp.append(jnp.array([alpha]),jnp.zeros(n_bands-1))


    with numpyro.plate("plate_b", n_bands):

        sig_int_b =  numpyro.sample("sig_intrinsic", dist.HalfCauchy(1))
        zpt_b = numpyro.sample("zeropoint",dist.Normal(loc=zpt_est,scale=1))

    
    n_sample_obj= len(np.unique(sample_idx))

    

    with numpyro.plate("plate_j", n_sample_obj*n_bands):
        mag_app_j = numpyro.sample("mag_app_j", dist.Uniform(8,25)).reshape(n_bands,n_sample_obj)
        mag_inst_j = mag_app_j - zpt_b.reshape(n_bands,1) + alpha_arr.reshape(n_bands,1)*(mag_app_j-beta)


    n_standard_obj= len(np.unique(standard_idx))

    with numpyro.plate("plate_i", n_standard_obj):

        mag_inst_i = mag_app_i - zpt_b.reshape(n_bands,1)


    n_sample_obs= sample_mags.shape[1]



    m_j = mag_inst_j[jnp.arange(n_bands).astype(int)[:,None], sample_idx]
    mask = sample_idx!=1000
    with numpyro.plate("data_k",n_sample_obs):
        with numpyro.plate("data_j", n_bands):
       
            full_var_j = (sig_int_b.reshape(n_bands,1)**2.+sig_j**2.)**0.5

            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("m_j", dist.Normal(loc=m_j,scale=full_var_j), obs=sample_mags)

    n_standard_obs= standard_mags.shape[1]

    m_i = mag_inst_i[jnp.arange(n_bands).astype(int)[:,None], standard_idx]


    mask = standard_idx!=1000
    with numpyro.plate("data_l",n_standard_obs):
        with numpyro.plate("data_i", n_bands):
            full_var_i = (sig_int_b.reshape(n_bands,1)**2. + sig_i**2.)**0.5
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("m_i", dist.Normal(loc=m_i,scale=full_var_i), obs=standard_mags)


def phot_cal_model(sample_idx,standard_idx,mag_app_i,sig_i,sig_j,sample_mags=None,standard_mags=None,zpt_est=24):
    
    

    n_bands= mag_app_i.shape[0]


    with numpyro.plate("plate_b", n_bands):

        sig_int_b =  numpyro.sample("sig_intrinsic", dist.HalfCauchy(1))
        zpt_b = numpyro.sample("zeropoint",dist.Normal(loc=zpt_est,scale=1))
        
        
        nu_b = numpyro.sample("nu", dist.HalfCauchy(3))

    
    n_sample_obj= len(np.unique(sample_idx))


    with numpyro.plate("plate_j", n_sample_obj*n_bands):
        mag_app_j = numpyro.sample("mag_app_j", dist.Uniform(8,25)).reshape(n_bands,n_sample_obj)
        mag_inst_j = mag_app_j - zpt_b.reshape(n_bands,1)


    n_standard_obj= len(np.unique(standard_idx))

    with numpyro.plate("plate_i", n_standard_obj):

        mag_inst_i = mag_app_i - zpt_b.reshape(n_bands,1)


    n_sample_obs= sample_mags.shape[1]

    m_j = mag_inst_j[jnp.arange(n_bands).astype(int)[:,None], sample_idx]
    mask = sample_idx!=1000



    with numpyro.plate("data_j1",n_sample_obs):
        with numpyro.plate("data_j2", n_bands-1):
            full_var_j = (sig_int_b.reshape(n_bands,1)**2.+sig_j**2.)**0.5

            with numpyro.handlers.mask(mask=mask[1:,:]):
                numpyro.sample("m_j", dist.StudentT(loc=m_j[1:,:],df=nu_b[1:].reshape(n_bands-1,1),scale=full_var_j[1:,:]), obs=sample_mags[1:,:])

    n_standard_obs= standard_mags.shape[1]

    m_i = mag_inst_i[jnp.arange(n_bands).astype(int)[:,None], standard_idx]

    mask = standard_idx!=1000
    with numpyro.plate("data_i1", n_standard_obs):
        with numpyro.plate("data_i2", n_bands-1):
            full_var_i = (sig_int_b.reshape(n_bands,1)**2. + sig_i**2.)**0.5
            with numpyro.handlers.mask(mask=mask[1:,:]):
                numpyro.sample("m_i", dist.StudentT(loc=m_i[1:,:],df=nu_b[1:].reshape(n_bands-1,1),scale=full_var_i[1:,:]), obs=standard_mags[1:,:])

    mask = sample_idx!=1000
    with numpyro.plate("data_k1", n_sample_obs):
        with numpyro.plate("data_k1", 1):
            full_var_j = (sig_int_b.reshape(n_bands,1)**2.+sig_j**2.)**0.5

            with numpyro.handlers.mask(mask=mask[:1,:]):
                numpyro.sample("m_k", dist.Normal(loc=m_j[:1,:],scale=full_var_j[:1,:]), obs=sample_mags[:1,:])


    mask = standard_idx!=1000
    with numpyro.plate("data_l1", n_standard_obs):
        with numpyro.plate("data_l2", 1):
            full_var_i = (sig_int_b.reshape(n_bands,1)**2. + sig_i**2.)**0.5
            with numpyro.handlers.mask(mask=mask[:1,:]):
                numpyro.sample("m_l", dist.Normal(loc=m_i[:1,:],scale=full_var_i[:1,:]), obs=standard_mags[:1,:])

def phot_cal_model_cycles(sample_idx,standard_idx,mag_app_i,sig_i,sig_j,sample_mags=None,standard_mags=None,zpt_est=24,samp_cycles=0,stand_cycles=0):
    
    

    n_bands= mag_app_i.shape[0]


    with numpyro.plate("plate_b", n_bands):

        sig_int_b =  numpyro.sample("sig_intrinsic", dist.HalfCauchy(1))
        zpt_b = numpyro.sample("zeropoint",dist.Normal(loc=zpt_est,scale=1))
        c20_offset = numpyro.sample("c20_offset",dist.Normal(loc=0,scale=1))
        
        nu_b = numpyro.sample("nu", dist.HalfCauchy(5))


    
    n_sample_obj= len(np.unique(sample_idx))


    with numpyro.plate("plate_j", n_sample_obj*n_bands):
        mag_app_j = numpyro.sample("mag_app_j", dist.Uniform(8,25)).reshape(n_bands,n_sample_obj)
        mag_inst_j = mag_app_j - zpt_b.reshape(n_bands,1)


    n_standard_obj= len(np.unique(standard_idx))

    with numpyro.plate("plate_i", n_standard_obj):

        mag_inst_i = mag_app_i -zpt_b.reshape(n_bands,1)


    n_sample_obs= sample_mags.shape[1]

    m_j = mag_inst_j[jnp.arange(n_bands).astype(int)[:,None], sample_idx] -samp_cycles*c20_offset.reshape(n_bands,1)
    mask = sample_idx!=1000
    with numpyro.plate("data_k",n_sample_obs):
        with numpyro.plate("data_j", n_bands):
       
            full_var_j = (sig_int_b.reshape(n_bands,1)**2.+sig_j**2.)**0.5

            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("m_j", dist.StudentT(loc=m_j,df=nu_b.reshape(n_bands,1),scale=full_var_j), obs=sample_mags)

    n_standard_obs= standard_mags.shape[1]

    m_i = mag_inst_i[jnp.arange(n_bands).astype(int)[:,None], standard_idx]  


    mask = standard_idx!=1000
    with numpyro.plate("data_l",n_standard_obs):
        with numpyro.plate("data_i", n_bands):
            full_var_i = (sig_int_b.reshape(n_bands,1)**2. + sig_i**2.)**0.5
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("m_i", dist.StudentT(loc=m_i,df=nu_b.reshape(n_bands,1),scale=full_var_i), obs=standard_mags)



    
    
    

cycles=False
all_student=False
all_normal= True



mag_table,smags=get_data(cycles)

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
if cycles:
    blank_table.add_row({'obj_ID': 'c20_offset '})
blank_table.add_row({'obj_ID': 'sig_intrinsic'})


if all_normal:
    blank_table.add_row({'obj_ID': 'alpha'})

    row_index =blank_table['obj_ID'] == 'alpha'


else:

    blank_table.add_row({'obj_ID': 'nu'})

    row_index =blank_table['obj_ID'] == 'nu'


sample_names = np.array([])
standard_names = np.array([])

max_stand=0
max_samp = 0


for i, pb in enumerate(mag_table):



        all_mags = mag_table[pb]
        mask = all_mags['objID'].isin(use_stars)


        sample_mags   = all_mags.copy()

        # the standards are "special" - they have apparent magnitudes from a
        # model and are used to set the zeropoint for everything else
        standard_mags = all_mags[mask].copy()

        # what are the unique stars
        standards  = standard_mags['objID'].unique()

        standard_names= np.unique(np.append(standard_names,standards))

  

        samples  = sample_mags['objID'].unique()

        sample_names = np.unique(np.append(sample_names,samples))

        max_stand = np.max([max_stand,len(standard_mags)])

        max_samp= np.max([max_samp,len(sample_mags)])




samp_mags = np.zeros((6,max_samp))+1000
samp_ids = np.zeros((6,max_samp)).astype(int)+1000
samp_err = np.zeros((6,max_samp))+1000

stand_mags = np.zeros((6,max_stand))+1000
stand_ids = np.zeros((6,max_stand)).astype(int)+1000
stand_err = np.zeros((6,max_stand))+1000

if cycles:
    cycle_samp_ids = np.zeros((6,max_samp)) + 10000
    cycle_stand_ids = np.zeros((6,max_stand)) + 10000

nstandards = len(standard_names)
standard_ind = list(range(nstandards))
standard_map = dict(zip(standard_names, standard_ind))

nsamples= len(sample_names)
sample_ind = list(range(nsamples))
sample_map = dict(zip(sample_names, sample_ind))

index_standards = {value:key for key, value in standard_map.items()}
index_samples = {value:key for key, value in sample_map.items()}

true_mags = np.zeros((6,len(standard_names)))


zpt_est= np.zeros(6)

for i, pb in enumerate(mag_table):



        all_mags = mag_table[pb]
        mask = all_mags['objID'].isin(use_stars)

        


        sample_mags   = all_mags.copy()

        standard_mags = all_mags[mask].copy()


        stand_ids[i,:len(standard_mags)] = [standard_map[x] for x in standard_mags['objID']]
        stand_mags[i,:len(standard_mags)] = standard_mags[ref]
        stand_err[i,:len(standard_mags)] = standard_mags[dref]

        samp_ids[i,:len(sample_mags)]=[sample_map[x] for x in sample_mags['objID']]
        samp_mags[i,:len(sample_mags)] = sample_mags[ref]
        samp_err[i,:len(sample_mags)] = sample_mags[dref]



        true_mags[i,:]  = np.array([smags.loc[x.replace('-','').lower(), pb] for x in standard_names])


        if cycles:
            cycle_stand_ids[i,:len(standard_mags)] =standard_mags['cycle']
            cycle_samp_ids[i,:len(sample_mags)] =sample_mags['cycle']

  


        zpt_est[i] = np.average(true_mags[i,stand_ids[i,:len(standard_mags)]]-stand_mags[i,:len(standard_mags)])

if all_student:
    nuts_kernel = NUTS(phot_cal_model_all_student,adapt_step_size=True)
elif all_normal:
    nuts_kernel = NUTS(phot_cal_model_all_normal,adapt_step_size=True)
elif cycles:
    nuts_kernel = NUTS(phot_cal_model_cycles,adapt_step_size=True)

else:
    nuts_kernel = NUTS(phot_cal_model,adapt_step_size=True)

mcmc = MCMC(nuts_kernel, num_samples=2000, num_warmup=2000,num_chains=4)
rng_key = random.PRNGKey(0)


if cycles:
    mcmc.run(rng_key,jnp.asarray(samp_ids),jnp.asarray(stand_ids),mag_app_i=jnp.asarray(true_mags),sig_i=jnp.asarray(stand_err),sig_j=jnp.asarray(samp_err),sample_mags=jnp.asarray(samp_mags),standard_mags=jnp.asarray(stand_mags),zpt_est=zpt_est,samp_cycles=jnp.asarray(cycle_samp_ids),stand_cycles=jnp.asarray(cycle_stand_ids))

else:
    mcmc.run(rng_key,jnp.asarray(samp_ids),jnp.asarray(stand_ids),mag_app_i=jnp.asarray(true_mags),sig_i=jnp.asarray(stand_err),sig_j=jnp.asarray(samp_err),sample_mags=jnp.asarray(samp_mags),standard_mags=jnp.asarray(stand_mags),zpt_est=zpt_est)

mcmc.print_summary()
samps=mcmc.get_samples()


mag_samps= samps['mag_app_j']
mags=np.mean(mag_samps,axis=0).reshape(6,len(sample_names)+1).T[:len(sample_names),:]
mags_err=np.std(mag_samps,axis=0).reshape(6,len(sample_names)+1).T[:len(sample_names),:]
sig_int = np.mean(samps['sig_intrinsic'],axis=0)
sig_int_err = np.mean(samps['sig_intrinsic'],axis=0)

zp= np.mean(samps['zeropoint'],axis=0)
zp_err= np.std(samps['zeropoint'],axis=0)

if cycles:
    c20_offset = np.mean(samps['c20_offset'],axis=0)
    c20_offset_err = np.std(samps['c20_offset'],axis=0)

if all_normal:
    alpha = np.mean(samps['alpha'])
    alpha_err =np.std(samps['alpha'])
else:
    nu= np.mean(samps['nu'],axis=0)
    nu_err= np.std(samps['nu'],axis=0)

name_map = at.Table.read('name_map.dat', names=['old','new'], format='ascii.no_header')
name_map = dict(zip(name_map['old'], name_map['new']))

for i, n in enumerate(sample_names):
    if n.startswith('SDSS-J'):
        n = n.split('.')[0].replace('-','')
    elif n.startswith('WD'):
        n = n.replace('WD-','wd').split('-')[0].split('+')[0]
    else:
        pass
    n = n.lower().replace('-','')
    n = name_map.get(n, n)
    sample_names[i] = n


                    

pbs = ['F160W','F275W','F336W','F475W','F625W','F775W']

for i,pb in enumerate(pbs):

    blank_table[pb][0]= zp[i]
    blank_table['d'+pb][0]= zp_err[i]

    if cycles:
        a=1

        if c20_offset_err[i]<0.9:
            blank_table[pb][1]= c20_offset[i]
            blank_table['d'+pb][1]= c20_offset_err[i]
        else:
            blank_table[pb][1]= np.nan
            blank_table['d'+pb][1]= np.nan
    elif all_normal:
        a=0

        if i==0:
            blank_table[pb][2]= alpha
            blank_table['d'+pb][2]= alpha_err

        else:
            blank_table[pb][2]= np.nan
            blank_table['d'+pb][2]= np.nan

    else:
        a=0


    blank_table[pb][a+1]= sig_int[i]
    blank_table['d'+pb][a+1]= sig_int_err[i]
    
    if all_normal==False:
        if i==0:
            if all_student:
                blank_table[pb][a+2] = nu[i]
                blank_table['d'+pb][a+2]= nu_err[i]

            elif cycles:
                blank_table[pb][a+2] = nu[i]
                blank_table['d'+pb][a+2]= nu_err[i]

            else:
                blank_table[pb][a+2] = np.nan
                blank_table['d'+pb][a+2]= np.nan

        else:
            blank_table[pb][a+2] = nu[i]
            blank_table['d'+pb][a+2]= nu_err[i]

    for j in range(len(sample_names)):

        if sample_names[j] not in list(blank_table['obj_ID']):
            blank_table.add_row({'obj_ID': sample_names[j]})

        row_index =blank_table['obj_ID'] == sample_names[j]

        blank_table[pb][row_index] = mags[j,i]
        blank_table['d'+pb][row_index] = mags_err[j,i]

                   


cols = blank_table.colnames
for c in cols:
    if blank_table[c].dtype == np.float64:
        blank_table[c].format = '%.6f'

if all_student:
    blank_table.write('vec_new_all_student_cal.dat',format='ascii.fixed_width',delimiter='  ',overwrite=True)
elif cycles:
    blank_table.write('vec_new_cylces.dat',format='ascii.fixed_width',delimiter='  ',overwrite=True)
elif all_normal:
     blank_table.write('crnl.dat',format='ascii.fixed_width',delimiter='  ',overwrite=True)
else:
     blank_table.write('vec_new_cal.dat',format='ascii.fixed_width',delimiter='  ',overwrite=True)

print(blank_table)
