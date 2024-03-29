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

def main():
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

        print(mag_table[pb])

    # init some structure to store the results for each passband
    result_table = OrderedDict()
    # drop some fields we do not need from the results
    drop_fields = ['hdi_3%','hdi_97%','mcse_mean','mcse_sd','ess_bulk','ess_tail','r_hat']
    # keep a track of all_objects
    all_objects = set()
    # and variable names
    #var_names = ['zeropoint', 'c20_offset', 'sig_intrinsic', 'nu']
    var_names = ['zeropoint', 'sig_intrinsic', 'nu']
    nvar = len(var_names)

    # work on each passband in the table
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

        with pm.Model() as hierarchical:
            # parameters from intrinsic dispersion and the zeropoint and cycle 20 offset in each passband
            sig_int  = pm.HalfCauchy('sig_{}'.format(pb), beta=1)
            zpt      = pm.Normal('zpt_{}'.format(pb), mu=zpt_est, sigma=1)

            n_plot_vars = nvar
            out_vars = var_names.copy()

            # the instrumental magnitudes of the standards (i) is the apparent magnitudes - the zeropoint
            mag_inst_i = mag_app_i[standard_idx] - zpt

            # create parameters for the apparent magntiudes for the sample stars - we want to infer these numbers
            mag_app_j  = pm.Uniform('mag_app_{}_j'.format(pb), lower=8, upper=25, shape=nsamples)

            mag_inst_j = mag_app_j[sample_idx] - zpt

            if pb != 'F160W':
                # the UVIS channel can have cosmic rays, and since we're using the individual FLCs
                # and not the DRCs, these are not rejected (deliberately as CR rejection seems to change the PSF)
                # we need some way of accounting for these outliers, so we use a likelihood that is a Student-T
                # we therefore need a nuisance parameter for the DOF of the student-T distribution in each passband
                nu       = pm.HalfCauchy('nu', beta=3)
                # the actual uncertainty is the observed uncertainty and the intrinsic dispersion in quadrature
                full_var_i = (sig_int**2. + standard_mags[dref]**2.)
                full_var_j = (sig_int**2. + sample_mags[dref]**2.)

                likelihood_sig_zpt = pm.StudentT('likelihood_sig_zpt', mu=mag_inst_i, nu=nu, lam=1./full_var_i, observed=standard_mags[ref])
                # the likelihood of the apparent mags and dispersion given the sample instrumental mags and applying the zeropoint
                likelihood_mag_sig = pm.StudentT('likelihood_mag_sig', mu=mag_inst_j, nu=nu, lam=1./full_var_j, observed=sample_mags[ref])

            else:
                # while there are also cosmic rays in the IR channel, we're less affected by them
                # because we get multiple reads for each FLC exposure and combine them
                # the actual uncertainty is the observed uncertainty and the intrinsic dispersion in quadrature
                full_sd_i = (sig_int**2. + standard_mags[dref]**2.)**0.5
                full_sd_j = (sig_int**2. + sample_mags[dref]**2.)**0.5

                likelihood_sig_zpt = pm.Normal('likelihood_sig_zpt', mu=mag_inst_i, sigma=full_sd_i, observed=standard_mags[ref])
                # the likelihood of the apparent mags and dispersion given the sample instrumental mags and applying the zeropoint
                likelihood_mag_sig = pm.Normal('likelihood_mag_sig', mu=mag_inst_j, sigma=full_sd_j, observed=sample_mags[ref])

                # we don't have nu so fix the number of variables
                n_plot_vars -= 1
                out_vars.remove('nu')

            likelihood = pm.math.concatenate([likelihood_mag_sig, likelihood_sig_zpt])

            # run the MCMC
            print("\n\nRunning {}".format(pb))
            trace = pm.sample(cores=2,  draws=40000, tune=20000, init='advi+adapt_diag')

            # make a plot for sanity checking
            fig2, axs = plt.subplots(nrows=n_plot_vars+1, ncols=2)
            pm.plot_trace(trace, axes=axs)
            fig2.savefig('Figures/htrace_ILAPHv{}{}_{}.pdf'.format(ilaph_version, suffix,pb))

            # get the results
            out = pm.summary(trace)
            print('SUMMARY:',out)

            old_names = list(out.index)
            print(old_names)
            new_names = []
            for name in old_names:
                if name.startswith('mag_app_{}_j'.format(pb)):
                    if len(name.split('_')[3])==4:
                        index = int(name.split('_')[3][2])
                
                    else:
                        index=int(name.split('_')[3][2:4])
                    new_names.append(index_sample[index])
                elif name in out_vars:
                    new_names.append(name)
                elif name.startswith('zpt'):
                    new_names.append('zeropoint')
                elif name.startswith('sig'):
                    new_names.append('sig_intrinsic')
                else:
                    new_names.append('c20_off')

            out['objID'] = new_names
            out.set_index('objID', inplace=True)
            out.drop(drop_fields, axis=1, inplace=True)
            columns = {'mean':pb, 'sd':'d'+pb}
            out.rename(columns=columns, inplace=True)
            print(out)
            result_table[pb] = out
    out_pb_order = ['F275W', 'F336W', 'F475W', 'F625W', 'F775W', 'F160W']
    pbcolor      = ['purple', 'blue', 'green', 'red', 'orange', 'black']
    out = None
    for pb in out_pb_order:
        this_pb = result_table.get(pb, None)
        if this_pb is None:
            continue

        if out is None:
            out = this_pb
        else:
            #out = pd.concat([out, this_pb], axis=1, sort=True, join='outer')
            out = pd.concat([out, this_pb], axis=1, sort=True)
    if out is None:
        message = 'No data was processed in any filter for any objects'
        raise RuntimeError(message)

    # sort the output rows
    sorter = var_names + sorted(list(all_objects))
    objID = out.index.tolist()
    out['obj_ID'] = objID
    out['obj_ID'] = out['obj_ID'].astype("category")
    out['obj_ID'].cat.set_categories(sorter, inplace=True)
    out.sort_values(['obj_ID'], inplace=True)
    objID = out.index.tolist()

    # just moves object ID to the first column
    cols = out.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    out = out[cols]

    # convert to astropy table to pretty print output
    out = at.Table.from_pandas(out)

    name_map = at.Table.read('name_map.dat', names=['old','new'], format='ascii.no_header')
    name_map = dict(zip(name_map['old'], name_map['new']))
    print(name_map)

    out2 = out.copy(copy_data=True)
    objID = out2['obj_ID']
    for i, n in enumerate(objID):
        if n.startswith('SDSS-J'):
            n = n.split('.')[0].replace('-','')
        elif n.startswith('WD'):
            n = n.replace('WD-','wd').split('-')[0].split('+')[0]
        else:
            pass
        n = n.lower().replace('-','')
        n = name_map.get(n, n)
        out2['obj_ID'][i] = n


    cols = out2.colnames
    for c in cols:
        if out2[c].dtype == np.float64:
            out2[c].format = '%.6f'
    print(out2)
    out2.write('WDphot_c25_ILAPHv{}{}.dat'.format(ilaph_version, suffix), format='ascii.fixed_width', delimiter='  ', overwrite=True, fill_values=[(ascii.masked, 'NaN')])

    objID = out['obj_ID']
    nobj = len(objID[nvar:])

    fig_big = plt.figure(figsize=(12,12))
    with PdfPages('Figures/ILAPHv{}{}_c25.pdf'.format(ilaph_version, suffix)) as pdf:
        for j, obj in enumerate(objID[nvar:]):
            fig = plt.figure(figsize=(12, 12))
            for i, pb in enumerate(out_pb_order):
                thispb = mag_table[pb]

                mask = (thispb['objID'] == obj)
                if len(thispb[mask]) == 0:
                    message = 'No data for {}'.format(obj)
                    print(message)
                    continue

                ax = fig.add_subplot(4,2,i+1)
                ax_big = fig_big.add_subplot(4, 2, i+1)

                time = thispb[mask]['MJD']
                mag  = thispb[mask][ref]
                err  = thispb[mask][dref]
                #cycle = thispb[mask]['cycle']
                zpt = out[0][pb]
                #c20off = out[1][pb]
                #if np.isnan(c20off) or pb=='F275W':
                #    c20off = 0.
                #ax.errorbar(time, mag+zpt+c20off*cycle, yerr=err, color=pbcolor[i], linestyle='None', marker='o', ms=3)
                ax.errorbar(time, mag+zpt, yerr=err, color=pbcolor[i], linestyle='None', marker='o', ms=3)

                mask2 = (out['obj_ID'] == obj)
                wmag  = out[mask2][pb].data[0]
                werr  = out[mask2]['d'+pb].data[0]
                #sig_int = out[2][pb]
                sig_int = out[1][pb]

                xmin, xmax = ax.get_xlim()
                #ax_big.errorbar(time, mag-wmag+zpt+c20off*cycle, yerr=err, color=pbcolor[i], linestyle='None', marker='o', ms=3, label='')
                ax_big.errorbar(time, mag-wmag+zpt, yerr=err, color=pbcolor[i], linestyle='None', marker='o', ms=3, label='')

                ax.fill_between([xmin, xmax], wmag + sig_int, wmag - sig_int, color='lightgray', label='int disp')
                ax.fill_between([xmin, xmax],  wmag+werr, wmag-werr, color='gray', label='ucer')
                ax.axhline(wmag, color=pbcolor[i], linestyle='--', lw=3, label='{:.3f}+/-{:.3f}, {:.3f}'.format(wmag, werr, sig_int))
                ax.legend(frameon=True, fontsize='x-small')
                ax.set_xlabel('MJD')
                ax.set_ylabel(pb)
                ax.invert_yaxis()
                if j == nobj-1:
                    xmin, xmax = ax_big.get_xlim()
                    ax_big.fill_between([xmin, xmax], sig_int, -sig_int, color='lightgray')
                    ax_big.axhline(0., color=pbcolor[i], linestyle='--', lw=3, label = 'N(0, {:.3f})'.format(sig_int))
                    ax_big.legend(frameon=False, fontsize='small')
                    ax_big.set_xlabel('MJD')
                    ax_big.set_ylabel(pb)
                    ax_big.invert_yaxis()
            fig.suptitle(obj)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        fig_big.tight_layout()
        fig_big.savefig('Figures/ILAPHv{}{}_combined.pdf'.format(ilaph_version, suffix))
        plt.close(fig_big)









if __name__=='__main__':
    sys.exit(main())
