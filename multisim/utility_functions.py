# -*- coding: utf-8 -*-
"""
@author: Johannes Elfner <johannes.elfner@googlemail.com>
Date: Oct 2017
"""

from functools import wraps as _wraps
import os
import glob as _glob
import shutil as _shutil
import sys
import types as _types

import numpy as np
import pandas as pd
import matplotlib as _mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as _mal
from matplotlib.collections import LineCollection as _LineCollection
import matplotlib.patheffects as _pe
from matplotlib.animation import FuncAnimation as _FuncAnimation
import re as _re
from scipy import stats

# from mpl_toolkits.mplot3d import Axes3D

from .simenv import SimEnv
from .precomp_funs import cp_water as _cp_water, rho_water as _rho_water

# ignore FutureWarnings:
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# if toolbox is installed, import it and set bool to enable overloading
# extracted functions with the functions from toolbox
try:
    import toolbox as tb

    _toolbox_installed = True
except ModuleNotFoundError:
    tb = None
    _toolbox_installed = False


def load_sim_results(
    filename, path=r'.\results\\', keys='all', multiindex=True
):
    # check for correct path layout:
    if (path is not None) and (len(path) > 0) and (path[-1] != '\\'):
        path = path + '\\'
    # check for correct key structuring:
    if keys != 'all':
        keys = list(keys) if isinstance(keys, tuple) else keys
        keys = [keys] if not isinstance(keys, list) else keys
        _keys = []
        for key in keys:
            _keys.append('/' + key if key[0] != '/' else key)
    else:
        _keys = keys

    # open store in read only mode:
    with pd.HDFStore(path + filename, mode='r') as store:
        # iterate over all groups and save them to a dict of dfs
        dfs = {}
        for p, g, lvs in store.walk():  # iterate over paths
            if p != '':
                dfs[p[1:]] = {}  # add subdict to path without leading /
                for lv in lvs:  # iterate over subgroups
                    key = p + '/' + lv
                    if _keys == 'all' or key in _keys:
                        dfs[p[1:]][lv] = store[key]
            else:
                # dfs[lvs[0]] = {lvs[0]: store[p + '/' + lvs[0]]}
                if _keys == 'all' or p + '/' + lvs[0] in _keys:
                    dfs[lvs[0]] = store[p + '/' + lvs[0]]
    # delete empty dict entries (no keys requested for these keys)
    empty_cols = [k for k, v in dfs.items() if len(v) == 0]  # find empty keys
    for col in empty_cols:  # pop empty cols
        dfs.pop(col, None)
    if len(dfs) == 0:
        raise ValueError('no matching key found!')
    # if multiindex dataframe is requested
    if multiindex:
        # generate keys for concatenating
        mi_tuples = []
        dfs_cc = {}
        for k, v in dfs.items():
            # skip simvecs, since these are NOT resampled!
            if 'sim_vecs' not in k:
                for k2, v2 in v.items():  # inner loop
                    mi_tuples.append((k, k2))
                    dfs_cc[(k, k2)] = v2  # unpack 2 level dict to 1 level dict
        # concat all values
        df_mi = pd.concat(dfs_cc.values(), keys=mi_tuples, axis=1)

        if 'sim_vecs' in dfs:
            return {'data': df_mi, 'sim_vecs': dfs['sim_vecs']}
        else:
            return df_mi
    return dfs


def load_results_by_name(
    set_identifier='LHS',
    set_numbers=[0, 1],
    set_name='flexplant_base',
    keys='all',
    result_file_ending='.h5',
    sub_dir='.\\finished',
    base_path=r'E:\Sim_results\\',
):
    assert set_numbers is None or isinstance(set_numbers, (int, list, tuple))
    if set_numbers is not None and isinstance(set_numbers, int):
        set_numbers = [set_numbers]
    if set_numbers is None:
        print('retrieving ALL RESULTS! WARNING, THIS MAY TAKE A LOOONG TIME!')
    _pth = os.path.abspath(
        '{0}\\{1}\\{2}\\'.format(base_path, set_name, sub_dir)
    )
    assert os.path.isdir(_pth), 'path not found:\n{0}'.format(_pth)
    ldir = os.listdir(_pth)  # get all items in path
    # get all dirs
    _subdirs = [
        sd
        for sd in ldir
        if os.path.isdir(os.path.abspath(_pth + '\\' + sd))
        and set_identifier in sd
    ]
    assert len(_subdirs) > 0, 'no simulations found with set_identifier'
    if set_numbers is not None:
        # if set number is None, get ALL results. WARNING, SLOW! else get just
        # some which match set number and set identifier:
        _sn_sdirs = [
            sd for sd in ldir if int(_re.findall(r'\d+', sd)[0]) in set_numbers
        ]
        assert len(_sn_sdirs) > 0, 'no simulations found with set_numbers'
    # generate paths to sim result files in each selected subdir:
    sim_res_fns = [
        _glob.glob(_pth + '\\' + sd + '\\*' + result_file_ending)[0]
        for sd in _sn_sdirs
    ]
    # load the sim res files
    sim_res = {}
    for i in range(len(sim_res_fns)):
        sim_res[int(_re.findall(r'\d+', _sn_sdirs[i])[0])] = load_sim_results(
            sim_res_fns[i], path='', keys=keys
        )
    return sim_res


def load_failed_results(
    file, base_pth=None, results_folder=r'E:\Sim_results\\'
):
    base_pth = '' if base_pth is None else base_pth
    with pd.HDFStore(base_pth + results_folder + file, mode='r') as sr0:
        sr0_data = {}
        for k in sr0.keys():
            _, k1, k2 = k.split('/')
            if k1 in sr0_data:
                sr0_data[k1][k2] = sr0[k]
            else:
                sr0_data[k1] = {k2: sr0[k]}
    return sr0_data


def package_results(
    name_feature='LHS',
    path=r'E:\Sim_results\\',
    move_to=r'.\finished\\',
    ndate_chars=20,
):
    """
    Package simulation results into folders.

    The `name_feature` indicator should be unique to each **set** of
    simulations and **exclude** numberings within a set. F.i. if the naming
    scheme of of the set to pack is
    `2020-03-17T15-29-11_flexi_LHS124_winter_electricity`, a good identifier
    would be either `LHS` is no other sets include `LHS` or
    `winter_electricity` if other sets include `LHS`. The numbering, in this
    case `124` should be excluded.
    """
    # if move to and path do not have closing slashes, add them
    path = (
        path + '\\'
        if path[-1] not in ('\\', '\\\\', '/', '//', '////')
        else path
    )
    move_to = (
        move_to + '\\'
        if move_to[-1] not in ('\\', '\\\\', '/', '//', '////')
        else move_to
    )
    # check if move to path is relative:
    rel_moveto = True if not os.path.isabs(move_to) else False
    # find h5 files and report csvs (both only exist if the sim. finished
    # successfully)
    h5_fls = _glob.glob('{0}*{1}*.h5'.format(path, name_feature))
    rprt_fls = _glob.glob('{0}*{1}*_REPORT.csv'.format(path, name_feature))
    # and also find log files if they exist:
    log_fls = _glob.glob('{0}*{1}*.log'.format(path, name_feature))
    # loop over h5 files:
    unresolved_files = []
    pending_sims = []
    for fl in h5_fls:
        # get base filepath without ending and/or report, bc etc. suffixes
        # and construct report and bcs path and log identifer from it
        fpath_base = fl.split('.h5')[0]
        fpath_rprt = fpath_base + '_REPORT.csv'
        fpath_bcs = fpath_base + '_BCs.csv'
        # get identifier for log files by splitting the file name at the nfeat
        log_idtfr = (
            name_feature + os.path.basename(fpath_base).split(name_feature)[1]
        )
        # find corresponding log file (if any)
        fpath_log = [lfl for lfl in log_fls if log_idtfr in lfl]
        # extract if any found, else set False
        fpath_log = fpath_log[0] if len(fpath_log) == 1 else False
        # extract new names from log name if found, else just strip leading
        # date by given number of chars:
        new_name_prefix = (
            os.path.basename(fpath_log).split('.')[0]
            if fpath_log
            else os.path.basename(fpath_base)[ndate_chars:]
        )
        # extract new folder name:
        new_dirname = name_feature + new_name_prefix.split(name_feature)[1]
        # check if report file exists. if not, set to unresolved and skip file
        # if not os.path.isfile(fpath_rprt):
        if fpath_rprt not in rprt_fls:
            unresolved_files.append(fpath_base)
            continue
        # also check if there is a file with the ending h5_tmp and skip if yes.
        # most probably there is still a sim running when this is true
        if os.path.isfile(fpath_base + '.h5_tmp'):
            pending_sims.append(fpath_base)
            continue  # skip file
        # now that report and h5 exist, try copying all files to a directory
        # consisting of the move_to path + subdir with log idtfr (if log was
        # found). path creation is depending on rel/abs move_to path
        if rel_moveto:
            move_to_pth = os.path.abspath(
                '{0}{1}{2}'.format(path, move_to, new_dirname)
            )
        else:  # if moveto is abspath, not primary path needed
            move_to_pth = os.path.abspath(
                '{0}{1}'.format(move_to, new_dirname)
            )
        move_to_fl = '{0}\\{{0}}{{1}}'.format(move_to_pth)
        # check if path exists and if not, create it:
        if not os.path.exists(move_to_pth):
            # try making the path and skip if already existing to avoid
            # accidentally overwriting results
            os.makedirs(move_to_pth)
        else:
            print('folder already existing, skipping {0}'.format(fl))
            continue
        # copy and rename (drop date string) h5
        # old_dn, old_fn = os.path.split()
        _shutil.move(fl, move_to_fl.format(new_name_prefix, '.h5'))
        # copy and rename (drop date string) report
        _shutil.move(
            fpath_rprt, move_to_fl.format(new_name_prefix, '_REPORT.csv')
        )
        try:  # try to copy BCs (bcs not checked for before, thus try/exc)
            _shutil.move(  # and rename (drop date string)
                fpath_bcs, move_to_fl.format(new_name_prefix, '_BCs.csv')
            )
        except BaseException as e:
            print(fpath_bcs, e)
        if fpath_log:  # copy log
            _shutil.move(fpath_log, move_to_pth)
    # get running sims:
    h5temp_fls = _glob.glob('{0}*{1}*.h5_tmp'.format(path, name_feature))
    pending_sims.append(h5temp_fls)
    print(
        'Packaged results: {0}\nUnresolved files: {1}\n'
        'Pending Simulations: {2}'.format(
            len(h5_fls) - len(unresolved_files),
            len(unresolved_files),
            len(pending_sims[0]),
        )
    )
    return {'unresolved': unresolved_files, 'pending': pending_sims[0]}


def add_line_to_log(line, filename='log.txt', path=''):
    with open(path + filename, 'a') as f:
        f.write('{}\n'.format(line))


def make_boundarycond_timeseries(
    bc_array,
    bc_freq,
    ts_start,
    ts_duration,
    ts_freq,
    bc_too_short='error',
    downsample_method='mean',
    upsample_method='ffill',
):

    # check if bc_array is of correct type:
    err_str = (
        'Boundary conditions array `bc_array` has to be a one '
        'dimensional numpy array!'
    )
    assert isinstance(bc_array, np.ndarray), err_str
    assert bc_array.ndim == 1, err_str
    # check if bc_freq is of correct type:
    err_str = (
        'Boundary conditions arrays frequency `bc_freq` has to be '
        'a non-array float or integer value of the unit [Hz]!'
    )
    assert type(bc_freq) == float or type(bc_freq) == int, err_str
    # check if ts_start is of correct type:
    err_str = (
        'Timeseries starting date has to be of the ISO-date format '
        '\'2018-04-02 00:00:00\' (if no HH:MM:ss given, this will '
        'by default set to 00:00:00) and must be a string type!'
    )
    assert type(ts_start) == str, err_str
    # check if ts_duration is of correct type:
    err_str = (
        'Timeseries duration `ts_duration` has to be a non-array float '
        'or integer value of the unit [s]!'
    )
    assert type(ts_duration) == float or type(ts_duration) == int, err_str
    # check if ts_freq is of correct type:
    err_str = (
        'Timeseries frequency `ts_freq` has to be a non-array float '
        'or integer value of the unit [Hz]!'
    )
    assert type(ts_freq) == float or type(ts_freq) == int, err_str
    # check if bc_too_short is of correct type:
    err_str = (
        'If boundary conditions array is too short for the timeseries '
        'duration, it can be either used multiple times by passing the '
        'argument `bc_too_short=\'stack\'` or an error will be shown '
        'if no argument or `bc_too_short=\'error\'` are passed!'
    )
    assert type(bc_too_short) == str, err_str
    # check if correct strings are given:
    assert bc_too_short == 'error' or bc_too_short == 'stack', err_str

    # construct frequency strings (since only integer values are allowed for
    # frequencies, using nanoseconds avoids losing precision):
    bc_freq_string = str(int(1 / bc_freq * 1e9)) + 'ns'
    ts_freq_string = str(int(1 / ts_freq * 1e9)) + 'ns'
    # construct time vector for bc_array with ts starting value:
    bc_timevec = pd.date_range(
        start=ts_start, periods=bc_array.shape[0], freq=bc_freq_string
    )
    # save first index of bc_timevec:
    bc_first_idx = bc_timevec[0]
    # merge array and timevec to pd.Series:
    bc = pd.Series(data=bc_array, index=bc_timevec)
    # get duration of bc_ts:
    bc_duration = bc.index[-1] - bc.index[0]
    # get wanted ts duration:
    ts_timedelta = pd.Timedelta(seconds=ts_duration)
    # check if bc_ts duration matches ts_duration:
    if bc_duration != ts_timedelta:
        # if greater than ts_duration, bc_ts can be cut off and resampled:
        if bc_duration > ts_timedelta:
            # cut off:
            bc = bc[bc_first_idx : bc_first_idx + ts_timedelta]
            # resample with ts_freq:
            # if downsampling has to be done:
            if ts_freq <= bc_freq:
                bc_ts = bc.resample(ts_freq_string, how=downsample_method)
            # if upsampling has to be done:
            else:
                bc_ts = bc.resample(ts_freq_string, how=upsample_method)
        elif bc_duration < ts_timedelta:
            # if shorter than ts_duration, either throw error or stack:
            if bc_too_short == 'error':
                raise ValueError(
                    'Boundary conditions array is shorter than '
                    'the required timeseries duration! Either '
                    'use option `bc_too_short=\'stack\'` or pass '
                    'a longer array!'
                )
            else:
                # calculate number of stacks:
                n_stacks = np.ceil(ts_timedelta / bc_duration).astype(int)
                # make copy of bc_array:
                bc_array_short = bc_array.copy()
                # stack array into new bc_array:
                for i in range(n_stacks):
                    bc_array = np.hstack((bc_array, bc_array_short))
                # make pandas Series again:
                # new timevec:
                bc_timevec = pd.date_range(
                    start=ts_start,
                    periods=bc_array.shape[0],
                    freq=bc_freq_string,
                )
                # new pd.Series:
                bc = pd.Series(data=bc_array, index=bc_timevec)
                # cut to final length:
                bc = bc[bc_first_idx : bc_first_idx + ts_timedelta]
                # resample with ts_freq:
                # if downsampling has to be done:
                if ts_freq <= bc_freq:
                    bc_ts = bc.resample(ts_freq_string, how=downsample_method)
                # if upsampling has to be done:
                else:
                    bc_ts = bc.resample(ts_freq_string, how=upsample_method)

    return bc_ts


def process_unevenly_spaced_timeseries(*, data, freq, how):
    assert (
        isinstance(data, (pd.Series, pd.DataFrame))
        and type(data.index) == pd.DatetimeIndex
    ), (
        '`data` must be given as a pandas Series or DataFrame with a '
        'DatetimeIndex.'
    )
    assert (
        type(freq) == str
    ), 'The resampling frequency must be given with `freq` as a string.'
    how_list = ['interpolate', 'forward_fill', 'backward_fill']
    assert type(how) == str and how in how_list, (
        'The resampling method has to be given with `how=X`, where X is one '
        'of the following: ' + str(how_list) + '\n'
        '    \'interpolate\': Linear interpolation of the values that need '
        ' to be filled. Best used for constantly changing values with a '
        'finite slope (no or minor jumps) like temperatures.\n'
        '    \'forward_fill\': Missing values will be filled by propagating '
        'the last valid observation forward. Best used for values changing '
        'abruptly, like massflows.\n'
        '    \'backward_fill\': Missing values will be filled by propagating '
        'the next valid observation backwards. Best used for values changing '
        'abruptly, like massflows. In most cases \'forward_fill\' is '
        'recommended. \'backward_fill\' is only useful, if forward filling is '
        'not giving a correct result.'
    )

    if how == 'interpolate':
        return data.resample(freq).mean().interpolate()
    elif how == 'forward_fill':
        return data.resample(freq).ffill()
    elif how == 'backward_fill':
        return data.resample(freq).bfill()


def interp_pipe_specs(A_i, pipe_type='EN10255_medium', base_pth=None):
    if base_pth is not None:
        ps = pd.read_pickle(base_pth + 'data_tables/pipe_specs.pkl')
    else:
        ps = pd.read_pickle(
            os.path.join(
                os.path.dirname(__file__), 'data_tables\\pipe_specs.pkl'
            )
        )
    ps_tp = ps.loc[:, pipe_type].copy().T
    # set a high and low value with the last/first gradient to get some
    # "extra"polation:
    grad_hi = ps_tp.iloc[-1] / ps_tp.iloc[-2]
    grad_lo = ps_tp.iloc[0] / ps_tp.iloc[1]
    new_val_hi = 0.1
    new_val_lo = 1e-5
    # get resulting values at new val
    rel_increase_hi = new_val_hi / ps_tp.iloc[-1].A_i * grad_hi / grad_hi.A_i
    rel_increase_lo = new_val_lo / ps_tp.iloc[0].A_i * grad_lo / grad_lo.A_i
    # set Ai as index
    ps_tp = ps_tp.set_index('A_i')
    # set new values
    ps_tp.loc[new_val_hi] = ps_tp.iloc[-1] * rel_increase_hi.iloc[1:]
    ps_tp.loc[new_val_lo] = ps_tp.iloc[0] * rel_increase_lo.iloc[1:]

    if isinstance(A_i, (int, float)):
        A_i = [A_i]
    elif not isinstance(A_i, (list, tuple)):
        raise TypeError
    for ai in A_i:
        if ai in (new_val_lo, new_val_hi):
            continue
        ps_tp.loc[ai] = np.nan
    # cubic interp for cubic relation of diameter to area
    ps_tp = ps_tp.sort_index().interpolate(method='cubic', limit_area='inside')
    # calculate d_i, d_o, A_wall and s directly:
    ps_tp.loc[A_i, 'd_i'] = np.sqrt(ps_tp.loc[A_i].index.values * 4 / np.pi)
    ps_tp.loc[A_i, 'd_o'] = np.sqrt(ps_tp.loc[A_i, 'A_o'] * 4 / np.pi)
    ps_tp.loc[A_i, 's'] = (ps_tp.loc[A_i, 'd_o'] - ps_tp.loc[A_i, 'd_i']) / 2
    ps_tp.loc[A_i, 'A_wall'] = (
        ps_tp.loc[A_i, 'A_o'] - ps_tp.loc[A_i].index.values
    )
    assert not ps_tp.isna().any().any(), 'Nan found in pipe specs!'
    return ps_tp.loc[A_i, :].reset_index()


def adjust_pipes(pspecs, base_path=None):
    err_str = (
        '`pspecs` must be a dict. Keys are part identifiers, values '
        'are lists/tuples consisting of a reference pipe-spec dict in the '
        'first element and a multiplication factor for the inner '
        'cross section area in the second element, f.i.\n'
        '{0}'.format(
            repr(
                {
                    'pipe1': (
                        {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN50'}},
                        1.753,
                    )
                }
            )
        )
    )
    assert isinstance(pspecs, dict), err_str
    for v in pspecs.values():
        assert isinstance(v, (list, tuple)), err_str
        assert len(v) == 2, err_str
        assert isinstance(v[0], dict), err_str
        assert isinstance(v[1], (float, int)), err_str
    # read pipe specs reference table
    ps_table = pd.read_pickle(
        os.path.join(os.path.dirname(__file__), 'data_tables\\pipe_specs.pkl')
    )
    # extract A_i from table and multiplicate values by mult factor
    ai_vals = {
        k: ps_table.loc['A_i', tuple(v[0]['all'].values())] * v[1]
        for k, v in pspecs.items()
    }
    pipe_specs = interp_pipe_specs(
        A_i=list(ai_vals.values()), base_pth=base_path
    )
    _pspecs = {
        k: {'all': vals[1].to_dict()}
        for k, vals in zip(ai_vals.keys(), pipe_specs.iterrows())
    }
    for k, v in _pspecs.items():
        v['all']['pipe_type'] = pspecs[k][0]['all']['pipe_type']
    return _pspecs


def database_import(*, standard_db=True, path=None, which='all'):
    """
    This function imports part, material or other databases from pickled
    DataFrames and returns them in a dictionary.
    """

    dbs = {}  # make  dict to save databases in

    if standard_db:  # import databases implemented in the simulation env.
        # get current path. To do this, check if __file__ exists (for example
        # when using py2exe it does not), else try using sys.argv:
        try:  # if not frozen as executable
            pth = os.path.dirname(os.path.realpath(__file__))
        except NameError:  # if frozen as executable
            pth = os.path.dirname(os.path.realpath(sys.argv[0]))
        pth = pth + r'\data_tables'  # append datatables path
        # find path to all pickled databases
        files = _glob.glob(os.path.join(pth, '*.pkl'))
        if which == 'all':  # import all databases
            for file in files:
                #  get name by splitting strings:
                db_name = file.split('\\')[-1].split('.')[0]
                dbs[db_name] = pd.read_pickle(file)
        else:  # else if not all shall be imported:
            for db in which:
                pth = pth + db + '.pkl'
                db_name = pth.split('\\')[-1].split('.')[0]
                dbs[db_name] = pd.read_pickle(file)
    else:  # else if not standard databases
        if not path:  # if path not given, get it from dialog
            from tkinter.filedialog import askdirectory

            pth = askdirectory(initialdir="D:/LRZ-SnS/Objekte")
        if which == 'all':  # if all databases should be loaded from path
            files = _glob.glob(os.path.join(pth, '*.pkl'))  # get all pickles
            for file in files:
                #  get name by splitting strings:
                db_name = file.split('\\')[-1].split('.')[0]
                dbs[db_name] = pd.read_pickle(file)
        else:  # else if not all shall be imported:
            for db in which:
                pth = pth + db + '.pkl'
                db_name = pth.split('\\')[-1].split('.')[0]
                dbs[db_name] = pd.read_pickle(file)

    return dbs  # return database dict


def update_databases(*, path=None, transpose=True):
    if not path:  # if no path was given, update the standard databases
        # get current path. To do this, check if __file__ exists (for example
        # when using py2exe it does not), else try using sys.argv:
        try:  # if not frozen as executable
            pth = os.path.dirname(os.path.realpath(__file__))
        except NameError:  # if frozen as executable
            pth = os.path.dirname(os.path.realpath(sys.argv[0]))
        pth = pth + r'\data_tables'  # append datatables path
        # find path to all pickled databases
        files = _glob.glob(os.path.join(pth, '*.xlsx'))
        for file in files:
            # read description to get start row and col.
            desc = pd.read_excel(file, sheet_name='description')
            start_row = desc.iloc[3, 1] - 1  # get start row (zero indexed)
            start_col = desc.iloc[4, 1] - 1  # get start row (zero indexed)
            # read data and consider start row and col
            idx_col = [x for x in range(int(start_col))]  # make idx col list
            idx_row = [x for x in range(int(start_row))]  # make idx row list
            db = pd.read_excel(  # read the real data
                file, sheet_name='data', index_col=idx_col, header=idx_row
            )
            if transpose:  # if database should be transposed
                db = db.T
            # get target path:
            trgt_pth = file.split('.xl')[0]  # get path before file ending
            db.to_pickle(trgt_pth + '.pkl')  # save to pickle
    else:  # else if specific path is given
        # read description to get start row and col.
        desc = pd.read_excel(path, sheet_name='description')
        start_row = desc.iloc[3, 1] - 1  # get start row (zero indexed)
        start_col = desc.iloc[4, 1] - 1  # get start row (zero indexed)
        # read data and consider start row and col
        idx_col = [x for x in range(int(start_col))]  # make idx col list
        idx_row = [x for x in range(int(start_row))]  # make idx row list
        db = pd.read_excel(  # read the real data
            path, sheet_name='data', index_col=idx_col, header=idx_row
        )
        if transpose:  # if database should be transposed
            db = db.T
        # get target path:
        trgt_pth = path.split('.xl')[0]  # get path before file ending
        db.to_pickle(trgt_pth + '.pkl')  # save to pickle


# %% decorators
# override/monkeypatch function definitions by imported module functions if
# available
def override(toolbox=None, submodule=None, function_name=None):
    assert toolbox is None or isinstance(toolbox, _types.ModuleType), (
        '`toolbox` must be either None if toolbox is not installed/required '
        'or the reference to the toolbox module.'
    )
    assert submodule is None or isinstance(submodule, str), (
        '`submodule` must be either None or the submodule name as str '
        'where to find the requested function.'
    )
    assert function_name is None or isinstance(function_name, str), (
        '`function_name` must be either None or the function name as a '
        'string. In the first case, the function name is assumed to be the '
        'same as the decorated function.'
    )

    def decorator(function):
        @_wraps(function)
        def wrapper(*args, **kwargs):
            if toolbox is None:
                return function(*args, **kwargs)
            else:
                assert (
                    submodule is not None
                ), 'If `toolbox` is not None, submodule must not be None.'
                if function_name is None:  # get function by decorated fun name
                    return getattr(
                        getattr(toolbox, submodule), function.__name__
                    )(*args, **kwargs)
                else:  # get function by explicit name
                    return getattr(getattr(toolbox, submodule), function_name)(
                        *args, **kwargs
                    )

        return wrapper

    return decorator


# %% basic statistics
@override(tb, 'sf')
def r_squared(y, y_hat):
    """
    Berechnet den Determinationskoeffizienten R**2 zwischen den Messdaten `y`
    und den vorhergesagten/simulierten Daten `y_hat`.
    Der Determinationskoeffizient gibt an, welcher Anteil der Varianz durch das
    zugrundeliegende Modell erklärt werden kann.

    `y_hat` kann auch mehrere Datenreihen enthalten. In diesem Fall wird der
    Determinationskoeffizient **je Spalte** berechnet.

    Parameters:
    -----------
    y : np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    """

    assert (
        not np.isnan(y).any().any()
    ), (  # double any for ndim >= 2
        '`y` contains NaNs. Please drop NaNs in y before calculating R^2.'
    )
    # broadcast y to 2D, if input is 2D to correctly calculate r^2 col wise
    if y_hat.ndim == 2:
        y = y[:, None] if y.ndim == 1 else y
    elif y_hat.ndim == 1 and y.ndim == 2:
        y_hat = y_hat[:, None]
    elif y_hat.ndim > 2:
        raise ValueError
    ss_res = ((y - y_hat) ** 2).sum(axis=0)  # residuals
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum()  # total sum of squares
    return 1 - ss_res / ss_tot


@override(tb, 'sf')
def r_squared_adj(p, r_sqrd=None, n=None, y=None, y_hat=None):
    r"""
    Berechnet den bereinigten (engl.: adjusted) Determinationskoeffizienten
    :math:`R_{adj}^2` zwischen den Messdaten `y` und den
    vorhergesagten/simulierten Daten `y_hat`.
    Der Determinationskoeffizient gibt an, welcher Anteil der Varianz durch das
    zugrundeliegende Modell erklärt werden kann.
    Der bereinigte Determinationskoeffizient berücksichtigt zusätzlich die
    Anzahl der Samples **UND** der unabhängigen Variablen. Der Einsatz
    übermäßig vieler Regressoren wird dadurch, insbesondere bei wenig Samples,
    negativ bewertet.
    Das bereinigte/adjusted/adjustierte Bestimmtheitsmaß wird entweder mit
    :math:`R_{adj}^2` oder :math:`\bar{R}^2` gekennzeichet.

    `y_hat` kann auch mehrere Datenreihen enthalten. In diesem Fall wird der
    Determinationskoeffizient **je Spalte** berechnet.

    Parameters:
    -----------
    p : int
        Total number of explanatory variables in the model, including
        higher order and interaction terms.
    r_sqrd : optional, float
        Non-adjusted R-squared value. Only required if `y` and `y_hat` are not
        given.
    n : optional. int
        Number of samples. Only required if `r_sqrd` is given explicitly.
    y : optional. np.array, pd.Series, pd.DataFrame
        Measurement/oberserved data. Only required, if :math:`R^2` is not given
        as argument.
    y_hat : optional. np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data. Only required, if :math:`R^2` is not
        given as argument.

    """

    if r_sqrd is not None:
        assert (n is not None) and (y is None) and (y_hat is None), (
            'If `r_sqrd` is given, `n` must be given and `y` and `y_hat` '
            'must not be given.'
        )
    else:  # if r_sqrd not given
        assert (n is None) and (y is not None) and (y_hat is not None), (
            'If `r_sqrd` is not given, `y` and `y_hat` must not be given and'
            '`n` must not be given and.'
        )
        r_sqrd = r_squared(y, y_hat)  # get R^2
        n = y_hat.shape[0]  # get number of samples
    # get adjusted r_sqrd
    r_sqrd_adj = 1 - (1 - r_sqrd) * (n - 1) / (n - p - 1)
    return r_sqrd_adj


@override(tb, 'sf')
def mean_abs_err(y, y_hat, roll=False, window=None, min_periods=None):
    # auch mean absolute deviation (MAD)
    # broadcast y to 2D, if input is 2D to correctly calculate r^2 col wise
    if y_hat.ndim == 2:
        y = y[:, None] if y.ndim == 1 else y
    elif y_hat.ndim == 1 and y.ndim == 2:
        y_hat = y_hat[:, None]
    elif y_hat.ndim > 2:
        raise ValueError
    if not roll:
        return np.abs(y - y_hat).mean(axis=0)
    else:
        return (
            pd.DataFrame(np.abs(y - y_hat))
            .rolling(window=window, min_periods=min_periods, center=True)
            .mean()
        )


@override(tb, 'sf')
def mean_signed_deviation(y, y_hat, roll=False, window=None, min_periods=None):
    # auch: mean biased error, MBE. Zur Berechnung des normalized mean biased
    # error (NMBE) muss mit normalized_err normalisiert werden, z.B. mit mean.
    # broadcast y to 2D, if input is 2D to correctly calculate r^2 col wise
    if y_hat.ndim == 2:
        y = y[:, None] if y.ndim == 1 else y
    elif y_hat.ndim == 1 and y.ndim == 2:
        y_hat = y_hat[:, None]
    elif y_hat.ndim > 2:
        raise ValueError
    if not roll:
        return (y_hat - y).mean(axis=0)
    else:
        return (
            pd.DataFrame(y_hat - y)
            .rolling(window=window, min_periods=min_periods, center=True)
            .mean()
        )


@override(tb, 'sf')
def mean_squared_err(y, y_hat, roll=False, window=None, min_periods=None):
    # broadcast y to 2D, if input is 2D to correctly calculate r^2 col wise
    if y_hat.ndim == 2:
        y = y[:, None] if y.ndim == 1 else y
    elif y_hat.ndim == 1 and y.ndim == 2:
        y_hat = y_hat[:, None]
    elif y_hat.ndim > 2:
        raise ValueError
    if not roll:
        return ((y - y_hat) ** 2).mean(axis=0)
    else:
        return (
            pd.DataFrame((y - y_hat) ** 2)
            .rolling(window=window, min_periods=min_periods, center=True)
            .mean()
        )


@override(tb, 'sf')
def rmse(y, y_hat, roll=False, window=None, min_periods=None):
    return np.sqrt(
        mean_squared_err(
            y, y_hat, roll=roll, window=window, min_periods=min_periods
        )
    )


@override(tb, 'sf')
def cv_rmse(y, y_hat, roll=False, window=None, min_periods=None):
    return rmse(
        y, y_hat, roll=roll, window=window, min_periods=min_periods
    ) / np.mean(y)


@override(tb, 'sf')
def normalized_err(y, y_hat, err_method='MSE', norm='IQR'):
    assert err_method in ('MSE', 'RMSE', 'MSD', 'MAE')
    assert norm in ('IQR', 'range', 'mean', 'median')

    err_methods = {  # supported error methods:
        'MSE': mean_squared_err,
        'RMSE': rmse,
        'MSD': mean_signed_deviation,
        'MAE': mean_abs_err,
    }
    err_fun = err_methods[err_method]

    if norm == 'mean':
        return err_fun(y, y_hat) / np.mean(y)
    elif norm == 'median':
        return err_fun(y, y_hat) / np.median(y)
    elif norm == 'IQR':
        q3, q1 = np.percentile(y, [75, 25])
        return err_fun(y, y_hat) / (q3 - q1)
    elif norm == 'range':
        desc = pd.Series(y).describe()
        return err_fun(y, y_hat) / (desc['max'] - desc['min'])


def remove_outliers_std(*, data, stdevs=3, fill_value='interpolate'):
    """
    Remove outliers of an array, DataFrame or Series based on the standard
    deviation. A multiple of the standard deviation is used to compute the
    outlier threshold.
    As default, `3 * std` is used as threshold and outliers will be removed and
    interpolated or filled with nan or a defined value.

    """

    assert isinstance(data, (np.ndarray, pd.Series, pd.DataFrame))
    assert isinstance(stdevs, (int, float)) and stdevs > 0
    assert fill_value in ['interpolate', 'nan', 'remove'] or isinstance(
        fill_value, (int, float)
    )

    #    mask = np.abs(data - data.mean()) <= (stdevs * data.std())
    #    use mask!

    if fill_value == 'interpolate':
        data[~(np.abs(data - data.mean()) <= (stdevs * data.std()))] = np.nan
        return data.interpolate(method='index', axis=0)
    elif fill_value == 'nan':
        data[~(np.abs(data - data.mean()) <= (stdevs * data.std()))] = np.nan
        return data
    elif fill_value == 'remove':
        if type(data) != pd.Series or (
            type(data) == np.ndarray and data.ndim > 1
        ):
            print(
                'Caution! If outliers shall be removed from `data`, `data` '
                'needs to be a Series or a one-dimensional np.ndarray. '
                'Otherwise outliers will be filled with `np.nan`.'
            )
        return data[np.abs(data - data.mean()) <= (stdevs * data.std())]
    else:
        data[
            ~(np.abs(data - data.mean()) <= (stdevs * data.std()))
        ] = fill_value
        return data


def remove_outliers_roll(
    *, data, window=3, stdevs=3, fill_value='interpolate'
):
    """
    Remove outliers of a DataFrame or Series based on the difference
    of the data with its rolling median

    """

    # copy to avoid overwriting
    data = data.copy()

    assert isinstance(data, (pd.Series, pd.DataFrame))
    assert isinstance(window, int) and window > 0
    #    assert isinstance(threshold, (int, float)) and threshold > 0
    assert isinstance(stdevs, (int, float)) and stdevs > 0
    assert fill_value in ['interpolate', 'nan', 'remove'] or isinstance(
        fill_value, (int, float)
    )

    rolling_median = (
        data.rolling(window=window, center=True, axis=0)
        .median()
        .fillna(method='bfill')
        .fillna(method='ffill')
    )

    difference = np.abs(data - rolling_median)  # get local diff to median
    deviation = difference.std() * stdevs  # get maximum deviation for outliers
    outliers = difference > deviation  # get outliers of dev. of roll. med. dif

    #    mask = np.abs(data - data.mean()) <= (stdevs * data.std())
    #    use mask!

    if fill_value == 'interpolate':
        data.values[outliers.values] = np.nan  # set outliers to nan
        try:  # some periodindex frequencies are problematic, thus catch error
            # interpolate outliers
            return data.interpolate(
                method='index', limit_area='inside', axis=0
            )
        except TypeError:
            # backup freq:
            freq = data.index.freq
            # convert period index to timestamp
            data.index = data.index.to_timestamp()
            data = data.interpolate(
                method='index', limit_area='inside', axis=0
            )  # interpolate
            # convert back to periodindex
            data.index = data.index.to_period(freq)
            return data
    elif fill_value == 'nan':
        data.values[outliers.values] = np.nan  # set outliers to nan
        return data
    elif fill_value == 'remove':
        if type(data) != pd.Series or (
            type(data) == np.ndarray and data.ndim > 1
        ):
            print(
                'Caution! If outliers shall be removed from `data`, `data` '
                'needs to be a Series or a one-dimensional np.ndarray. '
                'Otherwise outliers will be filled with `np.nan`.'
            )
        return data[~outliers]  # return data without outliers
    else:
        data[outliers] = fill_value  # fill outliers with value
        return data


def rolling_median(*, data, window):
    return data.rolling(window=window, center=True, axis=0).median()


def rolling_centered(
    df,
    window,
    min_periods=0.0,
    method='mean',
    win_type=None,
    on=None,
    axis=0,
    closed='left',
):
    """
    This expands time series rolling method with the possibility to center the
    rolling window. The timeseries needs to be evenly spaced, otherwise
    resampling needs to be performed.

    If `min_periods == 0` (default), rolling data will be calculated until the
    start and end without introducing new NaN, but at reduced window size at
    these locations. For a Series/DataFrame with NaN this results in filling
    `n = (window - 1) / 2` NaN at each end as well as NaN within the Series,
    for example for `window='5min'` two minutes of NaN will be filled.

    If `min_periods == window`, this is equivalent to using a centered rolling
    window without a time index. The values will be dropped at the start and
    end until min periods can be filled.

    If `min_periods < window`, rolling data at the start and end will be NaN
    until min_periods can be satisfied. Usually setting
    `min_periods = window - (window - 1) / 2` results in a Series/DataFrame
    of exactly the same start and end as the original Series/DataFrame.

    If `min_periods=None`, it will be set to the window length.
    `min_periods` can be given as a string, like window, or as an
    integer or float value.

    Parameters:
    -----------
        df : pd.DataFrame or pd.Series
        window : string
            String giving the window size **in time units**, that means a time
            identifier with a corresponding unit, for example
            **`window='1min'`**.
            **Otherwise the default unit of nanoseconds will be used**.
    """

    df = df.copy()  # copy to avoid altering the input df

    assert method in ('mean', 'max', 'min', 'median', 'sum')
    # try to get the frequency if it was lost somewhere (typically while
    # concatenating). if still none, raise error.
    if df.index.freq is None:
        df.index.freq = df.index.inferred_freq
    assert df.index.freq is not None, (
        'df must be an evenly spaced time series. Resample before using this '
        'method.'
    )

    # check if window contains some strings:
    win_err = (
        '`window` seems to be not consisting of a time+unit string. Please '
        'pass something like `window=\'10min\'`.'
    )
    assert isinstance(window, str), win_err
    assert _re.findall(r'[\D]', window) is not None, win_err

    # calculate shift indices:
    shift_idc = int((pd.to_timedelta(window) / df.index.freq) / 2)
    # calculate min periods in int:
    if min_periods is None:
        min_periods = None  # keep none
    elif isinstance(min_periods, str):
        min_periods = int(pd.to_timedelta(min_periods) / df.index.freq)
    else:
        min_periods = int(min_periods)

    # check if window is a multiple of the frequency and if yes, fall back to
    # the easy and loss-less method of doing an integer-index rolling method.
    # This avoids cutting off the end if min_periods < window.
    freq_ratio = pd.to_timedelta(window) / df.index.freq
    if abs(freq_ratio - int(freq_ratio)) < 1e-16:  # is ratio a multiple?
        # fall back to integer index rolling if freq ratio is a full multiple
        # backup index and replace with integer:
        index_bkp = df.index
        df.index = np.arange(df.shape[0])
        # compute window length in integer:
        window = int(pd.to_timedelta(window) / index_bkp.freq)
        if method == 'mean':
            df = df.rolling(
                window=window,
                min_periods=min_periods,
                center=True,
                win_type=win_type,
                on=on,
                axis=axis,
                closed=None,
            ).mean()
            df.index = index_bkp  # reindex with time index
        elif method == 'median':
            df = df.rolling(
                window=window,
                min_periods=min_periods,
                center=True,
                win_type=win_type,
                on=on,
                axis=axis,
                closed=None,
            ).median()
            df.index = index_bkp  # reindex with time index
        elif method == 'sum':
            df = df.rolling(
                window=window,
                min_periods=min_periods,
                center=True,
                win_type=win_type,
                on=on,
                axis=axis,
                closed=None,
            ).sum()
            df.index = index_bkp  # reindex with time index
        elif method == 'min':
            df = df.rolling(
                window=window,
                min_periods=min_periods,
                center=True,
                win_type=win_type,
                on=on,
                axis=axis,
                closed=None,
            ).min()
            df.index = index_bkp  # reindex with time index
        elif method == 'max':
            df = df.rolling(
                window=window,
                min_periods=min_periods,
                center=True,
                win_type=win_type,
                on=on,
                axis=axis,
                closed=None,
            ).max()
            df.index = index_bkp  # reindex with time index
        return df
    else:  # else ratio is not a multiple, do the time index rolling
        if method == 'mean':
            return (
                df.rolling(
                    window=window,
                    min_periods=min_periods,
                    center=False,
                    win_type=win_type,
                    on=on,
                    axis=axis,
                    closed=closed,
                )
                .mean()
                .shift(-shift_idc)
            )
        elif method == 'median':
            return (
                df.rolling(
                    window=window,
                    min_periods=min_periods,
                    center=False,
                    win_type=win_type,
                    on=on,
                    axis=axis,
                    closed=closed,
                )
                .median()
                .shift(-shift_idc)
            )
        elif method == 'sum':
            return (
                df.rolling(
                    window=window,
                    min_periods=min_periods,
                    center=False,
                    win_type=win_type,
                    on=on,
                    axis=axis,
                    closed=closed,
                )
                .sum()
                .shift(-shift_idc)
            )
        elif method == 'min':
            return (
                df.rolling(
                    window=window,
                    min_periods=min_periods,
                    center=False,
                    win_type=win_type,
                    on=on,
                    axis=axis,
                    closed=closed,
                )
                .min()
                .shift(-shift_idc)
            )
        elif method == 'max':
            return (
                df.rolling(
                    window=window,
                    min_periods=min_periods,
                    center=False,
                    win_type=win_type,
                    on=on,
                    axis=axis,
                    closed=closed,
                )
                .max()
                .shift(-shift_idc)
            )


def rolling_right_aligned(df, window, min_periods):
    shift = int(window[:-1]) - 1 if isinstance(window, str) else window - 1
    df = df.rolling(
        window=window, min_periods=min_periods, center=False
    ).shift(-shift)
    return df


def remove_outliers_mad(
    *,
    data,
    threshold=2.5,
    fill_value='interpolate',
    rolling=False,
    window='1H',
    dist_const=1.4826
):
    """
    Outlier removal by robust median estimator.

    Remove outliers of a DataFrame or Series based on the (rolling) median of
    the difference of the data to its (rolling) median.

    This method is preferred over identifying outliers be the standard
    deviation, since the median is highly robust to outliers, while the mean
    and the standard deviation are highly sensitive to outliers [1]_.

    .. [1] https://www.sciencedirect.com/science/article/pii/S0022103113000668
    .. [2] https://docs.oracle.com/cd/E40248_01/epm.1112/cb_statistical/frameset.htm?ch07s02s10s01.html
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        data = pd.Series(data=data)
    else:
        # copy data to avoid altering the inputs
        data = data.copy()

    if not rolling:
        median = data.median()
        absolute_thresh = (
            threshold * dist_const * (data - median).abs().median()
        )
        outliers = np.abs(data - median) > absolute_thresh
    else:
        raise NotImplementedError('Rolling not yet working correctly.')
        median = data.rolling(window=window).median()
        absolute_thresh = (
            threshold
            * dist_const
            * (data - median).abs().rolling(window=window).median()
        )
        outliers = np.abs(data - median) > absolute_thresh

    data[outliers] = np.nan

    if fill_value == 'interpolate':
        return data.interpolate(method='index', axis=0)
    elif fill_value == 'nan':
        return data
    elif fill_value == 'remove':
        if type(data) != pd.Series or (
            type(data) == np.ndarray and data.ndim > 1
        ):
            print(
                'Caution! If outliers shall be removed from `data`, `data` '
                'needs to be a Series or a one-dimensional np.ndarray. '
                'Otherwise outliers will be filled with `np.nan`.'
            )
        return data[~outliers]  # return data without outliers
    else:
        data[outliers] = fill_value  # fill outliers with value
        return data


class NetPlotter(SimEnv):
    def __init__(self):
        print(
            'To use the net plotter, call its staticmethod '
            '`NetPlotter.set_plot_shape()` for all parts to draw!'
        )

    warnings.warn(
        'NetPlotter is mostly deprecated and will be removed in future '
        'versions. Only its `plot_errors` method is currently still '
        'supported.',
        DeprecationWarning,
    )

    @staticmethod
    def set_plot_shape(
        *,
        part,
        start_pos,
        vertex_coords,
        linewidth=None,
        orientation=None,
        invert=None
    ):
        # make plotting dict:
        info_plot = dict()

        # assert and get start position:
        err_str = (
            '`start_pos` must be given as a tuple of length 2 with '
            'the first element giving the x-axis position and the '
            'second element giving the y-axis position of the first '
            'port of the part (in most cases the \'in\'-port) as an '
            'integer or float value,\n'
            'OR as `start_pos=\'auto\'` to automatically select the '
            'position of the connected part IF the connected part was'
            'already defined using the `set_plot_shape()`-method '
            'AND IF the connected port is either the FIRST or the LAST '
            'port of the part,\n'
            'OR as `start_pos=\'port_name\'` where \'port_name\' is '
            'the string name of one of the ports of the part IF the '
            'part connected to this port was already defined!'
        )
        if type(start_pos) == tuple:
            # if given as tuple, the start position is given directly
            assert len(start_pos) == 2, err_str
            assert (
                type(start_pos[0]) == int or type(start_pos[0]) == float
            ), err_str
            assert (
                type(start_pos[1]) == int or type(start_pos[1]) == float
            ), err_str
            info_plot['start_position'] = start_pos
        elif type(start_pos) == str and start_pos == 'auto':
            # elif given as auto, it is determined from connected parts
            assert start_pos == 'auto', err_str
            # loop over all ports to find first port which is connected to
            # another part which is already defined:
            found = False  # bool checker if one was found
            for i in range(len(SimEnv.parts[part].port_names)):
                # get connected part:
                conn_part = SimEnv.port_links[
                    part + ';' + SimEnv.parts[part].port_names[i]
                ]
                # conn part is part;port, split up:
                conn_part, conn_port = conn_part.split(';')
                # check if already defined for plotting and if True break:
                if (
                    hasattr(SimEnv.parts[conn_part], 'info_plot')
                    and SimEnv.parts[conn_part].plot_ready
                ):
                    found = True
                    # save connection info where the position is coming from
                    info_plot['auto_connection'] = {
                        'connected_part': conn_part,
                        'connected_port': conn_port,
                        'own_port': SimEnv.parts[part].port_names[i],
                    }
                    break
            # assert that at least one connected part was already defined by
            # NetPlotter:
            err_str = (
                'If `start_pos=\'auto\'` was chosen, at least one part '
                'connected to one of the current part\'s ports '
                'needs to be defined by `set_plot_shape()` ahead!'
            )
            assert found, err_str
            # get position of connected parts port and set that as start
            # position:
            start_pos = SimEnv.parts[conn_part].info_plot[conn_port][
                'coordinates'
            ]
        elif (
            type(start_pos) == str
            and start_pos in SimEnv.parts[part].port_names
        ):
            # if given as string, it is taken from the part connected to the
            # given port
            # get start position and info dict (ignore vector and length and
            # also pass dummy values to get direction for start pos)
            # get info dict key by constructing it before overwriting start pos
            d_key = 'auto_connection_' + start_pos
            start_pos, _, _, new_info = NetPlotter._get_drctn(
                ('port', start_pos), (0, 0), part
            )
            # reverse extract info from new info by reconstructing the port:
            info_plot['auto_connection'] = new_info[d_key]

        # assert linewidth:
        err_str = (
            'If `linewidth` is given, it must be given as a positive '
            'integer or float value!'
        )
        if linewidth is not None:
            # if linewidth is given, set it to info plot dict
            assert type(linewidth) == int or type(linewidth) == float, err_str
            assert linewidth >= 0, err_str
            info_plot['path_linewidth'] = linewidth
        else:
            # else set standard lw of 10
            info_plot['path_linewidth'] = 10

        # assert orientation:
        err_str = 'If `orientation` is given, it must be given as a string!'
        if orientation is not None:
            assert type(orientation) == str, err_str
            info_plot['orientation'] = orientation
        # assert invert (used for complex parts to:
        err_str = (
            'If `invert` is given, it must be given as a bool! '
            'Inverting is used for complex drawing parts like valves '
            'when the automatic positioning does not find the correct '
            'port positions.'
        )
        if invert is not None:
            assert type(invert) == bool, err_str
            info_plot['invert'] = invert

        #        # assert and get starting direction:
        #        err_str = ('`start_direction` must be given as a string and must be '
        #                   'either \'hor\' for a horizontal start or \'vert\' for a '
        #                   'vertical start.')
        #        assert type(start_direction) == str, err_str
        #        assert start_direction == 'hor' or start_direction == 'vert', err_str
        #        info_plot['start_direction'] = start_direction

        # assert and get vertex coordinates:
        err_str = (
            '`vertex_coords` must be given as a dict containing all '
            'vertices along which the part shall be plotted. The dict '
            'keys must be the vertex numbers starting from 1 for the '
            'first vertex (not including the starting point) and '
            'incrementing by 1 for each vertex. The coordinates of the '
            'vertices must be given as a tuple of length 2  with '
            'the first element giving the x-axis position and the '
            'second element giving the y-axis position of the vertex '
            'as an integer or float value. At least one vertex must be '
            'given for the end position of the part.\n'
            'The resulting dict must look like:\n'
            'vertex_coords = {1: (\'vec\', 2.4, 3),\n'
            '                 2: (\'pos\', 5, 3),\n'
            '                 3: (\'pos\', 7, 6),\n'
            '                 4: (\'port\', \'out\')}'
        )
        assert type(vertex_coords) == dict, err_str
        # assert that key step size is always 1:
        assert np.all(
            np.diff(np.asarray((list(vertex_coords.keys())))) == 1
        ), err_str
        # assert that first key is 1:
        assert list(vertex_coords.keys())[0] == 1, err_str

        # loop over dict to assert and get max number of vertices
        for key, value in vertex_coords.items():
            # check that keys are int and values are tuple of length 2 and
            # its values are float or int
            assert type(key) == int, err_str
            #            # removed since this will be done in get coords method
            #            assert type(value) == tuple and len(value) == 2, err_str
            #            assert ((type(value[0]) == int or type(value[0]) == float)
            #                    and (type(value[1]) == int
            #                         or type(value[1]) == float)), err_str
            # save max. key:
            max_key = key

        # save number of paths
        info_plot['number_paths'] = max_key
        # and save given vertex coords for later lookup:
        info_plot['vertex_coordinates'] = vertex_coords

        # now loop over dict again but this time with range (makes sure that
        # looping is in correct order!) and create start- and end-coordiantes
        # for each path as well as path vectors and lengths:
        # set variable for total path length to zero:
        total_length = 0
        # add path to dict:
        info_plot['path'] = dict()
        for i in range(1, max_key + 1):
            # calculate distance of path between old and new vertex:
            if i == 1:
                # for the first vertex
                # get starting position:
                path_start = start_pos

            else:
                # for all other vertices:
                # get starting position:
                #                path_start = vertex_coords[i - 1]  # replaced with new lookup
                path_start = info_plot['path'][i - 2]['end_coordinates']
            #            # get vector:
            #            vec = np.asarray(vertex_coords[i]) - np.asarray(path_start)
            #            # get length (vector magnitude)
            #            length = np.sqrt((vec * vec).sum())
            if vertex_coords[i][0] != 'port':
                # if port was not given for auto connect
                end_coords, vec, length = NetPlotter._get_drctn(
                    vertex_coords[i], path_start, part
                )
            else:
                # else port was given for auto connect, get also info dict:
                (end_coords, vec, length, new_info) = NetPlotter._get_drctn(
                    vertex_coords[i], path_start, part
                )
                info_plot.update(new_info)
            # update total length
            total_length += length
            # write to info dict:
            info_plot['path'][i - 1] = dict()
            info_plot['path'][i - 1] = {
                'start_coordinates': path_start,
                'end_coordinates': end_coords,
                #                                        'end_coordinates': vertex_coords[i],
                'vector': vec,
                'length': length,
            }

        # write total length to info dict:
        info_plot['total_path_length'] = total_length
        # get each paths proportion of the total length and calculate number
        # of cells which each path has:
        # check counter for total number of cells:
        total_num_cells = 0
        # total xy-vec for all paths (x value first cell in row, y in second):
        xy_vec_total = np.zeros((0, 2))
        # for all except last path set use endpoint to false:
        use_ep = False
        # loop over paths (again with range to make sure that order is ok) to
        # create all vectors etc. which are dependent on the total length:
        for i in range(0, max_key):
            # get path fraction:
            path_frac = info_plot['path'][i]['length'] / total_length
            # get number of cells (+1 since there needs to be one more
            # x-y-coords than cells to plot all cell values on lines, since
            # for example one cell needs one x-y-coords for the start and one
            # for the end point):
            num_cells = round(path_frac * SimEnv.parts[part].num_gp + 1)
            # add to total number of cells:
            total_num_cells += num_cells
            # check if length is reached at last iteration:
            if i == max_key - 1:
                # if last path is reached, set use endpoint to true:
                use_ep = True
                # and check for length:
                if total_num_cells != (SimEnv.parts[part].num_gp + 1):
                    # get difference:
                    dif = (SimEnv.parts[part].num_gp + 1) - total_num_cells
                    # add up to all relevant variables to make it work:
                    total_num_cells += dif
                    num_cells += dif
            # check that start and end coords are not the same:
            err_str = (
                'For part '
                + part
                + ' and vertice '
                + str(i + 1)
                + ' the start and end coordinates are the same!'
            )
            if np.all(
                info_plot['path'][i]['start_coordinates']
                == info_plot['path'][i]['end_coordinates']
            ):
                raise ValueError(err_str)
            # make x and y position vectors for cells along the path:
            x_vec = np.linspace(
                info_plot['path'][i]['start_coordinates'][0],
                info_plot['path'][i]['end_coordinates'][0],
                num_cells,
                endpoint=use_ep,
            )
            y_vec = np.linspace(
                info_plot['path'][i]['start_coordinates'][1],
                info_plot['path'][i]['end_coordinates'][1],
                num_cells,
                endpoint=use_ep,
            )
            # merge both vectors to two columns with each row having [x, y]
            # coordinates:
            xy_vec_path = np.zeros((x_vec.shape[0], 2))
            xy_vec_path[:, 0] = x_vec
            xy_vec_path[:, 1] = y_vec
            # add to total xy vector:
            xy_vec_total = np.concatenate((xy_vec_total, xy_vec_path), axis=0)
            # add all to info dict:
            info_plot['path'][i]['path_fraction'] = path_frac
            info_plot['path'][i]['num_cells'] = num_cells
            info_plot['path'][i]['path_xy_vector'] = xy_vec_path
            # now loop over the part's port indices and check if there are
            # ports along the current path. If yes, save each port's
            # coordinates to info_plot dict:
            # create a copy of port index to loop over it:
            if (
                'auto_connection' not in info_plot
                or SimEnv.parts[part].port_names.index(
                    info_plot['auto_connection']['own_port']
                )
                == 0
            ):
                # if the port which is at the start of the path is the first
                # port in the path OR NO auto connection method was chosen,
                # loop over port index in the standard order
                pi = SimEnv.parts[part]._port_own_idx.copy()
            else:
                # else the port which is at the start of the path is NOT the
                # first port in the path, loop over port index in reverse order
                pi = SimEnv.parts[part]._port_own_idx[::-1].copy()
            # create iterator object:
            itr = np.nditer(pi, flags=['c_index'])
            for p_idx in itr:
                # check if port is in current path:
                if (total_num_cells - num_cells) <= p_idx < total_num_cells:
                    # get coordinates of port depending on the index of the
                    # port:
                    # - ports at the start of a part (index 0) will take the
                    #   first xy-segment of a part's path
                    # - ports at the end of a part (index -1 or (num_gp - 1)
                    #   will take the last xy-segment of a part's path
                    # - ports inbetween will take the mean value of the segment
                    #   where they are located
                    if p_idx == 0:
                        port_coords = xy_vec_total[p_idx, :]
                    elif p_idx == (SimEnv.parts[part].num_gp - 1):
                        port_coords = xy_vec_total[p_idx + 1, :]
                    else:
                        port_coords = (
                            xy_vec_total[p_idx, :] + xy_vec_total[p_idx + 1, :]
                        ) / 2
                    # get name of port:
                    port_name = SimEnv.parts[part].port_names[itr.index]
                    # save to info_plot dict:
                    info_plot[port_name] = {'coordinates': port_coords}

        # assert that total number of cells matches number of gridpoints:
        err_str = (
            'The total number of cells is not equal to the number of '
            'gridpoints!'
        )
        assert total_num_cells == SimEnv.parts[part].num_gp + 1, err_str
        # save total xy vec to info dict:
        info_plot['xy_vector'] = xy_vec_total
        # also save x and y vectors for plotting:
        info_plot['x_vector'] = xy_vec_total[:, 0]
        info_plot['y_vector'] = xy_vec_total[:, 1]

        # save plotting dict to part:
        SimEnv.parts[part].info_plot = info_plot

        # set part to plot ready:
        SimEnv.parts[part].plot_ready = True

        # draw complex parts:
        if SimEnv.parts[part].plot_special:
            SimEnv.parts[part].draw_part('dummy', 0, draw=False)

    @staticmethod
    def _get_drctn(coords, start_coords, part):
        """
        Calculates the direction to the new target coordinates depending on the
        input type.

        - If a tuple of 'pos' and two integer or float values is given, this is
          taken as the target position. The values will be checked and the
          **target coordinates, vector to target and vector length** will be
          returned.
          Example: ('pos', 5, 7.3)
        - If a tuple of 'vec' and two integer or float values is given, this is
          taken as the vector to the target position. The values will be
          checked and the **target coordinates, vector to target and vector
          length** will be returned.
          Example: ('vec', 3.5, 9)
        - (If a tuple of 'len' and one integer or float value is given, this is
          taken as the path length. The direction will then be taken from the
          preceding path of the preceding part to calculate the new end
          position.
          Example: ('len', 7.0))
        - If a tuple of 'port' and one string is given, this string must be one
          of the own ports and the part connected to this port must be already
          defined by NetPlotter. The end position will then be taken from this
          connected part/port. The values will be checked and the **target
          coordinates, vector to target, vector length and a part of the info
          dict** will be returned.
          Example: ('port', 'out')
        """

        # assert that type is tuple:
        err_str = (
            'The directions to the next point must be given as a tuple '
            'containing a string giving the calculation procedure in '
            'the first element and the needed values in the second '
            '(and third) element. Recognized procedures are:\n'
            '- \'pos\' given with integer or float values at elements '
            'two and three for the direct x- and y-coordinates of the '
            'position\n'
            '- \'vec\' given with integer or float values at elements '
            'two and three being the x- and y-elements of a vector '
            'pointing to the new position\n'
            '- \'port\' given with a string value at element two '
            'depicting the own port which shall be linked to its '
            'connected port'
        )
        assert type(coords) == tuple, err_str
        # assert that first element is correct:
        assert coords[0] in ['pos', 'vec', 'port'], err_str

        # assert that start coords is correct:
        err_str = (
            '`start_coords` must be given as a tuple or numpy array of '
            'length 2 with both elements of type integer or float.'
        )
        assert (
            type(start_coords) == tuple or type(start_coords) == np.ndarray
        ) and len(start_coords) == 2, err_str

        # if position was given directly:
        if coords[0] == 'pos':
            # assert types
            assert (
                type(coords[1]) == int
                or type(coords[1]) == float
                and type(coords[2]) == int
                or type(coords[2]) == float
            ), err_str
            # get target coords as tuple:
            trgt_coords = coords[1:3]
            # get vector as array:
            vec = np.asarray(trgt_coords) - np.asarray(start_coords)
            # get length (vector magnitude)
            length = np.sqrt((vec * vec).sum())
            return trgt_coords, vec, length
        elif coords[0] == 'vec':
            # assert types
            assert (
                type(coords[1]) == int
                or type(coords[1]) == float
                and type(coords[2]) == int
                or type(coords[2]) == float
            ), err_str
            # get vector as array:
            vec = np.asarray(coords[1:3])
            # get target coords as tuple:
            trgt_coords = tuple(start_coords + vec)
            # get length (vector magnitude)
            length = np.sqrt((vec * vec).sum())
            return trgt_coords, vec, length
        elif coords[0] == 'port':
            # assert types
            assert type(coords[1]) == str, err_str
            err_str = (
                'The given port '
                + coords[1]
                + ' for part '
                + part
                + ' does not exist!'
            )
            assert coords[1] in SimEnv.parts[part].port_names, err_str
            # get connected part:
            conn_part = SimEnv.port_links[part + ';' + coords[1]]
            # conn part is part;port, split up:
            conn_part, conn_port = conn_part.split(';')
            # check if already defined for plotting:
            if (
                hasattr(SimEnv.parts[conn_part], 'info_plot')
                and SimEnv.parts[conn_part].plot_ready
            ):
                # construct dict key with own port name:
                d_key = 'auto_connection_' + coords[1]
                # save connection info where the position is coming from
                info_plot = dict()
                info_plot[d_key] = {
                    'connected_part': conn_part,
                    'connected_port': conn_port,
                    'own_port': coords[1],
                }
                # target coords as tuple:
                trgt_coords = SimEnv.parts[conn_part].info_plot[conn_port][
                    'coordinates'
                ]
                # get vector as array:
                vec = np.asarray(trgt_coords) - np.asarray(start_coords)
                # get length (vector magnitude)
                length = np.sqrt((vec * vec).sum())
                return trgt_coords, vec, length, info_plot
            else:
                raise ValueError(
                    'Part ' + conn_part + ' needs to be defined '
                    'for plotting before part ' + part + ' if it '
                    'is chosen as reference position!'
                )

    @staticmethod
    def add_sensor(*, part, cell, phys_quant='T', identifier=None, **kwargs):
        """
        Add a sensor to `part` at array index `cell` which displays the
        physical quantity `phys_quant` in the plot.
        """

        # assert that part is already plot ready:
        err_str = (
            'To add a sensor to the plot at part ' + part + ', the '
            'part must be defined by `set_plot_shape()` before!'
        )
        assert SimEnv.parts[part].plot_ready, err_str

        pt = dict()

        # assert that part exists:
        err_str = 'The given part ' + part + ' was not found!'
        assert part in SimEnv.parts, err_str
        pt['part'] = part

        # assert that cell exists:
        err_str = 'The given cell with index ' + str(
            cell
        ) + ' does not ' 'exist at ' + part + ' with physical quantity array shape ' + str(
            SimEnv.parts[part].T.shape
        )
        assert 0 <= cell < SimEnv.parts[part].T.shape[0], err_str
        pt['cell'] = cell

        # if animated, get this from kwargs dict:
        if 'animate' in kwargs:
            animate = kwargs['animate']
        else:
            # else set to false
            animate = False

        # get view to array cells
        if phys_quant == 'T':
            pt['pquant'] = SimEnv.parts[part].res[:, cell : cell + 1]
        elif phys_quant == 'dm':
            # get massflow
            if SimEnv.parts[part].dm_invariant:
                # get massflow if it is invariant in part
                pt['pquant'] = SimEnv.parts[part].res_dm[:, 0:1]
            else:
                # get massflow if it is variant in part
                pt['pquant'] = SimEnv.parts[part].res_dm[:, cell : cell + 1]
        else:
            raise TypeError('Physical quantity' + phys_quant + ' unknown!')
        # save to dict:
        pt['phys_quant'] = phys_quant

        # save identifier (even if it is None):
        pt['identifier'] = identifier

        # now get xy-plot-grid position of that sensor:
        pt['pos_end'] = SimEnv.parts[part].info_plot['xy_vector'][cell + 1]
        pt['pos_start'] = SimEnv.parts[part].info_plot['xy_vector'][cell]
        pt['pos_center'] = (pt['pos_start'] + pt['pos_end']) / 2
        # get direction vector:
        pt['vec_dir'] = pt['pos_end'] - pt['pos_start']
        # get rotation angle:
        pt['rot_angle'] = SimEnv._angle_to_x_axis(pt['vec_dir'])

        # construct drawing for sensor (line through cell center with circle
        # on top):
        # vector from cell center a tiny bit down and up and to circle center:
        pt['vec_bot'] = np.array([0, -0.1])
        pt['vec_top'] = np.array([0, 1])
        pt['vec_cc'] = np.array([0, 1.2])
        # rotate vectors:
        pt['vec_bot'] = SimEnv._rotate_vector(pt['vec_bot'], pt['rot_angle'])
        pt['vec_top'] = SimEnv._rotate_vector(pt['vec_top'], pt['rot_angle'])
        pt['vec_cc'] = SimEnv._rotate_vector(pt['vec_cc'], pt['rot_angle'])
        # construct circle center position:
        pt['pos_cc'] = pt['pos_center'] + pt['vec_cc']
        # construct circle around midpoint of start and pos:
        pt['circ'] = plt.Circle(
            tuple(pt['pos_cc']),
            radius=0.2,
            facecolor='None',
            edgecolor=[0, 0, 0],
            linewidth=1.5,
            zorder=5,
            animated=animate,
        )
        # construct x and y grid for lines:
        pt['x_grid'] = np.array(
            [
                pt['pos_center'][0] + pt['vec_bot'][0],
                pt['pos_center'][0] + pt['vec_top'][0],
            ]
        )
        pt['y_grid'] = np.array(
            [
                pt['pos_center'][1] + pt['vec_bot'][1],
                pt['pos_center'][1] + pt['vec_top'][1],
            ]
        )

        # make text displaying the sensors value:
        if phys_quant == 'T':
            if identifier is None:
                pt['txt_constr'] = '$' + phys_quant + r' = {0:6.2f}\,$°C'
            else:
                pt['txt_constr'] = '$' + pt['identifier'] + r' = {0:6.2f}\,$°C'
        elif phys_quant == 'dm':
            if identifier is None:
                pt['txt_constr'] = r'$\dot{{m}} = {0:6.3f}\,$kg/s'
            else:
                pt['txt_constr'] = (
                    '$' + pt['identifier'] + r' = {0:6.3f}\,$kg/s'
                )
        # get offset vector depending on rotation of sensor to deal with
        # none-quadratic form of textbox to avoid overlapping. only in the
        # range of +/-45° of pos. and neg. x-axis an offset vec length of
        # -20 is allowed, else -30:
        offset = (
            15
            if (
                0 <= pt['rot_angle'] <= 45 / 180 * np.pi
                or 135 / 180 * np.pi <= pt['rot_angle'] <= 225 / 180 * np.pi
                or pt['rot_angle'] >= 315 / 180 * np.pi
            )
            else 45
        )
        # get text offset from bottom point of pump by vector rotation:
        pt['txt_offset'] = tuple(
            SimEnv._rotate_vector(np.array([0, offset]), pt['rot_angle'])
        )

        # save to sensors dict:
        SimEnv.sensors.append(pt)

    @staticmethod
    def _draw_sensor(*, axis, step, animate):
        """
        This method is called by NetPlotter to draw the sensors.
        """

        # if animated, returns are needed. save them to:
        ani_ret = list()
        # loop over sensors:
        for i, sensor in enumerate(SimEnv.sensors):
            # try to draw sensor. if circle is already drawn somewhere else, a
            # RuntimeError will be caused. In this case reconstruct sensor and
            # draw again at the end:
            try:
                # add circle
                axis.add_patch(sensor['circ'])
            except RuntimeError:
                sens_bkp = sensor.copy()
                # clear sensor
                SimEnv.sensors[i] = []
                # add again with bkp values:
                NetPlotter.add_sensor(
                    part=sens_bkp['part'],
                    cell=sens_bkp['cell'],
                    phys_quant=sens_bkp['phys_quant'],
                    identifier=sens_bkp['identifier'],
                    animate=animate,
                )
            # add lines to plot
            axis.plot(
                sensor['x_grid'],
                sensor['y_grid'],
                color=[0, 0, 0],
                linewidth=1.5,
                zorder=5,
                animated=animate,
            )
            # generate text:
            txt = sensor['txt_constr'].format(sensor['pquant'][step, 0])
            # make annotation:
            sensor['ann'] = axis.annotate(
                txt,
                xy=(sensor['pos_cc']),
                xytext=sensor['txt_offset'],
                textcoords='offset points',
                ha='center',
                va='center',
                animated=animate,
            )
            # save animation returns to list in list
            ani_ret.append(
                [sensor['ann'], sensor['txt_constr'], sensor['pquant']]
            )

        # delete empty list elements:
        SimEnv.sensors = [lmnt for lmnt in SimEnv.sensors if lmnt != []]

        # if animated, return stuff:
        if animate:
            return ani_ret

    @staticmethod
    def plot_net(*, timestep, clim=None, extend_area=None, **kwargs):
        # assert that timestep is in range:
        err_str = (
            'The chosen `timestep` for NetPlotter.plot_net() is '
            'greater than the number of timesteps of the simulation! '
            'The simulation has ' + str(SimEnv.stepnum - 1) + ' steps.'
        )
        assert timestep < SimEnv.stepnum, err_str

        # set new clims for colorbar:
        if clim is not None:
            cmin = clim[0]
            cmax = clim[1]
        else:
            cmin = 0
            cmax = 100

        # extend drawing area (or reduce it) if needed for some parts:
        if extend_area is not None:
            err_str = (
                'Extend area has to be a tuple of to floats or '
                'integers giving the additional size of the plot.'
            )
            assert (
                type(extend_area) == tuple and len(extend_area) == 2
            ), err_str
            assert (
                type(extend_area[0]) == float or type(extend_area[0]) == int
            ), err_str
        else:
            # else set to standard parameters:
            extend_area = (3, 2)

        fig = plt.figure(**kwargs)
        ax = fig.gca()
        plt.box(False)

        # loop over all parts with plot special without drawing to get all the
        # special construction methods:
        for part in SimEnv.parts:
            if SimEnv.parts[part].plot_special:
                SimEnv.parts[part].draw_part(
                    ax, timestep, draw=False, animate=False
                )

        # draw sensors:
        NetPlotter._draw_sensor(axis=ax, step=timestep, animate=False)

        for part in SimEnv.parts:
            if (
                hasattr(SimEnv.parts[part], 'plot_ready')
                and SimEnv.parts[part].plot_ready
                and not SimEnv.parts[part].plot_special
            ):
                ip = SimEnv.parts[part].info_plot
                # get xy vector and reshape:
                xy_resh = ip['xy_vector'].reshape(-1, 1, 2)
                # make line segments:
                xy_seg = np.concatenate([xy_resh[:-1], xy_resh[1:]], axis=1)
                # make line collection
                lines = _LineCollection(
                    xy_seg,
                    array=SimEnv.parts[part].res[timestep, :],
                    cmap=plt.cm.plasma,
                    norm=plt.Normalize(cmin, cmax),
                    linewidths=ip['path_linewidth'],
                )
                # set edgecolor by using patheffects:
                lines.set_path_effects(
                    [
                        _pe.Stroke(
                            linewidth=ip['path_linewidth'] + 3,
                            foreground='black',
                        ),
                        _pe.Normal(),
                    ]
                )
                ax.add_collection(lines)
            elif (
                hasattr(SimEnv.parts[part], 'plot_ready')
                and SimEnv.parts[part].plot_ready
                and SimEnv.parts[part].plot_special
            ):
                # if part has a special drawing method, pass axis to its
                # method and get drawing:
                SimEnv.parts[part].draw_part(
                    ax, timestep, draw=True, animate=False
                )

        ax.autoscale()
        ax.set_aspect('equal')

        ax.set_xlim(
            ax.get_xlim()[0] - extend_area[0],
            ax.get_xlim()[1] + extend_area[0],
        )
        ax.set_ylim(
            ax.get_ylim()[0] - extend_area[1],
            ax.get_ylim()[1] + extend_area[1],
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # get axes locations to make colorbar in the correct size:
        divider = _mal(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        _ = plt.colorbar(
            lines, cax=cax, cmap=plt.cm.plasma, label='Temperatur in (°C)'
        )

        fig.show()

    @staticmethod
    def animate(
        *,
        t_start,
        t_stop,
        steps,
        clim,
        file=None,
        fps=25,
        dpi=300,
        extend_area=None,
        **kwargs
    ):
        """
        Makes an animation from the plotting function.
        """

        # assert that times are in range:
        err_str = (
            'The chosen `t_stop` in [s] for NetPlotter.plot_net() is '
            'greater than the timeframe of the simulation! The '
            'simulation timeframe is ' + str(SimEnv.timeframe) + ' steps.'
        )
        assert t_stop <= SimEnv.timeframe, err_str

        if clim is not None:
            cmin = clim[0]
            cmax = clim[1]
        else:
            cmin = 0
            cmax = 100

        # extend drawing area (or reduce it) if needed for some parts:
        if extend_area is not None:
            err_str = (
                'Extend area has to be a tuple of to floats or '
                'integers giving the additional size of the plot.'
            )
            assert (
                type(extend_area) == tuple and len(extend_area) == 2
            ), err_str
            assert (
                type(extend_area[0]) == float or type(extend_area[0]) == int
            ), err_str
        else:
            # else set to standard parameters:
            extend_area = (3, 2)

        # make figure:
        fig_ani = plt.figure(**kwargs)
        ax_a = fig_ani.gca()
        plt.box(False)

        # make annotation to show elapsed time:
        # start text:
        txt = 'elapsed time: ' + '$t = $' + r'0$\,$s'
        # text constructor for format function:
        txt_constr = 'elapsed time: ' + '$t = $' + r'{0:6.3f}$\,$s'
        ann_t = ax_a.annotate(
            txt,
            xy=(11, 6),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='center',
            animated=True,
        )
        # get array view to time:
        # therefore make a copy of it, expand its dims to allow correct
        # indexing in animation generation and get view to it:
        t_vec = np.expand_dims(SimEnv.time_vec.copy(), axis=1)
        t_view = t_vec[:]
        # add all to the text lists:
        txts = [ann_t]  # list for text annotation handles
        txts_constr = [txt_constr]  # list for text constructors
        txts_val_view = [t_view]  # list for views to values for texts

        # loop over all parts with plot special without drawing to get all the
        # special construction methods:
        for part in SimEnv.parts:
            if SimEnv.parts[part].plot_special:
                SimEnv.parts[part].draw_part(
                    ax_a, t_stop, draw=False, animate=True
                )

        line_ctr = 0  # counts the number of LineCollections
        xy_segs = list()  # list to save line segments in
        lws = list()  # list to save linewidths in
        peffs = list()  # list for patheffects
        data = list()  # list for temp data
        # loop over all parts to get their line segments, path effects, texts
        # etc. and save them to the lists above:
        for part in SimEnv.parts:
            if (
                hasattr(SimEnv.parts[part], 'plot_ready')
                and SimEnv.parts[part].plot_ready
                and not SimEnv.parts[part].plot_special
            ):
                line_ctr += 1
                ip = SimEnv.parts[part].info_plot
                # get xy vector and reshape:
                xy_resh = ip['xy_vector'].reshape(-1, 1, 2)
                # make line segments:
                xy_segs.append(
                    np.concatenate([xy_resh[:-1], xy_resh[1:]], axis=1)
                )
                # get linewidths and path effects:
                lws.append(ip['path_linewidth'])
                peffs.append(
                    [
                        _pe.Stroke(
                            linewidth=ip['path_linewidth'] + 3,
                            foreground='black',
                        ),
                        _pe.Normal(),
                    ]
                )
                data.append(SimEnv.parts[part].res)
            elif (
                hasattr(SimEnv.parts[part], 'plot_ready')
                and SimEnv.parts[part].plot_ready
                and SimEnv.parts[part].plot_special
            ):
                # if part has a special drawing method, pass axis to its
                # method, draw and get annotations from return:
                ann = SimEnv.parts[part].draw_part(
                    ax_a, 0, draw=True, animate=True
                )
                # extract from annotations: get txts (handle to annotations),
                # txts_constr (unformatted strings to pass to annotations which
                # have to be formatted with string.format()) and txts_val_view
                # (memory views to array cells to get format information from
                # as a vector over the timesteps):
                for lmnt in ann:
                    txts.append(lmnt[0])
                    txts_constr.append(lmnt[1])
                    txts_val_view.append(lmnt[2])

        # now draw the sensors and get their texts in a list of lists:
        sens_txt = NetPlotter._draw_sensor(axis=ax_a, step=0, animate=True)
        # loop over the texts and save them to the other lists:
        for lmnt in sens_txt:
            txts.append(lmnt[0])
            txts_constr.append(lmnt[1])
            txts_val_view.append(lmnt[2])

        # accessing max and min x and y values for axis scaling:
        xmax = -np.inf
        xmin = np.inf
        ymax = -np.inf
        ymin = np.inf
        for i, xy in enumerate(xy_segs):
            if xy[:, :, 0].max() > xmax:
                xmax = xy[:, :, 0].max()
            if xy[:, :, 0].min() < xmin:
                xmin = xy[:, :, 0].min()
            if xy[:, :, 1].max() > ymax:
                ymax = xy[:, :, 1].max()
            if xy[:, :, 1].min() < ymin:
                ymin = xy[:, :, 1].min()

        # make correct number of linecollection plots depending on line_ctr
        # and pass xy_segs and lws
        lines = [
            ax_a.add_collection(
                _LineCollection(
                    xy_segs[i],
                    array=[],
                    cmap=plt.cm.plasma,
                    norm=plt.Normalize(cmin, cmax),
                    linewidths=lws[i],
                    animated=True,
                )
            )
            for i in range(line_ctr)
        ]
        #        lines = [LineCollection(
        #                xy_segs[i], array=[],
        #                cmap=plt.cm.plasma,
        #                norm=plt.Normalize(vmin=0, vmax=100),
        #                linewidths=lws[i], animated=True)
        #            for i in range(line_ctr)]
        #        for l in lines:
        #            ax_a.add_collection(l)

        # add lines to ax:
        #        ax_a.add_collection(lines)

        ax_a.set_xlim(xmin - extend_area[0], xmax + extend_area[0])
        ax_a.set_ylim(ymin - extend_area[1], ymax + extend_area[1])
        ax_a.set_xticks([])
        ax_a.set_yticks([])
        ax_a.set_aspect('equal')

        # get axes locations to make colorbar in the correct size:
        divider = _mal(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        _ = plt.colorbar(
            lines[0], cax=cax, cmap=plt.cm.plasma, label='Temperatur in (°C)'
        )

        # define animation initialization function:
        def init():
            for i, lc in enumerate(lines):
                lc.set_array([])
                lc.set_path_effects(peffs[i])
            return lines, txts

        # define animation update function:
        def update(time_idx, data, txts_constr, txts_val_view):
            # put new data into the line plots:
            for i, lc in enumerate(lines):
                lc.set_array(data[i][time_idx, :])
            #                lc.set_norm(plt.Normalize(cmin, cmax))
            # update the texts:
            for i, txt in enumerate(txts):
                txt.set_text(
                    txts_constr[i].format(txts_val_view[i][time_idx, 0])
                )
            return lines, txts

        # define time_idx to animate over:
        time_idx = np.linspace(
            t_start / SimEnv.timeframe * steps,
            t_stop / SimEnv.timeframe * SimEnv.stepnum - 1,
            steps,
            dtype=np.int32,
        )

        #        raise ValueError

        ani = _FuncAnimation(
            fig_ani,
            update,
            init_func=init,
            frames=time_idx,
            fargs=(data, txts_constr, txts_val_view),
            interval=100,
            blit=False,
            repeat=False,
        )

        ax_a.relim()
        #        ax_a.autoscale_view()

        # get save information for animation:
        if file is None:
            file = 'animation.html'

        ani.save(file, fps=fps, dpi=dpi)

    #        plt.show()

    @staticmethod
    def plot_errors(simenv, rms_error=True, stability_violations=True):
        """
        Plot RMS errors and von Neumann stability for each part.

        Plots the accumulated RMS error for all parts as a bin plot and/or
        the times that each part has violated the von Neumann / L2 stability
        conditions.
        """
        returns = []  # axes handles to return as a list
        if rms_error:  # print RMS error
            # preallocate Series for errors:
            simenv.part_errors = pd.Series()
            # get errors for all numeric parts:
            for part in simenv.parts:
                if simenv.parts[part].solve_numeric:
                    # add error to DataFrame:
                    simenv.part_errors = pd.concat(
                        [
                            simenv.part_errors,
                            pd.Series(
                                simenv.parts[part]._trnc_err,
                                index=[simenv.parts[part].name],
                            ),
                        ]
                    )
            # make a bar plot with errors:
            fig_err = plt.figure()
            ax_err = fig_err.gca()
            ax_err.set_title('Accumulated root mean square error per part')
            simenv.part_errors.plot.bar(ax=ax_err)
            fig_err.tight_layout()
            helper = (
                'To reduce the RMS error:\n'
                + u' \u2022'
                + ' Decrease the cell size '
                r'$\Rightarrow$ higher accuracy'
                '\n'
                u' \u2022' + ' Reduce the energy flow'
            )
            ax_err.annotate(
                helper,
                (0.5, 1),
                (0, 0),
                xycoords='axes fraction',
                textcoords='offset points',
                va='top',
                ha='center',
                multialignment='left',
                bbox={'boxstyle': 'round', 'fc': 'w'},
            )
            fig_err.canvas.set_window_title(
                'Simulation: Accumulated RMS error'
            )
            returns.append(ax_err)

        if stability_violations:
            # preallocate Series for violations:
            simenv.part_stability_breaches = pd.Series()
            # get violoations for all numeric parts:
            for part in simenv.parts:
                if simenv.parts[part].solve_numeric:
                    # add error to DataFrame:
                    simenv.part_stability_breaches = pd.concat(
                        [
                            simenv.part_stability_breaches,
                            pd.Series(
                                simenv.parts[part]._stability_breaches,
                                index=[simenv.parts[part].name],
                            ),
                        ]
                    )
            # make a bar plot with breaches:
            fig_breach = plt.figure()
            ax_breach = fig_breach.gca()
            ax_breach.set_title('Von Neumann stability breaches per part')
            simenv.part_stability_breaches.plot.bar(ax=ax_breach)
            fig_breach.tight_layout()
            helper = (
                'To reduce stability breaches:\n'
                + u' \u2022'
                + ' Increase the cell size\n'
                u' \u2022' + ' Reduce the energy flow'
            )
            ax_breach.annotate(
                helper,
                (0.5, 1),
                (0, 0),
                xycoords='axes fraction',
                textcoords='offset points',
                va='top',
                ha='center',
                multialignment='left',
                bbox={'boxstyle': 'round', 'fc': 'w'},
            )
            fig_breach.canvas.set_window_title(
                'Simulation: Von Neumann stability breaches per part'
            )
            returns.append(ax_breach)
        return returns


# %% METERS
class Meters:
    def __init__(self, simenv, start_time):
        self._simenv = simenv  # save for attribute lookup

        self.start_time = pd.to_datetime(start_time)

        # get resampling specs:
        if self._simenv._disk_store['resample']:
            self._resample = True
            self._resample_freq = self._simenv._disk_store['resample_freq']
        else:
            self._resample = False
            self._resample_freq = None

        # get base simenv timevec:
        self._time_index = pd.to_datetime(
            self._simenv.time_vec, origin=self.start_time, unit='s'
        )
        # get meter timevec, depending on resampling or not
        if not self._resample:
            self._meter_time_index = self._time_index
        else:
            # etwas umstaendlich, aber einfach:
            self._meter_time_index = (
                pd.Series(index=self._time_index, data=np.nan)
                .resample(self._resample_freq)
                .mean()
                .interpolate()
                .index
            )

        self.meters = {}
        self.meters['heat_meter'] = {}
        self.meters['temperature'] = {}
        self.meters['massflow'] = {}
        self.meters['volumeflow'] = {}

        # make multiindex name specifier:
        self.__midx_names = ['type', 'name', 'sensor']
        mid = pd.MultiIndex(  # make empty multiindex
            levels=[[]] * 3, codes=[[]] * 3, names=['type', 'name', 'sensor']
        )
        # make empty dataframe
        self.meters_df = pd.DataFrame(
            columns=mid, index=self._meter_time_index
        )
        # also save meters and dataframe to SimEnv:
        self._simenv.meters = self.meters
        self._simenv.meters_df = self.meters_df
        # and check if disksaving is activated and if yes, save meters to disk
        self._disk_saving = self._simenv._SimEnv__save_to_disk
        if self._disk_saving:
            self._dstore = self._simenv._disk_store['store']

    def heat_meter(
        self,
        *,
        name,
        warm_part,
        warm_cell,
        cold_part,
        cold_cell,
        massflow_from_hot_part=False,
        full_output=False,
        evenly_spaced=False,
        freq='1s',
        how='interpolate'
    ):
        """
        Calculate heat meter values from cold to warm part.

        This classmethod calculates the heat flow between two parts and their
        selected cells. The first part defines the forward flow and should be
        the warm part/flow by definition, while the second part defines the
        cold return flow **and the massflow**.
        The following values are calculated:
            - heatflow in [kW]
            - flown heat Megajoule, cumulative in [MJ]
            - flown heat kWh, cumulative in [kWh]
            - massflow in [kg/s]
            - volume flow in [m^3/s]
            - flown mass in [kg]
            - flown volume in [m^3]
            - forward flow temperature in [°C]
            - return flow temperature in [°C]

        """
        err_str = (
            '`name=\'{0}\'` has already been assigned to '
            'another heat meter.'.format(name)
        )
        assert name not in self.meters['heat_meter'], err_str

        # check if parts and cell indices exist
        self._simenv._check_isinrange(
            part=warm_part, index=warm_cell, target_array='temperature'
        )
        self._simenv._check_isinrange(
            part=cold_part, index=cold_cell, target_array='temperature'
        )

        # get massflow depending on selected source:
        if massflow_from_hot_part:
            fidx_dm = self._simenv._massflow_idx_from_temp_idx(
                part=warm_part, index=warm_cell
            )
            mflow = self._simenv.parts[warm_part].res_dm[:, fidx_dm]
        else:  # massflow from cold part (default)
            sidx_dm = self._simenv._massflow_idx_from_temp_idx(
                part=cold_part, index=cold_cell
            )
            mflow = self._simenv.parts[cold_part].res_dm[:, sidx_dm]

        df = pd.DataFrame(index=self._time_index)  # make dataframe

        # forward and return flow temperature and difference
        df['T_ff'] = self._simenv.parts[warm_part].res[:, warm_cell]
        df['T_rf'] = self._simenv.parts[cold_part].res[:, cold_cell]
        df['T_diff'] = df['T_ff'] - df['T_rf']
        df['massflow_kgps'] = mflow.copy()  # copy to fix this value
        df['volume_flow_m3ps'] = df['massflow_kgps'] / _rho_water(
            df['T_rf'].values
        )
        # get massflow backup array for only positive values:
        massflow_bkp = mflow.copy()  # double copy not really required, but...
        df['flown_mass_kg'] = np.cumsum(
            massflow_bkp * self._simenv.time_step_vec
        )  # cumsum with neg. vals
        df['flown_volume_m3'] = (
            df['volume_flow_m3ps'] * self._simenv.time_step_vec
        ).cumsum()

        if full_output:  # these are not reqlly _always_ required...
            massflow_bkp[massflow_bkp < 0.0] = 0.0  # set negative to zeros
            df['flown_mass_pos_kg'] = np.cumsum(
                massflow_bkp * self._simenv.time_step_vec
            )  # only pos. cumsum
            df['flown_volume_pos_m3'] = np.cumsum(  # only pos. cumsum
                massflow_bkp
                * self._simenv.time_step_vec
                / _rho_water(df['T_rf'].values)
            )
        # get heatflow in [kW]
        df['heatflow_kW'] = (
            (df['T_ff'] - df['T_rf'])
            * df['massflow_kgps']
            / 1e3
            * (_cp_water(df['T_rf'].values) + _cp_water(df['T_ff'].values))
            / 2
        )
        hf_bkp = df['heatflow_kW'].copy()  # bkp heatflow for only-pos.-cumsum
        df['flown_heat_MJ'] = (
            np.cumsum(hf_bkp * self._simenv.time_step_vec) / 1e3
        )
        df['flown_heat_kWh'] = df['flown_heat_MJ'] / 3.6

        if full_output:  # these are not reqlly  _always_ required...
            hf_bkp[hf_bkp < 0.0] = 0.0  # set negative to zeros
            df['flown_heat_pos_MJ'] = (
                np.cumsum(hf_bkp * self._simenv.time_step_vec) / 1e3
            )
            df['flown_heat_pos_kWh'] = df['flown_heat_pos_MJ'] / 3.6

        # if even meter is requested, make it an even meter:
        if evenly_spaced:
            raise DeprecationWarning(
                'This is deprecated, since this may cause double resampling '
                'if simulation data is resampled (default), causing '
                'matplotlib to freeze. If additional resampling is required, '
                'please simply resample by hand. This method has done nothing '
                'else than simple resampling before...'
            )
            df = process_unevenly_spaced_timeseries(
                data=df, freq=freq, how=how
            )

        # resample if specified in simenv:
        df = self._resample_meter(df)

        # make multiindex for merging with dataframe for all sensors:
        tuples = []
        for col in df.columns:
            tuples.append(('heat_meter', name, col))
        midx = pd.MultiIndex.from_tuples(  # make multiindex
            tuples=tuples, names=self.__midx_names
        )
        self.meters_df[midx] = df  # add to dataframe

        # save to dict:
        self.meters['heat_meter'][name] = df
        if self._disk_saving:  # save to disk:
            self._save_to_disk(name, df)

    def temperature(self, *, name, part, cell):
        """
        Add a temperature sensor to meters.

        """
        # check if part and cell indices exist
        self._simenv._check_isinrange(
            part=part, index=cell, target_array='temperature'
        )

        err_str = (
            '`name={0}` has already been assigned to another '
            'temperature sensor.'.format(name)
        )
        assert name not in self.meters['temperature'], err_str

        df = pd.DataFrame(index=self._time_index)  # make dataframe
        # get temperature
        if self._simenv.parts[part].res.ndim == 2:
            df['T_' + name] = self._simenv.parts[part].res[:, cell]
        elif self._simenv.parts[part].res.ndim > 2:
            # if ndim of temp array > 1, res.ndim > 3. reshape to a flat
            # ndim=1 array PER timestep to index with flat index.
            df['T_' + name] = self._simenv.parts[part].res.reshape(
                -1, np.prod(self._simenv.parts[part].res.shape[1:])
            )[:, cell]

        # resample if specified in simenv:
        df = self._resample_meter(df)

        # make multiindex for merging with dataframe for all sensors:
        tuples = []
        for col in df.columns:
            tuples.append(('temperature', name, col))
        midx = pd.MultiIndex.from_tuples(  # make multiindex
            tuples=tuples, names=self.__midx_names
        )
        self.meters_df[midx] = df  # add to dataframe

        self.meters['temperature'][name] = df  # save to dict
        if self._disk_saving:  # save to disk:
            self._save_to_disk(name, df)

    def massflow(self, *, name, part, cell):
        """
        Add a massflow sensor in [kg/s] to meters.

        """
        # check if part and cell indices exist
        self._simenv._check_isinrange(
            part=part, index=cell, target_array='temperature'
        )
        # get index to massflow array
        idx_dm = self._simenv._massflow_idx_from_temp_idx(
            part=part, index=cell
        )

        err_str = (
            '`name=' + str(name) + '` has already been assigned to another '
            'massflow sensor.'
        )
        assert name not in self.meters['massflow'], err_str

        df = pd.DataFrame(index=self._time_index)  # make dataframe
        # get massflow
        df[name] = self._simenv.parts[part].res_dm[:, idx_dm]

        # resample if specified in simenv:
        df = self._resample_meter(df)

        # make multiindex for merging with dataframe for all sensors:
        tuples = []
        for col in df.columns:
            tuples.append(('massflow', name, col))
        midx = pd.MultiIndex.from_tuples(  # make multiindex
            tuples=tuples, names=self.__midx_names
        )
        self.meters_df[midx] = df  # add to dataframe

        self.meters['massflow'][name] = df  # save to dict
        if self._disk_saving:  # save to disk:
            self._save_to_disk(name, df)

    def volumeflow(self, *, name, part, cell):
        """
        Add a volumeflow sensor in [m^3/s] to meters.

        """
        # check if part and cell indices exist
        self._simenv._check_isinrange(
            part=part, index=cell, target_array='temperature'
        )
        # get index to massflow array
        idx_dm = self._simenv._massflow_idx_from_temp_idx(
            part=part, index=cell
        )

        err_str = (
            '`name=' + str(name) + '` has already been assigned to another '
            'massflow sensor.'
        )
        assert name not in self.meters['massflow'], err_str

        df = pd.DataFrame(index=self._time_index)  # make dataframe
        # get volumeflow
        df[name] = self._simenv.parts[part].res_dm[:, idx_dm] / _rho_water(
            self._simenv.parts[part].res[:, cell]
        )

        # resample if specified in simenv:
        df = self._resample_meter(df)

        # make multiindex for merging with dataframe for all sensors:
        tuples = []
        for col in df.columns:
            tuples.append(('volumeflow', name, col))
        midx = pd.MultiIndex.from_tuples(  # make multiindex
            tuples=tuples, names=self.__midx_names
        )
        self.meters_df[midx] = df  # add to dataframe

        self.meters['volumeflow'][name] = df  # save to dict
        if self._disk_saving:  # save to disk:
            self._save_to_disk(name, df)

    def get_sensors_df(self):
        return self.meters_df

    def get_sensors_dict(self):
        return self.meters

    def _resample_meter(self, df):
        """Resample meters if specified in simenv."""
        if self._resample:
            return df.resample(self._resample_freq).mean().interpolate()
        else:
            return df

    def _save_to_disk(self, name, meter):
        if self._save_to_disk:
            try:  # try to open in append/write mode
                self._dstore.open(mode='a')
            except ValueError:  # if alread open in read mode, close first
                self._dstore.close()
                self._dstore.open(mode='a')
            self._dstore.put(
                'meters/' + name,
                meter,
                format='table',
                complevel=self._simenv._disk_store['complevel'],
                complib=self._simenv._disk_store['complib'],
            )
            self._dstore.close()

    def pred_real_plot(
        self,
        show_rmse=True,
        show_cvrmse=True,
        hexbin=False,
        exclude_zeros=True,
        resample_uneven=True,
    ):
        """
        Plot a prediction-realization diagram for the specified time series.
        """


#        sources: ashrae und schittgabler zeitreihenanalyse für R oder so...
#        move outside of meters to be able to use it on all kind of TS?
#        check if TS is even AND if freq is the same as measurement-TS-freq


# %% and again some basic statistics. the order here is just baaad
def is_evenly_spaced(data, return_freq=False):
    """
    This function checks if a pandas Series or DataFrame with DatetimeIndex is
    evenly spaced. Returns True if the time series is evenly spaced, else
    False.
    If `return_freq=True` is set, the frequency of the time series will be
    returned as the second argument.
    """
    assert isinstance(
        data.index, pd.DatetimeIndex
    ), 'The index must be a DatetimeIndex.'
    if not return_freq:
        return data.index.freq is not None
    else:
        return (data.index.freq is not None, data.index.freq)


def fit_time_series(
    *, pred, rlz, resample_uneven=True, freq=None, how=None, return_df=False
):
    """
    Checks for matching of type and shape of the time series `pred` and `rlz`.
    If the time series do not match and the index is a DatetimeIndex,
    `fit_time_series` tries to fit the time series by resampling if
    `resample_uneven=True` is set.
    If `return_df=True` is set, the time series will be returned in a single
    DataFrame. Otherwise a tuple of two time series will be returned.

    Returns:
    --------
    pred : pd.Series
        Fitted prediction time series, if `return_df=False`.
    rlz : pd.Series
        Fitted realization time series, if `return_df=False`.
    """

    err_type = (  # type error
        'The prediction (sim. data) `pred` and the realization (measurement '
        'data) `rlz` have to be given as a pandas Series.'
    )
    assert isinstance(pred, pd.Series) and isinstance(rlz, pd.Series), err_type
    err_idx = (  # index error
        'The index of the pandas Series has to be of the same type and shape '
        'for `pred` and `rlz`. The following index types are supported:\n'
        '    - DatetimeIndex: The frequency of both indices must be exactly '
        'equal. Starting time and/or offset is not being checked for. If at '
        'least one of the time series is unevenly spaced, it has to be '
        'resampled by setting `resample_uneven=True`.\n'
        '    - Integer, Float or Range indices: The indices will be ignored '
        'and both time series will be analyzed according the the order of the '
        'array elements.'
    )
    err_shape = (  # shape error
        'The shape of the prediction time series ' + str(pred.shape) + ' '
        'does not match the shape of the realization time series '
        + str(rlz.shape)
        + '. This may either result from passing '
        'differently shaped evenly spaced time series OR from resampling '
        'unevenly spaced time series.'
    )
    if isinstance(pred.index, pd.DatetimeIndex):  # if DatetimeIndex
        assert isinstance(rlz.index, pd.DatetimeIndex), err_idx  # same idx
        # get all frequencies and related bools:
        target_freq = pd.tseries.frequencies.to_offset(freq)
        pred_evenly, pred_freq = is_evenly_spaced(pred, return_freq=True)
        rlz_evenly, rlz_freq = is_evenly_spaced(rlz, return_freq=True)
        # if target freq is not set, take rlz freq as target freq
        target_freq = target_freq if target_freq is not None else rlz_freq
        # if not evenly spaced or frequency is not the target frequency:
        if (
            not pred_evenly
            or not rlz_evenly
            or pred_freq != target_freq
            or rlz_freq != target_freq
        ):
            assert resample_uneven, err_idx  # assert that resampling is set
            err_uneven = (
                'If `resample_uneven=True` is set and at least one time '
                'series is unevenly spaced, the resampling frequency and '
                'method have to be given with `freq` and `how`. Pass any '
                'dummy value to receive more information about accepted '
                'values.'
            )
            # check if pred is unevenly spaced or same freq as target freq:
            if not pred_evenly or pred_freq != target_freq:
                assert freq is not None and how is not None, err_uneven
                pred = process_unevenly_spaced_timeseries(  # make evenly spcd
                    data=pred, freq=freq, how=how
                )
            #            if not is_evenly_spaced(rlz):  # check if rlz is unevenly spaced
            # check if rlz is unevenly spaced or same freq as target freq:
            if not rlz_evenly or rlz_freq != target_freq:
                assert freq is not None and how is not None, err_uneven
                rlz = process_unevenly_spaced_timeseries(  # make evenly spaced
                    data=rlz, freq=freq, how=how
                )
        # else do nothing and just assert that shapes match
        assert pred.shape == rlz.shape, err_shape  # assert that shapes match
    else:  # if other indices
        assert pred.shape == rlz.shape, err_shape  # assert that shapes match

    # return results:
    if not return_df:
        return pred, rlz
    else:
        return pd.concat((pred, rlz), axis=1)


def ts_performance_indicators(
    *,
    pred,
    rlz,
    exclude_zeros=True,
    exclude_values=None,
    exclude_range=None,
    resample_uneven=True,
    freq=None,
    how=None
):

    pred_rlz = fit_time_series(  # fit the timeseries, in necessary
        pred=pred,
        rlz=rlz,
        resample_uneven=resample_uneven,
        freq=freq,
        how=how,
        return_df=True,
    )

    err_excl = (
        'To exclude specific values/ranges from performance indicator '
        'calculation, the following options can be set:\n'
        '    - exclude_zeors: True or False. If True, zeros, which exist in '
        'both time series at the same location, will be excluded.\n'
        '    - exclude_values: list of values to exclude.\n'
        '    - exclude_range: Range of values to exclude, given as a list '
        'with 2 elements. The first element is the lower border, the second '
        'element is the upper border. The borders are also excluded.'
    )
    assert isinstance(exclude_zeros, bool), err_excl
    if exclude_zeros:  # exclude zeros which are in BOTH ts from perf. analysis
        # exclude zeros where both are zero
        pred_rlz = pred_rlz.loc[(pred_rlz != 0).all(axis=1)]
    if exclude_values is not None:  # exclude values which are in
        assert isinstance(exclude_values, list) and isinstance(
            exclude_values[0], (int, float)
        ), err_excl
        print('exclude only if in both or also if only in one?')
        # loop over values and create new masks
        raise NotImplementedError
    if exclude_range is not None:  # exclude values which are in
        assert isinstance(exclude_range, list) and isinstance(
            exclude_range[0], (int, float)
        ), err_excl
        print('exclude only if in both or also if only in one?')
        # construct mask like: mask = (0.5 < dfx) & (dfx < 0.8)
        raise NotImplementedError

    # calculate:
    # residuals or absolute error:
    residuals = pred_rlz.iloc[:, 0] - pred_rlz.iloc[:, 1]
    # res_desc = residuals.describe()  # get all general residual values
    # get all general values of the realization data
    rlz_desc = pred_rlz.iloc[:, 1].describe()
    n = residuals.size  # number of observations
    ss_res = (residuals ** 2).sum()  # sum of squares of residuals
    # total sum of squares (definition: sum((y_i - mean(y))**2))
    ss_total = (
        rlz_desc['std'] ** 2 * rlz_desc['count']
    )  # faster than definition
    mean_bias_err = residuals.sum() / n  # mean absolute biased error (MBE)
    mean_abs_err = residuals.abs().sum() / n  # mean absolute error (MAE)
    nmbe = mean_bias_err / rlz_desc['mean']  # normalized mean bias error NMBE
    rmse = np.sqrt(ss_res / n)  # root mean square error
    cv_rmse = rmse / rlz_desc['mean']  # coeff. of variation of rmse (CV(RMSE))
    nrmse = rmse / (rlz_desc['max'] - rlz_desc['min'])  # normalized RMSE
    nrmse_iqr = rmse / (  # norm over interquartile range
        abs(rlz_desc['75%'] - rlz_desc['25%'])
    )
    r_squared = 1 - ss_res / ss_total  # coefficient of determination

    # Summarize in Series
    perf_indicators = pd.Series(
        data=[
            n,
            mean_bias_err,
            mean_abs_err,
            nmbe,
            rmse ** 2,
            rmse,
            cv_rmse,
            nrmse,
            nrmse_iqr,
            r_squared,
        ],
        index=[
            'n_samples',
            'MBE',
            'MAE',
            'NMBE',
            'MSE',
            'RMSE',
            'CV(RMSE)',
            'NRMSE',
            'NRMSE_IQR',
            'R2',
        ],
    )

    return perf_indicators


# %% plotting
def plt_prediction_realization(
    *,
    pred,
    rlz,
    ax=None,
    plot_type='scatter',
    plot_every=1,
    show_linregress=True,
    exclude_zeros=True,
    resample_uneven=True,
    freq=None,
    how=None,
    freq_perf=None,
    errors=dict(
        R2=True,
        MSE=True,
        CVRMSE=True,
        NRMSE=False,
        NRMSE_IQR=False,
        NMBE=True,
        MBE=False,
        MAE=False,
    ),
    aspect='equal',
    scttr_kwds=dict(c='C0', s=6, fc='none', ec='k', alpha=0.5),
    kde_kwds={
        'contour': True,
        'vmin': 1e-2,
        'cont_lines': False,
        'norm': True,
    },
    diag_kwds=dict(color='k', ls='-', lw=1),
    err_kwds=dict(
        bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.5, pad=0.2)
    ),
    err_loc='bottom right',
    legend_kwds=dict(),
    auto_label=True,
    sprache='eng'
):
    """
    Make a prediction realization plot.

    Parameters
    ----------
    * : TYPE
        DESCRIPTION.
    pred : TYPE
        DESCRIPTION.
    rlz : TYPE
        DESCRIPTION.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    plot_type : str, optional
        Plot type to use for plotting the distributions. Supported plot types
        are 'scatter' (suitable for small numbers of samples), 'hexbin'
        (suitable for single plots) and 'kde' (slow, but suitable for all
        types). The default is 'scatter'.
    plot_every : TYPE, optional
        DESCRIPTION. The default is 1.
    show_linregress : TYPE, optional
        DESCRIPTION. The default is True.

    exclude_zeros : bool, optional
        Exclude data where **both** are zero. The default is True.
    resample_uneven : TYPE, optional
        DESCRIPTION. The default is True.
    freq : TYPE, optional
        DESCRIPTION. The default is None.
    how : TYPE, optional
        DESCRIPTION. The default is None.
    freq_perf : TYPE, optional
        DESCRIPTION. The default is None.
    errors : TYPE, optional
        DESCRIPTION. The default is dict(            R2=True, MSE=True, CVRMSE=True, NRMSE=False, NRMSE_IQR=False,            NMBE=True, MBE=False, MAE=False        ).
    aspect : TYPE, optional
        DESCRIPTION. The default is 'equal'.
    scttr_kwds : TYPE, optional
        DESCRIPTION. The default is dict(c='C0', s=6, fc='none', ec='k', alpha=.5).
    kde_kwds : TYPE, optional
        DESCRIPTION. The default is {'contour': True, 'vmin': 1e-2, 'cont_lines': False,                  'norm': True}.
    diag_kwds : TYPE, optional
        DESCRIPTION. The default is dict(color='k', ls='-', lw=1).
    err_kwds : TYPE, optional
        DESCRIPTION. The default is dict(bbox=dict(            boxstyle="round", fc="w", ec="k", alpha=.5, pad=0.2)).
    err_loc : TYPE, optional
        DESCRIPTION. The default is 'bottom right'.
    legend_kwds : TYPE, optional
        DESCRIPTION. The default is dict().
    sprache : TYPE, optional
        DESCRIPTION. The default is 'eng'.

    Returns
    -------
    ax_pr : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """

    assert plot_type in ('scatter', 'hexbin', 'kde')

    # fit time series if necessary and concat to df:
    df = fit_time_series(
        pred=pred,
        rlz=rlz,
        resample_uneven=resample_uneven,
        freq=freq,
        how=how,
        return_df=True,
    )
    # Exclude samples where BOTH are zeros to avoid accumulating too much data
    # at zero (this may reduce the overall performance indicators, thus yields
    # an estimate which is worse than the true estimate)
    if exclude_zeros:
        df = df[(df != 0).all(axis=1)]

    # get data min and max values:
    data_min = min(df.min())
    data_max = max(df.max())

    # if separate freq for perf given, else take general freq
    freq_perf = freq if freq_perf is None else freq_perf

    # calculate performance indicators:
    perf_ind = ts_performance_indicators(
        pred=df.iloc[:, 0],
        rlz=df.iloc[:, 1],  # columns pred and rlz in df
        exclude_zeros=False,  # already excluded, if True, a few lines before
        exclude_values=None,
        exclude_range=None,
        resample_uneven=resample_uneven,
        freq=freq_perf,
        how=how,
    )

    # make prediction realization plot:
    if ax is None:
        fig_pr, ax_pr = plt.subplots()
    else:
        ax_pr = ax
        # fig_pr = ax.get_figure()
    indicators_txt = ''  # empty string for indicator texts to add to plot
    line_colors = 'black'  # line colors for bisecting and rmse lines

    # translate to errors tuple:
    _errs = tuple(
        k.replace('CVRMSE', 'CV(RMSE)') for k, v in errors.items() if v
    )

    # Plot the data, either as a hexbin, 2d-kde or scatter plot.
    if plot_type == 'hexbin':
        colnames = df.columns  # get column names
        handles = df.plot.hexbin(
            x=colnames[1],
            y=colnames[0],
            ax=ax_pr,
            cmap='Blues',  # cmap='viridis',
            extent=[data_min, data_max, data_min, data_max],
        )
        line_colors = 'C1'  # set line colors to contrast colormap
    elif plot_type == 'kde':
        handles = plot_2d_kde(
            x=df.iloc[:, 1],
            y=df.iloc[:, 0],
            ax=ax_pr,
            cmap='Blues',
            line_color='k',
            errors=_errs,
            **kde_kwds
        )
    else:  # scatter plot
        handles = plot_pr_scatter(
            df.iloc[:, 1],
            df.iloc[:, 0],
            errors=_errs,
            plot_every=plot_every,
            scttr_kwds=scttr_kwds,
            diag_kwds=diag_kwds,
            err_kwds=err_kwds,
            err_loc=err_loc,
            legend_kwds=legend_kwds,
            auto_label=auto_label,
            sprache=sprache,
            err_vals={k: perf_ind.loc[k] for k in _errs},
            ax=ax_pr,
        )

    # only for hexbin, since the rest is given by plot_pr_scatter:
    if plot_type == 'hexbin':
        # set axis labels:
        ax_pr.set_xlabel('Measurement data')
        ax_pr.set_ylabel('Simulation data')
        # set axis limits:
        ax_pr.set_xlim(data_min, data_max)
        ax_pr.set_ylim(data_min, data_max)

        # plot intersection line:
        ax_pr.plot(
            [data_min, data_max],
            [data_min, data_max],
            linewidth=1,
            color=line_colors,
            label='Bisecting line',
        )

        # Add error indicators
        if errors.get('MBE', False):  # show_mean_bias_e:
            indicators_txt = (indicators_txt + 'MBE = {:.1f}' + '\n').format(
                perf_ind['mean_bias_err']
            )
        if errors.get('MAE', False):
            indicators_txt = (indicators_txt + 'MAE = {:.1f}' + '\n').format(
                perf_ind['mean_abs_err']
            )
        if errors.get('NMBE', False):
            indicators_txt = (indicators_txt + 'NMBE = {:.3f}' + '\n').format(
                perf_ind['nmbe']
            )
        # plot rmse confidence interval and print value
        if errors.get('RMSE', False):
            ax_pr.plot(  # plot upper rmse confidence interval
                [data_min, data_max],
                [data_min + perf_ind['rmse'], data_max + perf_ind['rmse']],
                linestyle='--',
                linewidth=1,
                color=line_colors,
                label='RMSE',
            )
            ax_pr.plot(  # plot lower rmse confidence interval
                [data_min, data_max],
                [data_min - perf_ind['rmse'], data_max - perf_ind['rmse']],
                linestyle='--',
                linewidth=1,
                color=line_colors,
            )
            # print coefficient of variation of RMSE
            indicators_txt = (indicators_txt + 'RMSE = {:.1f}' + '\n').format(
                perf_ind['rmse']
            )
        if (
            show_linregress
        ):  # plot linear regression to show constant deviation
            slope, intercept, _, _, _ = stats.linregress(
                df.iloc[:, 1], df.iloc[:, 0]
            )
            ax_pr.plot(
                [data_min, data_max],
                [data_min + intercept, intercept + slope * data_max],
                label='lin. regression',
                linewidth=1,
                color='C3',
            )
        if errors.get('CVRMSE', False):  # coefficient of variation of RMSE
            indicators_txt = (
                indicators_txt + 'CV(RMSE) = {:.3f}' + '\n'
            ).format(perf_ind['cv_rmse'])
        if errors.get('NRMSE', False):  # show mean normalized RMSE
            indicators_txt = (indicators_txt + 'NRMSE = {:.3f}' + '\n').format(
                perf_ind['nrmse']
            )
        if errors.get('NRMSE_IQR', False):  # show mean normalized RMSE
            indicators_txt = (
                indicators_txt + 'NRMSE IQR = {:.3f}' + '\n'
            ).format(perf_ind['nrmse_iqr'])
        if errors.get('R2', False):  # show mean normalized RMSE
            indicators_txt = (indicators_txt + 'R$^2$ = {:.3f}' + '\n').format(
                perf_ind['r_squared']
            )

        # Show levend
        ax_pr.legend(loc=2)

    # Set subplots aspect ratio
    ax_pr.set_aspect(aspect)

    if indicators_txt != '':  # if indicators were added
        if indicators_txt[-1:] == '\n':  # if linebreak is at the end
            indicators_txt = indicators_txt[:-1]  # drop last linebreak
        indicators_txt = 'Indicators:\n' + indicators_txt  # add description
        ax_pr.annotate(  # plt annotation with selected indicators
            indicators_txt,
            xy=(0.95, 0.05),
            xycoords='axes fraction',
            horizontalalignment='right',
            verticalalignment='bottom',
            bbox=dict(
                boxstyle='round', facecolor='white', ec='0.5', alpha=0.8
            ),
        )

    return handles  # return handles


@override(tb, 'tp')
def plot_pr_scatter(
    y,
    y_hat,
    ax=None,
    aspect='equal',
    ax_scaling_tight=False,
    errors=('R2', 'MSE', 'CV(RMSE)', 'NMBE'),
    scttr_kwds=dict(c='C0', s=6, fc='none', ec='C0', alpha=1.0),
    diag_kwds=dict(color='k', ls='-', lw=1),
    plt_err_range='RMSE',
    err_rng_kwds=dict(color='k', ls='--', lw=1),
    err_kwds=dict(
        bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.5, pad=0.2)
    ),
    err_loc='bottom right',
    legend_kwds=dict(),
    auto_label=True,
    sprache='eng',
    plot_every=1,
    fig_kwds=dict(figsize=(8 / 2.54, 8 / 2.54)),
    err_vals=None,
):
    """
    Make a prediction-realization scatter plot.

    Copied from toolbox to have a build independent of the toolbox package.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwds)
    else:
        fig = ax.get_figure()

    # Get min, max and min-max range
    pr_min, pr_max = (
        np.min([y.min(), y_hat.min()]),
        np.max([y.max(), y_hat.max()]),
    )
    minmax_range = pr_max - pr_min

    assert sprache in ('eng', 'de')
    if sprache == 'eng':
        datap = 'Data points'
        bsline = 'Bisecting line'
        xlabel = 'Measurement'
        ylabel = 'Prediction'
    elif sprache == 'de':
        datap = 'Datenpunkte'
        bsline = '$y = x$'
        xlabel = 'Messwerte'
        ylabel = 'Vorhersage'

    # Make the scatter plot
    ax.scatter(y[::plot_every], y_hat[::plot_every], label=datap, **scttr_kwds)
    # Plot the diagonal over a slightly extended min-max range
    ax.plot(
        [pr_min - 0.05 * minmax_range, pr_max + 0.05 * minmax_range],
        [pr_min - 0.05 * minmax_range, pr_max + 0.05 * minmax_range],
        label=bsline,
        **diag_kwds
    )

    # Annotate the errors
    annotate_errors(
        ax=ax,
        y=y,
        y_hat=y_hat,
        errors=errors,
        err_loc=err_loc,
        err_vals=err_vals,
        err_kwds=err_kwds,
    )
    # Set the plot aspect ratio
    ax.set_aspect(aspect)
    # Scale axis limits
    if ax_scaling_tight:
        ax.autoscale(tight=True)
    else:
        lims = (pr_min - 0.05 * minmax_range, pr_max + 0.05 * minmax_range)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_ylim(ax.get_xlim())

    # axis labels as SI conform labels, legend, grid and finally tight layout
    if auto_label:
        si_axlabel(ax, label=ylabel)
        si_axlabel(ax, label=xlabel, which='x')
    ax.grid(True, which='both')
    ax.legend(**legend_kwds)
    # fig.tight_layout(pad=0)

    return fig, ax


@override(tb, 'tp')
def plot_2d_kde(
    x,
    y,
    steps=100,
    ax=None,
    contour=True,
    cmap='Blues',
    norm=True,
    vmin=None,
    vmax=None,
    cont_lines=True,
    cbar=True,
    line_color='k',
    fontsize=8,
    aspect='equal',
    extend='both',
    cm_under='w',
    cm_over='k',
    errors=('R2', 'CV(RMSE)', 'NMBE'),
    plt_kwds={},
    **err_kwds
):
    """
    Make a prediction-realization plot with a 2D-KDE.

    Copied from toolbox to have a build independent of the toolbox package.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    assert isinstance(steps, (int, complex))
    steps = 100 * 1j if isinstance(steps, int) else steps
    m1, m2 = x.copy(), y.copy()
    xmin, xmax, ymin, ymax = m1.min(), m1.max(), m2.min(), m2.max()
    X, Y = np.mgrid[xmin:xmax:steps, ymin:ymax:steps]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    if norm:
        vmin_, vmax_ = Z.min(), Z.max()
        # overwrite norm values if given explicitly
        vmin = vmin if vmin is not None else vmin_
        vmax = vmax if vmax is not None else vmax_
    if not contour:
        im = ax.imshow(
            np.rot90(Z),
            cmap=cmap,
            **plt_kwds,
            extent=[xmin, xmax, ymin, ymax],
            extend=extend,
            vmin=vmin,
            vmax=vmax
        )
    else:
        im = ax.contourf(
            X, Y, Z, cmap=cmap, extend=extend, vmin=vmin, vmax=vmax
        )
        if cont_lines:
            clines = ax.contour(
                X,
                Y,
                Z,
                **plt_kwds,
                extend=extend,
                vmin=vmin,
                vmax=vmax,
                colors=line_color
            )
            ax.clabel(clines, inline=1, fontsize=fontsize)
    # extend colors if set
    if extend in ('both', 'max'):
        im.cmap.set_over(cm_over)
    if extend in ('both', 'min'):
        im.cmap.set_under(cm_under)
    # Plot colorbar
    if cbar:
        cax_div = _mal(ax).append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(mappable=im, cax=cax_div)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect(aspect)
    ax.grid(True)

    annotate_errors(ax=ax, y=x, y_hat=y, errors=errors, err_kwds=err_kwds)

    return {'fig': fig, 'ax': ax, 'mappable': im, 'cbar': cbar}


@override(tb, 'tp')
def annotate_errors(
    ax,
    y,
    y_hat,
    errors,
    err_loc='bottom right',
    err_vals=None,
    fontsize=8,
    err_kwds=dict(
        bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.5, pad=0.2)
    ),
):
    """
    Annotate a validation plot with errors measures.

    Copied from toolbox to have a build independent of the toolbox package.
    """
    annot_str = ''
    if 'R2' in errors:
        if err_vals is None or 'R2' not in err_vals:
            r2 = r_squared(y, y_hat)
        else:
            r2 = err_vals['R2']
        annot_str += r'$R^2={0:.3f}$'.format(r2) + '\n'
    if 'MSE' in errors:
        if err_vals is None or 'MSE' not in err_vals:
            mse = mean_squared_err(y, y_hat)
        else:
            mse = err_vals['MSE']
        annot_str += r'$MSE={0:.3G}$'.format(mse) + '\n'
    if 'CV(RMSE)' in errors:
        if err_vals is None or 'CV(RMSE)' not in err_vals:
            cvrmse = cv_rmse(y, y_hat)
        else:
            cvrmse = err_vals['CV(RMSE)']
        annot_str += r'$CV(RMSE)={0:.3f}$'.format(cvrmse) + '\n'
    if 'NMBE' in errors:
        if err_vals is None or 'NMBE' not in err_vals:
            nmbe = normalized_err(y, y_hat, err_method='MSD', norm='mean')
        else:
            nmbe = err_vals['NMBE']
        annot_str += r'$NMBE={0:.3f}$'.format(nmbe)

    if annot_str[-1:] == '\n':
        annot_str = annot_str[:-1]

    annotate_axes(
        ax, annotations=[annot_str], loc=err_loc, fontsize=fontsize, **err_kwds
    )


@override(tb, 'tp')
def si_axlabel(
    ax,
    label,
    unit=None,
    which='y',
    style='SI',
    spacing=r'\;',
    invert_position=False,
):
    r"""
    Generate axlabels which conform to several naming conventions.

    Generates a SI- and DIN-conform label for the axis, for example
    `label='Temperatur'` and unit='°C' will produce the string
    **`r'$Temperature\;/\;\mathrm{°C}$'`**.

    Supported styles
    ----------------
    **`style='SI'`** (default)
        **`r'$Temperature\\;/\\;\\mathrm{°C}$'`**
    **`style='in'`**
        **`r'$Temperature\\;in\\;\\mathrm{°C}$'`**,
    **`style='parentheses'`**
        **`r'$Temperature,\\;(\\mathrm{°C})$'`** (not encouraged),
    **`style='IEEE'`
        **`r'$Temperature\\;(\\mathrm{°C})$'`**, recommended for IEEE articles.
    If multiple quantities shall be labeled, f.i.
    ``Temperatur / °C, Leistung / kW``, `label` and `unit` must be tuples or
    lists of the same length, containing the required labels/units.
    The spacing between label, unit and, if any, other signs, is set with
    `spacing='...'`, defaulting to `'\;'`.

    Parameters
    ----------
    ax : ax reference
        Ax on which the labels has to be placed.
    label : str, tuple, list
        Ax label. Can contain LaTeX equations, but the LaTeX equation
        identifiers `$equation...$` must be omitted. F.i.
        `label='\dot{Q}_{TWE}'`.
    unit : str, tuple, list, optional
        SI unit of the quantitiy to label, f.i. `unit='kWh'`. If not given, the
        divisor sign will also be dropped.
    which : str
        Which axis to decorate. Can be `'y'`, `'x'`, `'z'` or `'cbar'`. For
        `'cbar'`, the cbar reference has to be passed with the `ax` parameter.
    style : str
        Formatting style to apply to the label. Defaults to 'SI' (recommended).
        Other allowed formats are 'in', f.i. Power in kW, or 'parentheses',
        Power in (kW), (not recommended).
        For IEEE publications the style 'IEEE' is recommended, producing
        `'Power (kW)'` (like 'parentheses' but without the word `'in'`).
    spacing : str
        Spacing to apply to the label. Refer to LaTeX equation spacing
        manual for a list of options. Default is '\;', which is a full space.

    **Copied from toolbox to have a build independent of the toolbox package.**
    """
    assert style in ('SI', 'in', 'parentheses', 'IEEE')
    if isinstance(label, str):
        label = (label,)
        assert isinstance(unit, str) or unit in ('', 'none', None)
        unit = (unit,)
    elif isinstance(label, (tuple, list)):
        assert (unit in ('', 'none', None)) or isinstance(unit, (tuple, list))
        if isinstance(unit, (tuple, list)):
            assert len(unit) == len(label)
        else:
            unit = (unit,) * len(label)

    full_axlabel = ''
    for lbl, unt in zip(label, unit):
        assert '$' not in lbl and unt is None or '$' not in unt, (
            'Do not pass a latex equation sign ($) to the labeling function. '
            'It will be added automatically. If equation output does not '
            'work, instead pass a raw string with `r\'label/unit\'`.'
        )

        unit_ = None if unt in ('', 'none') else unt

        # replace spaces in label string with latex space string
        label_ = lbl.replace(' ', spacing)
        # replace raw percentage signs with \%
        if unit_ is not None:
            unit_ = unit_.replace(r'\%', '%').replace('%', r'\%')
        # construct string
        if style == 'SI':
            if unit_ is not None:
                axlabel = r'${0}{3}{2}{3}\mathrm{{{1}}}$'.format(
                    label_, unit_, '/', spacing
                )
            else:
                axlabel = r'${0}$'.format(label_)
        elif style == 'in':
            if unit_ is not None:
                axlabel = r'${0}{2}in{2}\mathrm{{{1}}}$'.format(
                    label_, unit_, spacing
                )
            else:
                axlabel = r'${0}$'.format(label_)
        elif style == 'parentheses':
            if unit_ is not None:
                axlabel = r'${0}{2}in{2}(\mathrm{{{1}}})$'.format(
                    label_, unit_, spacing
                )
            else:
                axlabel = r'${0}$'.format(label_)
        elif style == 'IEEE':
            if unit_ is not None:
                axlabel = r'${0}{2}(\mathrm{{{1}}})$'.format(
                    label_, unit_, spacing
                )
            else:
                axlabel = r'${0}$'.format(label_)
        full_axlabel += axlabel
    full_axlabel = full_axlabel.replace('$$', ',' + spacing)
    # set to axis label
    if which == 'y':
        ax.set_ylabel(full_axlabel)
        if invert_position:
            ax.yaxis.set_label_position('right')
            ax.tick_params(
                left=False, right=True, labelleft=False, labelright=True
            )
    elif which == 'x':
        ax.set_xlabel(full_axlabel)
        if invert_position:
            ax.xaxis.set_label_position('top')
            ax.tick_params(
                bottom=False, top=True, labelbottom=False, labeltop=True
            )
    elif which == 'z':
        ax.set_zlabel(full_axlabel)
    elif which == 'cbar':
        ax.set_label(full_axlabel)


@override(tb, 'tp')
def annotate_axes(
    axes,
    fig=None,
    annotations=None,
    loc='top left',  # xy=[(.05, .95)],
    bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.8, pad=0.3),
    txt_offset=8,
    fontsize=9,
    **kwds
):
    """
    Create axes annotations like (a), (b), (c)...

    Parameters
    ----------
    axes : matplotlib.Axes
        DESCRIPTION.
    fig : matplotlib.Figure, optional
        DESCRIPTION. The default is None.
    annotations : list, tuple, string, optional
        DESCRIPTION. The default is None.
    loc : string, optional
        DESCRIPTION. The default is 'top left'.
    # xy : TYPE, optional
        DESCRIPTION. The default is [(.05, .95)].
    bbox : dict, optional
        DESCRIPTION. The default is dict(boxstyle="round", fc="w", ec="k", alpha=.8, pad=0.35).
    txt_offset : int, float, optional
        DESCRIPTION. The default is 8.
    **kwds : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    **Copied from toolbox to have a build independent of the toolbox package.**
    """
    if isinstance(fig, _mpl.figure.Figure):
        assert axes is None, (
            'If a figure is given with `fig`, axes need to be set to None '
            'with `axes=None`.'
        )
        axes = fig.get_axes()
    # if single axes given, store in list for convenience
    if not isinstance(axes, (list, tuple)):
        axes = [axes]
    # check if annotations are given and if not, use the alphabet:
    if annotations is None:
        annt = ['{0})'.format(ch) for ch in 'abcdefghijklmnopqrstuvwxyz']
    else:
        annt = annotations
    # check if annotations are given and if not, calculate positions:
    #    if len(xy) == 1:
    #        xy_perc = xy[0]  # backup percentag values
    #        xy = []
    #        for ax in axes:
    #            xlim, ylim = ax.get_xlim(), ax.get_ylim()
    #            xy.append(
    #                ((xlim[1] - xlim[0]) * xy_perc[0] + xlim[0],
    #                 (ylim[1] - ylim[0]) * xy_perc[1] + ylim[0]))
    # catch other args:
    xycoords = kwds['xycoords'] if 'xycoords' in kwds else 'axes fraction'
    ha = kwds['ha'] if 'ha' in kwds else 'center'
    va = kwds['va'] if 'va' in kwds else 'center'
    zorder = kwds['zorder'] if 'zorder' in kwds else 500

    assert loc in (
        'top left',
        'top right',
        'bottom left',
        'bottom right',
        'top center',
        'lower center',
        'center left',
        'center right',
    )
    if loc == 'top left':
        xy = (0, 1)
        xytext = (txt_offset, -txt_offset)
        ha, va = 'left', 'top'
    elif loc == 'top right':
        xy = (1, 1)
        xytext = (-txt_offset, -txt_offset)
        ha, va = 'right', 'top'
    elif loc == 'bottom left':
        xy = (0, 0)
        xytext = (txt_offset, txt_offset)
        ha, va = 'left', 'bottom'
    elif loc == 'bottom right':
        xy = (1, 0)
        xytext = (-txt_offset, txt_offset)
        ha, va = 'right', 'bottom'
    elif loc == 'top center':
        xy = (0.5, 1)
        xytext = (0, -txt_offset)
        ha, va = 'center', 'top'
    elif loc == 'lower center':
        xy = (0.5, 0)
        xytext = (0, txt_offset)
        ha, va = 'center', 'bottom'
    elif loc == 'center left':
        xy = (0, 0.5)
        xytext = (txt_offset, 0)
        ha, va = 'left', 'center'
    elif loc == 'center right':
        xy = (1, 0.5)
        xytext = (-txt_offset, 0)
        ha, va = 'right', 'center'
    # overwrite align if given:
    ha = kwds['ha'] if 'ha' in kwds else ha
    va = kwds['va'] if 'va' in kwds else va

    # iterate over axes:
    for i, ax in enumerate(axes):
        ax.annotate(
            s=annt[i],
            xy=xy,
            xycoords=xycoords,
            xytext=xytext,
            textcoords='offset points',
            ha=ha,
            va=va,
            zorder=zorder,
            bbox=bbox,
            fontsize=fontsize,
        )


# %% deprecated
def opt_control(
    *,
    simenv_creater_function,
    LabData,
    control,
    actuator,
    terms,
    start_values,
    labdata_flow,
    opt_flow_and_temp=False,
    labdata_time_offset=0,
    **kwargs
):

    if opt_flow_and_temp:
        part2 = kwargs['part2']
        cell2 = kwargs['cell2']
        labdata_temp = kwargs['labdata_temp']
        flow_only = False
    else:
        flow_only = True

    #    simenv = simenv_creater_function(
    #            Pumpe_FriWa_Kp=start_values[0], Pumpe_FriWa_Ki=start_values[1],
    #            Pumpe_FriWa_antiwindup=start_values[2], LabData=LabData,
    #            build_simenv=True, **kwargs)

    from scipy.optimize import minimize  # ,  # newton

    # wrapper function around phex alpha to feed this into minimizer:

    def simenv_wrapper(start_values):
        # create simulation environment
        simenv = simenv_creater_function(
            Pumpe_FriWa_Kp=start_values[0],
            Pumpe_FriWa_Ki=start_values[1],
            Pumpe_FriWa_antiwindup=start_values[2],
            LabData=LabData,
            build_simenv=True,
            **kwargs
        )

        #        i = 0
        #        for kx in start_values:
        #            if i == 0:
        #                simenv.ctrls[control].kp = kx
        #            elif i == 1:
        #                simenv.ctrls[control].ki = kx
        #            elif i == 2:
        #                simenv.ctrls[control].kd = kx
        #            i += 1
        #
        #        simenv.start_sim()

        # add meter at actuator to get flow and timeindex:
        meters = Meters(simenv=simenv, start_time=LabData.index[0])
        meters.massflow(name='dm_actuator', part=actuator, cell=0)
        # resample meter to evenly spaced data for comparison with LabData:
        even_massflow = process_unevenly_spaced_timeseries(
            data=meters.meters['massflow']['dm_actuator'],
            freq=pd.infer_freq(labdata_flow.index),
            how='forward_fill',
        )
        #        end = meters.meters['massflow']['dm_actuator'].index[-1]
        end = even_massflow.index[-1]

        # copy compared values and then delete class instance:
        act_res_dm = simenv.parts[actuator].res_dm.copy()
        if not flow_only:
            part2_res = simenv.parts[part2].res.copy()
        del simenv  # delete for full restart

        #        if flow_only:
        #            # get sum of deviations for the massflow
        #            print(end)
        #            print(act_res_dm.shape)
        #            print(labdata_flow[:end].values.shape)
        #            return (act_res_dm - labdata_flow[:end].values).sum()
        #        else:
        #            # get sum of deviations for massflow and one temperature
        #            return (
        #                (part2_res[:, cell2] - labdata_temp[:end]).sum()
        #                + (act_res_dm.res_dm - labdata_flow[:end]).sum())
        if flow_only:
            # get sum of deviations for the massflow
            print(end)
            print(even_massflow.values.shape)
            print(labdata_flow[:end].values.shape)
            return (even_massflow.values - labdata_flow[:end].values).sum()
        else:
            # get sum of deviations for massflow and one temperature
            return (part2_res[:, cell2] - labdata_temp[:end]).sum() + (
                act_res_dm.res_dm - labdata_flow[:end]
            ).sum()

    res = minimize(simenv_wrapper, start_values)

    return res
