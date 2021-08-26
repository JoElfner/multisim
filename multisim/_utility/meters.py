# -*- coding: utf-8 -*-
"""
Created on 26 Aug 2021

@author: Johannes Elfner
"""

import numpy as np
import pandas as pd
import warnings

import multisim._precompiled.material_properties as _mp
import multisim.utility_functions as _uf


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
        df['volume_flow_m3ps'] = df['massflow_kgps'] / _mp.rho_water(
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
                / _mp.rho_water(df['T_rf'].values)
            )
        # get heatflow in [kW]
        df['heatflow_kW'] = (
            (df['T_ff'] - df['T_rf'])
            * df['massflow_kgps']
            / 1e3
            * (
                _mp.cp_water(df['T_rf'].values)
                + _mp.cp_water(df['T_ff'].values)
            )
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
            df = _uf.process_unevenly_spaced_timeseries(
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
        df[name] = self._simenv.parts[part].res_dm[:, idx_dm] / _mp.rho_water(
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

        warnings.warn(
            (
                '`pred_real_plot` is deprecated. Maybe it will be re-added in a '
                'future version'
            ),
            DeprecationWarning,
        )
        # sources: ashrae und schittgabler zeitreihenanalyse für R oder so...
        # move outside of meters to be able to use it on all kind of TS?
        # check if TS is even AND if freq is the same as measurement-TS-freq
