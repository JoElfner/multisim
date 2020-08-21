# -*- coding: utf-8 -*-
import os
import pytest
import sys

# from .context import multisim as ms
import multisim as ms

# disable writing bytecode to avoid creating pycache
sys.dont_write_bytecode = True

print(
    'see https://eth-brownie.readthedocs.io/en/stable/tests-pytest-intro.html#isolation-fixtures'
    'for how to revert states between tests.'
)
print(
    'and to parametrize tests: '
    'https://eth-brownie.readthedocs.io/en/stable/tests-pytest-intro.html#parametrizing-tests'
)


# %% Init sim model as isolated fixture
# make the tested state a fixture for further testing
@pytest.fixture(
    scope='module',
    autouse=True,
    params=[
        dict(
            set_tf=dict(timeframe=500, adaptive_steps=True),
            set_ds=dict(
                save=True,
                save_every_n_steps=50,
                start_date='2020-01-01',
                resample_final=True,
                resample_freq='1s',
            ),
            set_slvr=dict(solver='heun', allow_implicit=True),
        ),
        dict(
            set_tf=dict(timeframe=500, adaptive_steps=True),
            set_ds=dict(
                save=True,
                save_every_n_steps=50,
                start_date='2020-01-01',
                resample_final=False,
                resample_freq='1s',
            ),
            set_slvr=dict(solver='heun', allow_implicit=True),
        ),
        dict(
            set_tf=dict(timeframe=500, adaptive_steps=False, timestep=1.0),
            set_ds=dict(
                save=True,
                save_every_n_steps=50,
                start_date='2020-01-01',
                resample_final=False,
                resample_freq='1s',
            ),
            set_slvr=dict(solver='heun', allow_implicit=True),
        ),
        dict(
            set_tf=dict(timeframe=500, adaptive_steps=False, timestep=1.0),
            set_ds=dict(
                save=False,
                save_every_n_steps=50,
                start_date='2020-01-01',
                resample_final=True,
                resample_freq='1s',
            ),
            set_slvr=dict(solver='heun', allow_implicit=True),
        ),
        dict(
            set_tf=dict(timeframe=500, adaptive_steps=True),
            set_ds=dict(
                save=True,
                save_every_n_steps=50,
                start_date='2020-01-01',
                resample_final=True,
                resample_freq='1s',
            ),
            set_slvr=dict(solver='heun', allow_implicit=False),
        ),
        dict(
            set_tf=dict(timeframe=500, adaptive_steps=True),
            set_ds=dict(
                save=True,
                save_every_n_steps=50,
                start_date='2020-01-01',
                resample_final=False,
                resample_freq='1s',
            ),
            set_slvr=dict(solver='heun', allow_implicit=False),
        ),
        dict(
            set_tf=dict(timeframe=500, adaptive_steps=False, timestep=1.0),
            set_ds=dict(
                save=True,
                save_every_n_steps=50,
                start_date='2020-01-01',
                resample_final=False,
                resample_freq='1s',
            ),
            set_slvr=dict(solver='heun', allow_implicit=False),
        ),
    ],
)
def make_sim(request):
    # init sim
    sim = ms.Models(suppress_printing=False)
    # set timeframe:
    sim.set_timeframe(
        **request.param['set_tf'],
        min_stepsize=1e-3,
        max_stepsize=10.0,
        rtol=1e-3,
        atol=1e-3,
        max_factor=10,
        min_factor=1e-2
    )
    # set disksaving:
    sim.set_disksaving(
        **request.param['set_ds'],
        path=r'./test_output_files//',
        create_new_folder=True,
        overwrite=True,
        complevel=5,
        complib='zlib'
    )
    # set solver:
    sim.set_solver(**request.param['set_slvr'])


# and isolate it to allow reverting for each new test
@pytest.fixture
def isolation(fn_isolation, autouse=True):
    pass


# %% Test adding parts
# test adding TES
# =============================================================================
# @pytest.mark.parametrize(
#     ('volume,gridpoints,tes_do,tes_new_ports,tes_init,store_results'), [
#         (.5, 10, 2., newpor, tinit, stres8, ),
#         (2.1, 100, 1., newpor, tinit, stres8, ),
#         (.5, 10, 2., newpor, tinit, stres8, ),
#         (2.1, 100, 1., newpor, tinit, stres8, ),
#         (.5, 10, 2., newpor, tinit, stres8, ),
#         (2.1, 100, 1., newpor, tinit, stres8, ),])
# def test_add_tes(
#         make_sim, volume, gridpoints, tes_do, tes_new_ports,
#         tes_init, store_results):
#     make_sim.add_part(
#         ms.ap.Tes, name='tes', volume=volume,
#         grid_points=gridpoints, outer_diameter=tes_do,
#         shell_thickness=5e-3, new_ports=tes_new_ports,
#         insulation_thickness=0.2, insulation_lambda=.035,
#         T_init=tes_init, T_amb=25, material='carbon_steel',
#         pipe_specs={
#             'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN40'}},
#         store_results=store_results)
# =============================================================================

# # test disk saving init
# @pytest.mark.parametrize(
#     ('save,evry_step,name,startdate,resample,freq,new_folder,overwrite,'
#      'complvl,complib'), [
#         (True, 1., 'simmi', '2020-01-01', True, '1s',
#          True, False, 9, 'zlib'),
#         (True, 1000, None, '2020-01-01', True, '100s',
#          False, True, 0, 'gzip'),
#         (True, 100, 25., 1970, True, 25,
#          True, False, 9, 'zlib'),
#         (False, 10000, 'simmi', '2020-01-01', True, '1s',
#          True, False, 9, 'zlib')])
# def test_diskaving(
#         make_sim, save, evry_step, name, startdate, resample, freq,
#         new_folder, overwrite, complvl, complib):
#     make_sim.set_disksaving(
#         save=save, save_every_n_steps=evry_step, sim_name=name,
#         path=r'./test_output_files//',
#         start_date=startdate, resample_final=resample, resample_freq=freq,
#         create_new_folder=new_folder,
#         overwrite=overwrite, complevel=complvl, complib=complib
#     )


# %% run test
# pytest.main()

# %% remove all files created by test:
# remove files created by simulation
fls_tof = os.listdir('./test_output_files')
for f in fls_tof:
    os.remove(os.path.join('./test_output_files', f))
