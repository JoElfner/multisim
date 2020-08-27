# -*- coding: utf-8 -*-
# import os
import pytest
import sys

# sys.path.append('..')
import multisim as ms


# disable writing bytecode to avoid creating pycache
sys.dont_write_bytecode = True

# print('see https://eth-brownie.readthedocs.io/en/stable/tests-pytest-intro.html#isolation-fixtures'
#       'for how to revert states between tests.')
# print('and to parametrize tests: '
#       'https://eth-brownie.readthedocs.io/en/stable/tests-pytest-intro.html#parametrizing-tests')


# %% Test basic simulation invocation
@pytest.mark.parametrize('suppress_printing', [True, False])
def test_initsim(suppress_printing):
    ms.SimEnv(suppress_printing)


# make the tested state a fixture for further testing
@pytest.fixture(scope='module', params=[True, False])
def make_sim(request):
    return ms.SimEnv(request.param)


# and isolate it to allow reverting for each new test
@pytest.fixture
def isolation(fn_isolation):
    pass


# test timeframe init
@pytest.mark.parametrize(
    (
        'timeframe,adapt_steps,timestep,min_stepsize,max_stepsize,'
        'rtol,atol,max_factor,min_factor'
    ),
    [
        (0.1, True, 1.0, 1e-12, 1.0, 1e-8, 1e-8, 1.01, 1e-5),
        (1000, True, 1.0, 1, 100, 0.99, 0.99, 100, 0.99),
        (1000.0, False, 1.0, 1.0, 1.0, 1e-5, 1e-5, 10.0, 1e-3),
        (12345.0, True, 1.0, 1e-3, 10, 1e-3, 1e-3, 10, 1e-2),
    ],
)
def test_timeframe(
    make_sim,
    timeframe,
    adapt_steps,
    timestep,
    min_stepsize,
    max_stepsize,
    rtol,
    atol,
    max_factor,
    min_factor,
):
    make_sim.set_timeframe(
        timeframe=timeframe,
        adaptive_steps=adapt_steps,
        timestep=timestep,
        min_stepsize=min_stepsize,
        max_stepsize=max_stepsize,
        rtol=rtol,
        atol=atol,
        max_factor=max_factor,
        min_factor=min_factor,
    )


# test disk saving init
@pytest.mark.parametrize(
    (
        'save,evry_step,name,startdate,resample,freq,new_folder,overwrite,'
        'complvl,complib'
    ),
    [
        (True, 1.0, 'simmi', '2020-01-01', True, '1s', True, False, 9, 'zlib'),
        (True, 1000, None, '2020-01-01', True, '100s', False, True, 0, 'gzip'),
        (True, 100, 25.0, 1970, True, 25, True, False, 9, 'zlib'),
        (
            False,
            10000,
            'simmi',
            '2020-01-01',
            True,
            '1s',
            True,
            False,
            9,
            'zlib',
        ),
    ],
)
def test_diskaving(
    make_sim,
    save,
    evry_step,
    name,
    startdate,
    resample,
    freq,
    new_folder,
    overwrite,
    complvl,
    complib,
):
    make_sim.set_disksaving(
        save=save,
        save_every_n_steps=evry_step,
        sim_name=name,
        path=r'./test_output_files//',
        start_date=startdate,
        resample_final=resample,
        resample_freq=freq,
        create_new_folder=new_folder,
        overwrite=overwrite,
        complevel=complvl,
        complib=complib,
    )


# test solver init
@pytest.mark.parametrize(
    ('solver,allow_implicit'), [('heun', True), ('heun', False)]
)
def test_setsolver(make_sim, solver, allow_implicit):
    make_sim.set_solver(solver, allow_implicit=allow_implicit)


# %% run test
# pytest.main()

# %% remove all files created by test:
# remove files created by simulation
# fls_tof = os.listdir('./test_output_files')
# for f in fls_tof:
#     os.remove(os.path.join('./test_output_files', f))
