# -*- coding: utf-8 -*-
import pytest
import sys

import numpy as np

import multisim as ms


# disable writing bytecode to avoid creating pycache
sys.dont_write_bytecode = True


# %% make fixture of simenv to avoid having to rebuild the basics for each test

# make the tested state a fixture for further testing
@pytest.fixture(scope='function', params=[True])
def make_sim(request):
    simenv = ms.SimEnv(request.param)
    simenv.set_timeframe(timeframe=100, adaptive_steps=True)
    simenv.set_disksaving(save=False, sim_name='sim_a')
    simenv.set_solver(solver='heun', allow_implicit=False)
    return simenv


# and isolate it to allow reverting for each new test
@pytest.fixture
def isolation(fn_isolation):
    pass


# %% test some basic parts
# test timeframe init
@pytest.mark.parametrize(
    ('theta_init,theta_amb,massflow,expected'),
    [
        (
            60 + 20 * np.sin(np.linspace(0, 2 * np.pi, num=20)),
            10.0,
            0.0,
            np.array(
                [
                    60.5,
                    66.15,
                    71.88,
                    76.3,
                    78.92,
                    79.46,
                    77.85,
                    74.29,
                    69.14,
                    62.97,
                    56.45,
                    50.28,
                    45.13,
                    41.56,
                    39.96,
                    40.5,
                    43.12,
                    47.54,
                    53.26,
                    58.42,
                ]
            ),
        ),
        (
            30 + 10 * np.cos(np.linspace(0, 2 * np.pi, num=20)),
            50.0,
            0.2,
            np.array(
                [
                    85.0,
                    85.0,
                    85.0,
                    85.0,
                    84.99,
                    84.99,
                    84.99,
                    84.99,
                    84.99,
                    84.99,
                    84.99,
                    84.99,
                    84.98,
                    84.98,
                    84.98,
                    84.98,
                    84.98,
                    84.98,
                    84.98,
                    84.97,
                ]
            ),
        ),
    ],
)
def test_pipe_with_pump(make_sim, theta_init, theta_amb, massflow, expected):
    testsim = make_sim

    # define pipe specifications for all pipes and ports
    pipe_specs = {'all': {'pipe_type': 'EN10255_medium', 'DN': 'DN25'}}
    # set general specifications for all parts
    general_specs = dict(
        insulation_thickness=1e-2,
        insulation_lambda=0.035,
        T_init=theta_init,
        T_amb=theta_amb,
        material='carbon_steel',
        pipe_specs=pipe_specs,
    )

    testsim.add_part(
        part=ms.ap.PipeWithPump,
        name='pipe',
        length=2.0,
        grid_points=20,
        start_massflow=massflow,
        ctrl_required=False,
        const_val=massflow,
        **general_specs,
    )

    testsim.add_open_port('BC_in', constant=True, temperature=85.0)
    testsim.add_open_port('BC_out', constant=True, temperature=15.0)

    testsim.connect_ports(
        first_part='BoundaryCondition',
        first_port='BC_in',
        scnd_part='pipe',
        scnd_port='in',
    )
    testsim.connect_ports(
        first_part='BoundaryCondition',
        first_port='BC_out',
        scnd_part='pipe',
        scnd_port='out',
    )

    testsim.initialize_sim()
    testsim.start_sim()

    assert np.all(testsim.parts['pipe'].res[-1].round(2) == expected)


# %% run tests locally
if __name__ == 'main':
    pytest.main()
