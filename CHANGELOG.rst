=========
Changelog
=========

| BUG: Bug resolved
| ENH: Enhancement of the module
| DEP: Deprecation
| MAINT: Changes to the package environment
| DOC: Changes to the package documentation
| BLD: Changes to the package build

0.10.0 - 2020-12-08
==================

* MAINT: API: Main class `Models` renamed to `SimEnv`. `Models` can still be imported/used for backwards compatibility, but will be deprecated in a future release.
* DEP: Many deprecated/commented LOC removed.
* DEP: :py:func:`utility_functions.NetPlotter`: now raises a deprecation warning. The only remaining supported method is `plot_errors`. Will be removed in future releases.
* ENH: Started integrating a new way of checking for kwds in :py:func:`parts.pipe.Pipe` when supporting two different arg naming conventions.
* ENH: :py:func:`SimEnv.start_sim`: Started refactoring. Next refactor `_finalize_sim`, especially disk saving with context manager.
* BUG: dict.get() misplaced argument bug resolved.
* BLD: Required package versions updated.

0.9.9 - 2020-08-26
==================

* MAINT: API-cleaning: Name mangled most imports.
* DEP: Removed roughly 5k lines of deprecated or commented code. Many more to go...
* MAINT: Fully :py:func:`black`'ed and :py:func:`flake8`'ed all parts of the package.
* DEP: Some non-optimized parts removed from the package.
* BUG: :py:func:`utility_functions.plt_prediction_realization` bugs resolved.
* ENH: Many commented lines removed.
* ENH: :py:func:`utility_functions` minor changes to argument handling and error naming conventions.

0.9.8 - 2020-07-23
==================

* BLD: Basic pytest testing added in `./tests`
* BLD: Travis CI configured with flake8, mypy and basic pytest testing.
* ENH: `simenv.Models` now supports storing arbitrary data to disk. Data storing methods can be registered to `Models._disk_store_utility` during class initialization.
* BUG: `controllers.ModelPredCHP`: Critical bug resolved, causing the modelpredictive control to optimize over the total kumulative heat demand instead of the expected kumulative heat demand.
* BUG: `controllers.ModelPredCHP`: Critical bug resolved, causing the modelpredictive control to not increase the optimization step counter when the emergency control hits, cause a delay in the profile slices used for optimization.
* ENH: `controllers.ModelPredCHP._tes_heat_content`: TES energy content calculation refactored.
* ENH: `controllers.ModelPredCHP._longterm_clip_w_cont`: Long-term optimizer continuity enhanced to treat current step result depending on last step's result.
* ENH: `controllers.ModelPredCHP`: Emergency CHP activation implemented as a two-level control: If the PV falls below setpoint, a mixed mode is activated. This is a 50:50 compound of full power and electricity profile optimized control. If the PV falls below the setpoint minus an additional threshold, the CHP plant goes into full power operation.
* ENH: `controllers.ModelPredCHP._adapt_cap`: Upper and lower TES capacities used for estimating the remaining capacity before full/empty are adapted each time the emergency control hits.
* ENH: `controllers.ModelPredCHP._restore_adptd_caps`: Adapted capacities are slowly restored to initial values of a given timeframe, default 24h.
* ENH: `controllers.ModelPredCHP`: Adaption and restoring of TES capacities is performed only emergency control was active recently and/or the capacities have not been fully restored.
* ENH: `controllers.ModelPredCHP`: Modelpredictive optimization results can now be stored on disk.
* ENH: `controllers.ModelPredCHP._longterm_clip_array`: Method added to clip longterm optimization results to modulation range for TES SOC checking.
* ENH: `controllers.ModelPredCHP._longterm_adj_by_soc`: For badly scaled CHP plants, optimization may yield false results (f.i. CHP P_el is much larger than the required power, thus modulation is set to values <.5). In these cases, longterm optimization results are adjusted by the predicted SOC.
* ENH: `controllers.ModelPredCHP`: Major refactoring of the methods. There is still a massive potential for further refactoring, removing deprecated/commented code, breakpoints etc...

0.9.7 - 2020-06-15
==================

* ENH: `parts.consumer.space_heating` added.
* ENH: `parts.consumer.speicherladesys` added.
* ENH: `parts.pipe.Pipe.init_part()` argument error corrected.
* ENH: `set_disksaving` can now infer the start_date from boundary conditions.
* ENH: `utility_functions.Meters.heat_meter` can now calculate heat flow from massflows at the hot part.
* ENH: `utility_functions.Meters.heat_meter` by default reduces the output by omitting the positive-only flows.
* ENH: `utility_functions.Meters.heat_meter` unit of power changed W -> kW and of energy J -> MJ.
* ENH: `parts.heated_pipe` emergency shutdown when surpassing a temperature added, default 110°C.
* ENH: isinstance checks for int and float will be expanded step-by-step to also check for np.int and np.float to avoid some occasionally occurring errors.
* MAINT: Lots of dead code removed.
* BUG: Topology handling with operation routine 5 improved, resp. error message for unsafe topology added.
* MAINT: `utility_functions.NetPlotter` reintegrated and basic error plotting functionality restored.
* ENH: Local variable `time` in `Models.start_sim` method is now an instance variable named `time_sim` for simulation environment wide access.
* ENH: `precomp_funs.startup_factor_gas` added as a compound factor consisting of the thermal and electrical startup scaled by the efficiencies given in the XRGi20 datasheet.
* ENH: `parts.chp_plant.CHPPlant`: Integration of `precomp_funs.startup_factor_gas` to calculate the gas input during startup.
* ENH: `controllers.ModelPredCHP`: Model predictive controller for CHP plant added. The MPC consists of 3 layers: First/outer layer for switching on/off the CHP plant if the heat storage is empty/full, overrides layers 2 and 3; second/middle layer for optimizing the the CHP plant operation schedule by means of an opertation cost function constrained by the TES SOC and mean heat/electric demands in a selected timeframe every few seconds (default: timeframe of 2 days every 900 seconds); third/inner layer to optimize the CHP modulation in each step by means of an operation cost function, depending on decisions made in the second layer.
* ENH: Model predictive control electricity led CHP plant added to `suppliers.chp_with_fghex`.
* BLD: `setup.py` version dependencies updated, most specifically now requiring Python >= 3.7.
* ENH: `parts.part_modules.supplier.chp_with_fg_hex` fully integrated model predictive control.

0.9.6 - 2020-03-05
==================

* ENH: `utility_functions.package_results` now also accepts absolute paths for `move_to`.
* BLD: All references to external non-standard modules (not available on PyPI removed, most notably to `toolbox` module.
* ENH: Functions in `utility_functions` that previously required `toolbox` module implemented directly.
* ENH: Functions in `utility_functions` that previously required `toolbox` module will be overriden by by `toolbox` implementations if `toolbox` module is installed.
* MAINT: `setup.py` now requires at least a specific version of the required module.
* BUG: `setup.py` `install_requires` previously had 'sklearn' as a requirement. This is the import name. Replaced with the correct module name 'scikit-learn'.
* DOC: `setup.py` classifiers added.
* BLD: `setup.cfg` added with `[bdist_wheel] universal=0` and included license.
* BLD: `bdist_wheel --universal` removed from `setup.py`.
* BLD: `setup.py` automatically sets `bdist_wheel --python-tag`.
* ENH: `utility_functions.package_results` path finding optimized.

0.9.5 - 2020-03-03
==================

* DOC: `INSTALL.rst.txt` added with instructions on how to package, build, install and distribute MultiSim.
* MAINT: `setup.py` now supports automatic upload to PyPI via `twine` with `python setup.py upload`
* ENH: Automatic version numbering in `setup.py` download URL.
* DOC: `README.rst` updated.

0.9.4 - 2020-03-02
==================

* ENH: Main sim. class `Models` made directly accessible from top-level package.
* DEP: Access to `multisim.se` will be restricted in oncoming versions.
* DOC: Changelog formatting improved.
* BLD: `setup.py` tweaked for PyPI distribution.
* BLD: Package released on Github (private repo).
* BLD: `.gitignore` added

0.9.3 - 2020-02-19
==================

* ENH: Import of submodules in package `__init__.py` to enable top-level access to submodules.
* ENH: `utility_functions.load_sim_results` now takes a `keys` argument to only load specific columns.
* ENH: `utility_functions.load_results_by_name` takes and passes on the `keys` result.
* ENH: `utility_functions.load_sim_results` works by concatenating columns instead of copying value arrays, increasing performance by a factor of about 30.
* DOC: Changelog description added.

0.9.2 - 2020-02-14
==================

* ENH: `utility_functions.package_results` function added to allow for easy structuring of results.
* ENH: `utility_functions.load_results_by_name` convenience wrapper added for `load_sim_results` to allow easy pathless loading of structured results.

0.9.1 - 2020-02-11
==================

* ENH: Added bypassing to hex_condensing_polynome to allow for massflows >> max water massflow.

0.9.0 - 2020-02-11
==================

* Initial release
* All imports made relative imports
* Packaging of the simulation environment started