# -*- coding: utf-8 -*-
"""
Created on 28 Aug 28 2021

@author: Johannes Elfner
"""

import copy as _copy

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as _mal
import numpy as _np
import pandas as _pd
import scipy.stats as _sst

from . import stat_error_measures as _sem


def heatmap_from_df(
    df,
    ax=None,
    figsize=(16 / 2.54, 10 / 2.54),
    cbar=True,
    cbar_ax=None,
    cmap='plasma',
    ylabel=None,
    cbar_label=None,
    label_style='SI',
    vmin=None,
    vmax=None,
    linewidth=0,
    limit_to_valid_data=True,
    log_cbar=False,
    plt_kwds={},
    cbar_kwds={},
    extend_over=None,
    extend_under=None,
):
    """
    Plot heatmap from dataframe.

    Parameters
    ----------
    df : pandas DataFrame
        DESCRIPTION.
    ax : matplotlib.axes, optional
        Axes to plot on. If not provided, a new figure will be created. The
        default is None.
    figsize : tuple, optional
        Figure size in inches. The default is (16 / 2.54, 10 / 2.54).
    cbar : bool, optional
        Plot colorbar? The default is True.
    cbar_ax : matplotlib.axes, optional
        Axes to plot colorbar on. The default is None.
    cmap : str, optional
        Colormap to use. The default is 'plasma'.
    ylabel : None, str, optional
        String to use as ylabel. The default is None.
    cbar_label : None, str, optional
        String to use as colorbar label. The default is None.
    label_style : str, optional
        Label formatting style to use with `si_axlabel`. The default is 'SI'.
    vmin : None, int, float, optional
        vmin to pass to matplotlib.pcolormesh. The default is None.
    vmax : None, int ,float, optional
        vmax to pass to matplotlib.pcolormesh. The default is None.
    linewidth : int, float, optional
        linewidth (lines between filled areas) to pass to
        matplotlib.pcolormesh. The default is 0.
    limit_to_valid_data : bool, optional
        Find first valid indices for x and y axis. Cuts out np.nan areas.
        The default is True.
    log_cbar : bool, optional
        Logarithmic scaling for colorbar. The default is False.
    plt_kwds : dict, optional
        Additional arguments to pass on to matplotlib.pcolormesh. The default
        is {}.
    cbar_kwds : dict, optional
        Additional arguments to pass on to the colorbar. The default is {}.
    extend_over : None, str, tuple, optional
        Set color for out-of-bound values larger than vmax. Will be applied to
        `cmap.set_over()`. The default is None.
    extend_under : None, str, tuple, optional
        Set color for out-of-bound values lower than vmin. Will be applied to
        `cmap.set_under()`. The default is None.

    Returns
    -------
    fig : matplotlib.figure
        Figure containing the plot.
    ax : matplotlib.axes
        Axes containing the plot.

    """

    assert isinstance(
        df, _pd.DataFrame
    ), '`df` must be a pandas DataFrame.'  # assert that df is a pd dataframe

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # cut to non-nan data range
    if limit_to_valid_data:
        # transposed plotting, thus transposed index finding
        fvxi = df.T.first_valid_index()  # first valid x entry
        fvyi = df.first_valid_index()  # first valid y entry
        lvxi = df.T.last_valid_index()  # last valid x entry
        lvyi = df.last_valid_index()  # last valid y entry
        df = df.copy().loc[fvyi:lvyi, fvxi:lvxi]

    # set ax frequency if period index data is given
    if isinstance(df.index, _pd.PeriodIndex):
        ax.xaxis.freq = df.index.freq.rule_code
    if isinstance(df.columns, _pd.PeriodIndex):
        ax.yaxis.freq = df.columns.freq.rule_code

    X, Y = _np.meshgrid(range(df.shape[0]), df.columns)
    X, Y = _np.meshgrid(df.index, df.columns)

    # make log colorbar
    if log_cbar:
        plt_kwds['norm'] = mpl.colors.LogNorm()
    # extend colorbar
    # get cbar and copy it only for extension to avoid altering other
    # figure's cbars
    if extend_over is not None or extend_under is not None:
        import copy

        cmap = copy.copy(plt.cm.get_cmap(cmap))
    if extend_over is not None:
        if extend_over == 'cmap':
            extend_over = cmap.colors[-1]
        cmap.set_over(extend_over)
        cbar_kwds['extend'] = 'max' if extend_under is None else 'both'
    if extend_under is not None:
        if extend_under == 'cmap':
            extend_under = cmap.colors[0]
        cmap.set_under(extend_under)
        cbar_kwds['extend'] = 'min' if extend_over is None else 'both'

    cax = ax.pcolormesh(
        X,
        Y,
        df.T,
        vmax=vmax,
        vmin=vmin,
        cmap=cmap,
        linewidth=linewidth,
        antialiased=False,
        **plt_kwds
    )
    if ylabel is not None:
        assert isinstance(ylabel, (tuple, list)) and len(ylabel) == 2
        si_axlabel(
            ax=ax,
            label=ylabel[0],
            unit=ylabel[1],
            which='y',
            style=label_style,
        )
    if cbar:
        if cbar_ax is None:
            cbar = fig.colorbar(cax, ax=ax, **cbar_kwds)
        else:
            cbar = fig.colorbar(mappable=cax, cax=cbar_ax, **cbar_kwds)
        if cbar_label is not None:
            assert (
                isinstance(cbar_label, (tuple, list)) and len(cbar_label) == 2
            )
            si_axlabel(
                ax=cbar,
                label=cbar_label[0],
                unit=cbar_label[1],
                which='cbar',
                style=label_style,
            )

    ax.autoscale(tight=True)
    return fig, ax


def prediction_realization_scatter(
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
    language='eng',
    plot_every=1,
    fig_kwds=dict(figsize=(8 / 2.54, 8 / 2.54)),
    err_vals=None,
):
    """
    Plot prediction-realization (PR) scatter plot.

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Realization/measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    ax : matplotlib.axes, optional
        Axes to plot on. If not provided, a new figure will be created. The
        default is None.
    aspect : str, int, float, optional
        Aspect ratio to use for axes. The default is 'eqaul'.
    ax_scaling_tight : bool, optional
        Use tight scaling for the axes. The default is False.
    errors : tuple, optional
        Statistical error measures to print in the plot. The default is
        ('R2', 'MSE', 'CV(RMSE)', 'NMBE').
    scttr_kwds : dict, optional
        Additional arguments to pass on to matplotlib.scatter. The default
        is dict(c='C0', s=6, fc='none', ec='C0', alpha=1.0).
    diag_kwds : dict, optional
        Additional arguments to pass on to plotting the halfing diagonal. The
        default is dict(color='k', ls='-', lw=1).
    plt_err_range : str, optional
        Plot error range around the diagonal. The default is 'RMSE'.
    err_rng_kwds : dict, optional
        Additional arguments to pass on to plotting the error range. The
        default is dict(color='k', ls='--', lw=1).
    err_kwds : dict, optional
        Additional arguments to define the box-style around the error measures.
        The default is dict(
        bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.5, pad=0.2)).
    err_loc : str, optional
        Where to place the box with the error measures. The default is
        'bottom right'.
    err_rng_kwds : dict, optional
        Additional arguments to pass on to legend creation. The default is
        dict().
    auto_label : bool, optional
        Label x- and y-axis automatically using SI-style.
    language : str, optional
        Language to use for labeling. The default is 'eng'.
    plot_every : int, optional
        Plot every n points to reduce plot size. The default is 1.
    fig_kwds : dict, optional
        Additional arguments to pass on to figure creation. The
        default is dict(figsize=(8 / 2.54, 8 / 2.54)).
    err_vals : None, tuple, optional
        Error values to use for annotation. The default is None.

    Returns
    -------
    fig : matplotlib.figure
        Figure containing the plot.
    ax : matplotlib.axes
        Axes containing the plot.

    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kwds)
    else:
        fig = ax.get_figure()

    pr_min, pr_max = (
        _np.min([y.min(), y_hat.min()]),
        _np.max([y.max(), y_hat.max()]),
    )
    minmax_range = pr_max - pr_min

    assert language in ('eng', 'de')
    if language == 'eng':
        datap = 'Data points'
        bsline = 'Bisecting line'
        xlabel = 'Measurement'
        ylabel = 'Prediction'
    elif language == 'de':
        datap = 'Datenpunkte'
        bsline = '$y = x$'
        xlabel = 'Messwerte'
        ylabel = 'Vorhersage'

    ax.scatter(y[::plot_every], y_hat[::plot_every], label=datap, **scttr_kwds)
    ax.plot(
        [pr_min - 0.05 * minmax_range, pr_max + 0.05 * minmax_range],
        [pr_min - 0.05 * minmax_range, pr_max + 0.05 * minmax_range],
        label=bsline,
        **diag_kwds
    )

    annotate_errors(
        ax=ax,
        y=y,
        y_hat=y_hat,
        errors=errors,
        err_loc=err_loc,
        err_vals=err_vals,
        err_kwds=err_kwds,
    )

    ax.set_aspect(aspect)
    if ax_scaling_tight:
        ax.autoscale(tight=True)
    else:
        lims = (pr_min - 0.05 * minmax_range, pr_max + 0.05 * minmax_range)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_ylim(ax.get_xlim())

    if auto_label:
        si_axlabel(ax, label=ylabel)
        si_axlabel(ax, label=xlabel, which='x')

    ax.grid(True)

    ax.legend(**legend_kwds)

    fig.tight_layout(pad=0)

    return fig, ax


def prediction_realization_2d_kde(
    y,
    y_hat,
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
    Plot 2-dimensional prediction-realization (PR) KDE plot.

    Uses gaussian kernel density estimate

    Parameters
    ----------
    y : np.array, pd.Series, pd.DataFrame
        Realization/measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    steps : int, optional
        Steps to use for calculation the gaussian kernel density estimate. The
        default is 100.
    ax : matplotlib.axes, optional
        Axes to plot on. If not provided, a new figure will be created. The
        default is None.
    contour : bool, optional
        Plot as a contour plot. The default is True.
    cmap : str, optional
        Colormap to use. The default is 'Blues'.
    norm : bool, optional
        Normalize linearly to vmin-vmax range **OR** set to 'log' for
        logarithmic normalization between vmin and vmax. The default is True.
    vmin : None, int, float, optional
        Minimum value to display. The default is None.
    vmax : None, int ,float, optional
        Maximum value to display. The default is None.
    cont_lines : bool, optional
        Display contour lines if plotting as a contour plot. The default
        is True.
    cbar : bool, optional
        Plot colorbar. The default is True.
    line_color : str, optional
        Color for halfing diagonal. The default is 'k'.
    fontsize : int, optional
        Font size for error measures. The default is 8.
    aspect : str, int, float, optional
        Aspect ratio to use for axes. The default is 'eqaul'.
    extend : str, optional
        Extend values or clip under vmin, over vmax or both. The default
        is 'both'.
    cm_under : str, optional
        Color to use for values clipped under vmin. The default is 'w'.
    cm_over : str, optional
        Color to use for values clipped over vmax. The default is 'k'.
    errors : tuple, optional
        Statistical error measures to put into axis. The default
        is ('R2', 'CV(RMSE)', 'NMBE').
    plt_kwds : dict, optional
        Additional arguments to pass on to the plotting methods (either
        matplotlib.imshow, matplotlib.contour or matplotlib.contourf). The
        default is dict().
    **err_kwds : keyword arguments, optional
        Additional keyword arguments to pass to error calculation

    Returns
    -------
    {'fig': fig, 'ax': ax, 'mappable': im, 'cbar': cbar} : dict
        Dictionary of created plotting objects.

    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    assert isinstance(steps, (int, complex))
    # calculate gaussian kernel
    steps = 100 * 1j if isinstance(steps, int) else steps
    m1, m2 = y.copy(), y_hat.copy()
    xmin, xmax, ymin, ymax = m1.min(), m1.max(), m2.min(), m2.max()
    X, Y = _np.mgrid[xmin:xmax:steps, ymin:ymax:steps]
    positions = _np.vstack([X.ravel(), Y.ravel()])
    values = _np.vstack([m1, m2])
    kernel = _sst.gaussian_kde(values)
    Z = _np.reshape(kernel(positions).T, X.shape)

    # copy cmap and extend if requested
    _cmap = _copy.copy(mpl.cm.get_cmap(cmap))
    if extend in ('both', 'max'):
        _cmap.set_over(cm_over)
    if extend in ('both', 'min'):
        _cmap.set_under(cm_under)

    if norm and norm != 'log':
        vmin_, vmax_ = Z.min(), Z.max()
        # overwrite norm values if given explicitly
        vmin = vmin if vmin is not None else vmin_
        vmax = vmax if vmax is not None else vmax_
        cont_kwds = {}
    elif norm == 'log':
        if _np.any(Z < 0.0):
            raise ValueError('Values below 0 not supported for log scale')
        # add min val to allow log sampling, since 0 is also not supported
        Z += 1e-9
        vmin_, vmax_ = Z.min(), Z.max()
        vmin = vmin if vmin is not None else vmin_
        vmax = vmax if vmax is not None else vmax_
        cont_kwds = {
            'norm': mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
            'locator': mpl.ticker.LogLocator(),
        }
    if not contour:  # just plot the kde as an image/pcolormesh
        imkwds = (
            {  # imshow does not support locator, thus only norm
                'norm': mpl.colors.LogNorm(vmin, vmax)
            }
            if norm == 'log'
            else {}
        )
        im = ax.imshow(
            _np.rot90(Z),
            cmap=_cmap,
            **plt_kwds,
            extent=[xmin, xmax, ymin, ymax],
            vmin=vmin,
            vmax=vmax,
            **imkwds
        )
    else:  # plot it as a filled contour
        try:  # This has many problems with log, thus also try tricont
            im = ax.contourf(
                X,
                Y,
                Z,
                cmap=_cmap,
                extend=extend,
                vmin=vmin,
                vmax=vmax,
                **cont_kwds
            )
        except ValueError:
            im = ax.tricontourf(
                X.reshape(-1),
                Y.reshape(-1),
                Z.reshape(-1),
                cmap=_cmap,
                extend=extend,
                vmin=vmin,
                vmax=vmax,
                **cont_kwds
            )
        if cont_lines:  # add contour lines
            clines = ax.contour(
                X,
                Y,
                Z,
                **plt_kwds,
                extend=extend,
                vmin=vmin,
                vmax=vmax,
                colors=line_color,
                **cont_kwds
            )
            ax.clabel(clines, inline=1, fontsize=fontsize)

    # Plot colorbar
    if cbar:
        cax_div = _mal(ax).append_axes('right', size='5%', pad=0.05)
        cbar = plt.colorbar(mappable=im, cax=cax_div)
        # cbar.update_bruteforce(im)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_aspect(aspect)
    ax.grid(True)

    annotate_errors(ax=ax, y=y_hat, y_hat=y, errors=errors, err_kwds=err_kwds)

    return {'fig': fig, 'ax': ax, 'mappable': im, 'cbar': cbar}


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


def sinum_frmt(x):
    """
    Generate SIunitx style formatted number output for plotting tables to TeX.

    This function can be used as a formatter for printing DataFrames to Latex
    tabes when using the Latex package SIunitx.

    Parameters
    ----------
    x : int, float
        Value to format.

    Returns
    -------
    str
        Formatted value.

    """
    if isinstance(x, (int, float)):
        if x < 1e3:
            return r'\num{' + '{0:.3G}'.format(x) + '}'
        elif x >= 1e3:
            return r'\num{' + '{0:.0f}'.format(x) + '}'
        else:  # x is nan
            return '-'
    else:
        return x


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
    Annotate statistical error measures in a plot.

    Parameters
    ----------
    ax : matplotlib.axes
        Axes to add error measures to.
    y : np.array, pd.Series, pd.DataFrame
        Realization/measurement/oberserved data.
    y_hat : np.array, pd.Series, pd.DataFrame
        Predicted/forecast/simulated data.
    errors : tuple
        Errors to calculate.
    err_loc : str, optional
        Location where to print errors in the axes. The default
        is 'bottom right'.
    err_vals : None, dict, optional
        Instead of calculating errors, use these values. The default is None.
    fontsize : int, optional
        Fontsize to use for printing errors. The default is 8.
    err_kwds : dict, optional
        Box style to use for printing errors. The default is
        dict(bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.5, pad=0.2)).

    Returns
    -------
    None.

    """
    annot_str = ''
    if 'R2' in errors:
        if err_vals is None or 'R2' not in err_vals:
            r2 = _sem.r_squared(y, y_hat)
        else:
            r2 = err_vals['R2']
        annot_str += r'$R^2={0:.3f}$'.format(r2) + '\n'
    if 'MSE' in errors:
        if err_vals is None or 'MSE' not in err_vals:
            mse = _sem.mean_squared_err(y, y_hat)
        else:
            mse = err_vals['MSE']
        annot_str += r'$MSE={0:.3G}$'.format(mse) + '\n'
    if 'CV(RMSE)' in errors:
        if err_vals is None or 'CV(RMSE)' not in err_vals:
            cvrmse = _sem.cv_rmse(y, y_hat)
        else:
            cvrmse = err_vals['CV(RMSE)']
        annot_str += r'$CV(RMSE)={0:.3f}$'.format(cvrmse) + '\n'
    if 'NMBE' in errors:
        if err_vals is None or 'NMBE' not in err_vals:
            nmbe = _sem.normalized_err(y, y_hat, err_method='MSD', norm='mean')
        else:
            nmbe = err_vals['NMBE']
        annot_str += r'$NMBE={0:.3f}$'.format(nmbe)

    if annot_str[-1:] == '\n':
        annot_str = annot_str[:-1]

    annotate_axes(
        ax, annotations=[annot_str], loc=err_loc, fontsize=fontsize, **err_kwds
    )


def annotate_axes(
    axes,
    fig=None,
    annotations=None,
    loc='top left',  # xy=[(.05, .95)],
    bbox=dict(boxstyle="round", fc="w", ec="k", alpha=0.8, pad=0.3),
    txt_offset=8,
    xy_offset=(0, 0),
    fontsize=9,
    **kwds
):
    """
    Create axes annotations like (a), (b), (c)...

    Parameters
    ----------
    axes : tuple, matplotlib.Axes
        Axes to print annotations to. If tuple, label each consequently.
    fig : matplotlib.Figure, optional
        Figure of which axis should be labeled. The default is None.
    annotations : list, tuple, string, optional
        Annotations to use. If None, the lower case alphabet will be used.
        The default is None.
    loc : string, optional
        Location in each axes to print annotations to. The default
        is 'top left'.
    bbox : dict, optional
        Boxstyle to use for annotations. The default is
        dict(boxstyle="round", fc="w", ec="k", alpha=.8, pad=0.35).
    txt_offset : int, float, optional
        Text offset from axes border in points. The default is 8.
    xy_offset : tuple, optional
        Additional xy-offset from axes border in points. The default is (0, 0).
    fontsize : int, optional
        Fontsize for annotating the axes. The default is 9.
    **kwds : keyword arguments
        Additional alignment arguments to pass on the matplotlib.annotate.

    Returns
    -------
    None.

    """
    if isinstance(fig, mpl.figure.Figure):
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
        xytext = (txt_offset + xy_offset[0], -txt_offset + xy_offset[1])
        ha, va = 'left', 'top'
    elif loc == 'top right':
        xy = (1, 1)
        xytext = (-txt_offset + xy_offset[0], -txt_offset + xy_offset[1])
        ha, va = 'right', 'top'
    elif loc == 'bottom left':
        xy = (0, 0)
        xytext = (txt_offset + xy_offset[0], txt_offset + xy_offset[1])
        ha, va = 'left', 'bottom'
    elif loc == 'bottom right':
        xy = (1, 0)
        xytext = (-txt_offset + xy_offset[0], txt_offset + xy_offset[1])
        ha, va = 'right', 'bottom'
    elif loc == 'top center':
        xy = (0.5, 1)
        xytext = (0 + xy_offset[0], -txt_offset + xy_offset[1])
        ha, va = 'center', 'top'
    elif loc == 'lower center':
        xy = (0.5, 0)
        xytext = (0 + xy_offset[0], txt_offset + xy_offset[1])
        ha, va = 'center', 'bottom'
    elif loc == 'center left':
        xy = (0, 0.5)
        xytext = (txt_offset + xy_offset[0], 0 + xy_offset[1])
        ha, va = 'left', 'center'
    elif loc == 'center right':
        xy = (1, 0.5)
        xytext = (-txt_offset + xy_offset[0], 0 + xy_offset[1])
        ha, va = 'right', 'center'
    # overwrite align if given:
    ha = kwds['ha'] if 'ha' in kwds else ha
    va = kwds['va'] if 'va' in kwds else va

    # iterate over axes:
    for i, ax in enumerate(axes):
        ax.annotate(
            text=annt[i],
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
