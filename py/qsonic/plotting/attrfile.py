import numpy as np
import matplotlib.pyplot as plt
import fitsio
from scipy.interpolate import CubicSpline

from qsonic.utils import get_lambda, get_data_case_insensitive

DESI_WLIMITS = {'B': [3600., 5800.], 'R': [5760., 7620.], 'Z': [7520., 9824.]}


def plot_many_attrs_eta_varlss(
        listListAttrs, ncols=4, colsize=6, rowsize=5,
        markAmpRegions=True, plotEta=True, plotSpline=True
):
    mainAttr = listListAttrs[0]
    namplifiers = len(mainAttr)

    nrows = int(np.ceil(namplifiers / ncols))

    fig, axs = plt.subplots(
        nrows, ncols,
        sharex='all', sharey='none',
        gridspec_kw={'hspace': 0.12, 'wspace': 0.2},
        figsize=(colsize * ncols, rowsize * nrows)
    )

    noff_cols = ncols * nrows - namplifiers
    for jj in range(noff_cols):
        axs[-1, -1 - jj].set_axis_off()

    if plotEta:
        plotFnc = AttrFile.plot_eta
        scilimits = (-2, -2)
    else:
        plotFnc = AttrFile.plot_varlss
        scilimits = (0, 0)

    for iz in range(namplifiers):
        row = int(iz / ncols)
        col = iz % ncols
        ax = axs[row, col]
        plotFnc(mainAttr[iz], otherAttrs=[_[iz] for _ in listListAttrs[1:]],
                ax=ax, plotSpline=plotSpline)

        if col != 0:
            ax.set_ylabel("")

        if row != nrows - 1:
            ax.set_xlabel("")

        if markAmpRegions:
            b_line = np.sum(DESI_WLIMITS['B']) / 2.
            ax.axvline(b_line, ls='--', c='k')
            ax.axvspan(
                DESI_WLIMITS['R'][0], DESI_WLIMITS['B'][1],
                fc='k', alpha=0.5
            )

        ax.autoscale()
        ax.ticklabel_format(
            style='sci', axis='y', scilimits=scilimits, useMathText=True)


def add_minor_grid(ax):
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=1)


class AttrFile():
    @staticmethod
    def variance_function(var_pipe, var_lss, eta):
        return eta * var_pipe + var_lss

    @staticmethod
    def _parse_otherattrs(otherAttrs):
        if otherAttrs is None:
            otherAttrs = []
        elif isinstance(otherAttrs, AttrFile):
            otherAttrs = [otherAttrs]
        elif not isinstance(otherAttrs, list):
            raise Exception("otherAttrs must be a list or AttrFile.")
        return otherAttrs

    def __init__(self, fname, name="qsonic"):
        self.fname = fname
        self.name = name
        fattr = fitsio.FITS(fname)

        self.cont = fattr['CONT'].read()
        self.stacked_flux = fattr['STACKED_FLUX'].read()
        if 'VAR_FUNC-fit' in fattr:
            self.varfunc = fattr['VAR_FUNC-fit'].read()
        else:
            self.varfunc = fattr['VAR_FUNC'].read()

        hdr = dict(fattr['VAR_STATS'].read_header())
        varstats = fattr['VAR_STATS'].read().reshape(
            hdr['NWBINS'], hdr['NVARBINS'])
        hdr['data'] = varstats
        self.varstats = hdr

        fattr.close()
        self.marker = None

    def plot_varpipe_varobs(self, iwave, ax=None, plotfit=True):
        if ax is None:
            ax = plt.gca()

        data = self.varstats['data'][iwave]
        w2 = ((data['num_qso'] >= self.varstats['MINNQSO'])
              & (data['num_pixels'] >= self.varstats['MINNPIX']))
        ax.errorbar(
            data['var_pipe'][w2], data['var_delta'][w2],
            yerr=data['e_var_delta'][w2],
            fmt='.', alpha=1, label="Nominal range")

        # Plot extended range
        w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100) & (~w2)
        ax.errorbar(
            data['var_pipe'][w2], data['var_delta'][w2],
            yerr=data['e_var_delta'][w2],
            fmt='.', alpha=1, label="Extended range")

        # Plot var_lss, eta fitting function
        if plotfit:
            w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100)
            var_pipe = data['var_pipe'][w2]
            var_pipe = var_pipe[var_pipe > 0]
            var_pipe = np.logspace(
                np.log10(var_pipe.min()), np.log10(var_pipe.max()), 100)

            var_lss = get_data_case_insensitive(self.varfunc, 'VAR_LSS')[iwave]
            e_lss = get_data_case_insensitive(self.varfunc, 'E_VAR_LSS')[iwave]
            eta = get_data_case_insensitive(self.varfunc, 'ETA')[iwave]
            e_eta = get_data_case_insensitive(self.varfunc, 'E_ETA')[iwave]

            yfit = AttrFile.variance_function(var_pipe, var_lss, eta)
            ymin = AttrFile.variance_function(
                var_pipe, var_lss - e_lss, eta - e_eta)
            ymax = AttrFile.variance_function(
                var_pipe, var_lss + e_lss, eta + e_eta)

            ax.plot(var_pipe, yfit, 'k-', label="Fit")
            ax.fill_between(var_pipe, ymin, ymax, alpha=0.6, fc='grey')

        # Add texts
        meanl = np.mean(data['wave'])
        ax.text(
            0.06, 0.94, f"{meanl:.0f} A",
            transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            color="#9f2305",
            bbox={'facecolor': 'white', 'pad': 0.3, "ec": "#9f2305",
                  'alpha': 0.5, 'boxstyle': 'round', "lw": 2}
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Observed variance")

        add_minor_grid(ax)
        ax.legend(loc="lower right")

        return ax

    def plot_varpipe_meandelta(self, iwave, ax=None):
        if ax is None:
            ax = plt.gca()

        data = self.varstats['data'][iwave]
        w2 = ((data['num_qso'] >= self.varstats['MINNQSO'])
              & (data['num_pixels'] >= self.varstats['MINNPIX']))
        ax.errorbar(
            data['var_pipe'][w2], data['mean_delta'][w2],
            yerr=np.sqrt(data['var_delta'][w2] / data['num_pixels'][w2]),
            fmt='.', alpha=1, label="Nominal range")

        # Plot extended range
        w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100) & (~w2)
        ax.errorbar(
            data['var_pipe'][w2], data['mean_delta'][w2],
            yerr=np.sqrt(data['var_delta'][w2] / data['num_pixels'][w2]),
            fmt='.', alpha=1, label="Extended range")

        # Add texts
        meanl = np.mean(data['wave'])
        ax.text(
            0.06, 0.94, f"{meanl:.0f} A",
            transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            color="#9f2305",
            bbox={'facecolor': 'white', 'pad': 0.3, "ec": "#9f2305",
                  'alpha': 0.5, 'boxstyle': 'round', "lw": 2}
        )

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Mean delta")
        ax.axhline(0, c='k')

        add_minor_grid(ax)
        ax.legend(loc="lower right")
        return ax

    def plot_varpipe_varobs_ratio(
            self, iwave, ax=None, print_chi2=True, ylim=(0.9, 1.2)
    ):
        if ax is None:
            ax = plt.gca()

        var_lss = get_data_case_insensitive(self.varfunc, 'VAR_LSS')[iwave]
        e_var_lss = get_data_case_insensitive(self.varfunc, 'E_VAR_LSS')[iwave]
        eta = get_data_case_insensitive(self.varfunc, 'ETA')[iwave]
        e_eta = get_data_case_insensitive(self.varfunc, 'E_ETA')[iwave]
        data = self.varstats['data'][iwave]

        w2 = ((data['num_qso'] >= self.varstats['MINNQSO'])
              & (data['num_pixels'] >= self.varstats['MINNPIX']))
        var_pipe = data['var_pipe'][w2]
        var_obs = data['var_delta'][w2]
        e_var_obs = data['e_var_delta'][w2]
        yfit = AttrFile.variance_function(var_pipe, var_lss, eta)
        ax.errorbar(
            var_pipe, var_obs / yfit, yerr=e_var_obs / yfit,
            fmt='.', alpha=1, label="Nominal range")

        if print_chi2:
            d = var_obs - yfit
            chi2 = np.sum(d**2 / e_var_obs**2)
            print(f"Diagonal chi2 / dof: {chi2:.2f} / ({d.size} - 2)")
            if 'cov_var_delta' in data.dtype.names:
                invcov = np.linalg.inv(data['cov_var_delta'][w2, :][:, w2])
                chi2 = np.dot(d, invcov.dot(d))
                print(f"Invcov chi2 / dof: {chi2:.2f} / ({d.size} - 2)")

        # Plot extended range
        w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100) & (~w2)
        var_pipe = data['var_pipe'][w2]
        yfit = AttrFile.variance_function(var_pipe, var_lss, eta)
        ax.errorbar(
            var_pipe, data['var_delta'][w2] / yfit,
            yerr=data['e_var_delta'][w2] / yfit,
            fmt='.', alpha=1, label="Extended range")

        # Plot shaded tilt
        w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100)
        var_pipe = data['var_pipe'][w2]
        var_pipe = var_pipe[var_pipe > 0]
        var_pipe = np.logspace(
            np.log10(var_pipe.min()), np.log10(var_pipe.max()), 100)

        yfit = AttrFile.variance_function(var_pipe, var_lss, eta)
        ymin = AttrFile.variance_function(
            var_pipe, var_lss - e_var_lss, eta - e_eta) / yfit
        ymax = AttrFile.variance_function(
            var_pipe, var_lss + e_var_lss, eta + e_eta) / yfit
        ax.fill_between(var_pipe, ymin, ymax, alpha=0.6, fc='grey')

        # Add texts
        meanl = np.mean(data['wave'])
        ax.text(
            0.76, 0.94, f"{meanl:.0f} A",
            transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            color="#9f2305",
            bbox={'facecolor': 'white', 'pad': 0.3, "ec": "#9f2305",
                  'alpha': 0.5, 'boxstyle': 'round', "lw": 2}
        )

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Observed variance / Model")
        ax.axhline(1, c='k')

        add_minor_grid(ax)
        ax.legend(loc="lower right", ncol=2)

        if isinstance(ylim, tuple):
            ax.set_ylim(*ylim)

        return ax

    def plot_one_cov(self, iwave, cmap='RdBu_r'):
        data = self.varstats['data'][iwave]
        vp = data['var_pipe']
        w0 = vp > 0

        w2 = ((data['num_qso'] >= self.varstats['MINNQSO'])
              & (data['num_pixels'] >= self.varstats['MINNPIX']))
        if w2.sum() < 5:
            print("Used extended cuts due to low statistics")
            w2 = ((data['num_qso'] >= self.varstats['MINNQSO'] // 2)
                  & (data['num_pixels'] >= self.varstats['MINNPIX'] // 2))

        w2 = w2[w0]
        vp = vp[w0]
        a10 = 10**(np.log10(vp[1]) - np.log10(vp[0]))
        vp_edges = 2 * vp / (1 + a10)
        vp_edges = np.append(vp_edges, vp_edges[-1] * a10)

        e = np.sqrt(data['cov_var_delta'].diagonal())[w0]
        e = np.where(e == 0, 1, e)

        matrix = data['cov_var_delta'][w0, :][:, w0] / np.outer(e, e)
        matrix[np.isnan(matrix)] = 0

        fig, ax = plt.subplots()

        im = ax.pcolormesh(
            vp_edges, vp_edges, matrix,
            cmap=cmap, shading='flat', vmin=-1, vmax=1)

        m2 = np.zeros_like(matrix) * np.nan
        m2[~w2, :] = 1
        m2[:, ~w2] = 1
        plt.pcolormesh(
            vp_edges, vp_edges, m2, shading='flat',
            alpha=0.2, cmap='Greys_r')

        ax.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xticks(np.logspace(-4, 1, 6))
        ax.set_yticks(np.logspace(-4, 1, 6))
        ax.set_xlim(vp[vp > 0].min(), vp.max())
        ax.set_ylim(vp[vp > 0].min(), vp.max())
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Pipeline variance")

        cbar = fig.colorbar(im, ticks=np.linspace(-1, 1, 6))
        cbar.ax.tick_params()

        meanl = np.mean(data['wave'])
        ax.text(
            0.06, 0.94, f"{meanl:.0f} A",
            transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            color="#9f2305",
            bbox={'facecolor': 'white', 'pad': 0.3, "ec": "#9f2305",
                  'alpha': 0.9, 'boxstyle': 'round', "lw": 2}
        )
        return ax

    def plot_mean_cont(self, otherAttrs=None, ax=None):
        """
        Args:
            otherAttrs (list(AttrFile))
        """
        if ax is None:
            ax = plt.gca()

        other_attrs = AttrFile._parse_otherattrs(otherAttrs)
        fattr = [self] + other_attrs

        for f in fattr:
            data = f.cont
            ax.errorbar(
                data['lambda_rf'], data['mean_cont'], data['e_mean_cont'],
                label=f.name, capsize=0)

        ax.set_ylabel("Mean continuum")
        ax.set_xlabel("Rest-frame wavelength [A]")
        add_minor_grid(ax)
        ax.legend()
        return ax

    def plot_eta(
            self, otherAttrs=None, ax=None, plotSpline=False,
            markAmpRegions=True, colors=None, alphas=None
    ):
        if ax is None:
            ax = plt.gca()

        other_attrs = AttrFile._parse_otherattrs(otherAttrs)
        fattr = [self] + other_attrs

        if not isinstance(colors, list):
            colors = [plt.cm.tab10(i) for i in range(len(fattr))]

        if not isinstance(alphas, list):
            alphas = [1 for i in range(len(fattr))]

        if markAmpRegions:
            b_line = np.sum(DESI_WLIMITS['B']) / 2.
            ax.axvline(b_line, ls='--', c='k', label="Amplifier boundary in B")
            ax.axvspan(
                DESI_WLIMITS['R'][0], DESI_WLIMITS['B'][1],
                fc='k', alpha=0.6, label="Transition from B to R"
            )

        for i, f in enumerate(fattr):
            fmt = f.marker if f.marker else '.'
            if not plotSpline:
                fmt += '-'
            varfunc = f.varfunc
            wave = get_lambda(varfunc)
            eta = get_data_case_insensitive(varfunc, 'ETA')
            e_eta = get_data_case_insensitive(varfunc, 'E_ETA')
            ax.errorbar(
                wave, eta - 1, e_eta,
                fmt=fmt, label=f.name, c=colors[i], alpha=alphas[i]
            )

            if plotSpline:
                spl = CubicSpline(wave, eta)
                x = np.linspace(wave[0] - 60., wave[-1] + 60., 100)
                ax.plot(x, spl(x) - 1, '-', c=colors[i], alpha=alphas[i] * 0.7)

        add_minor_grid(ax)
        ax.set_ylabel(r"$\eta - 1$")
        ax.axhline(0, c='k')
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylim(-0.03, 0.02)
        ax.ticklabel_format(
            style='sci', axis='y', scilimits=(-2, -2), useMathText=True)
        ax.yaxis.get_offset_text()
        ax.legend()
        return ax

    def plot_varlss(
            self, otherAttrs=None, ax=None, plotSpline=False,
            markAmpRegions=False
    ):
        if ax is None:
            ax = plt.gca()

        other_attrs = AttrFile._parse_otherattrs(otherAttrs)
        fattr = [self] + other_attrs
        fmt = '.' if plotSpline else '.-'

        for i, f in enumerate(fattr):
            wave = get_lambda(f.varfunc)
            var_lss = get_data_case_insensitive(f.varfunc, 'VAR_LSS')
            e_var_lss = get_data_case_insensitive(f.varfunc, 'E_VAR_LSS')

            ax.errorbar(wave, var_lss, e_var_lss, fmt=fmt, label=f.name,
                        c=plt.cm.tab10(i))

        if plotSpline:
            for i, f in enumerate(fattr):
                wave = get_lambda(f.varfunc)
                var_lss = get_data_case_insensitive(f.varfunc, 'VAR_LSS')
                spl = CubicSpline(wave, var_lss)
                x = np.linspace(wave[0] - 60., wave[-1] + 60., 100)
                ax.plot(x, spl(x), '-', c=plt.cm.tab10(i), alpha=0.7)

        if markAmpRegions:
            b_line = np.sum(DESI_WLIMITS['B']) / 2.
            ax.axvline(b_line, ls='--', c='k')
            ax.axvspan(DESI_WLIMITS['R'][0], DESI_WLIMITS['B'][1],
                       fc='k', alpha=0.5)

        add_minor_grid(ax)
        ax.set_ylabel(r"$\sigma^2_\mathrm{LSS}$")
        ax.set_xlabel("Wavelength [A]")
        ax.legend()
        return ax

    def plot_stacked_flux(
            self, otherAttrs=None, ax=None, colors=None, alphas=None
    ):
        if ax is None:
            plt.figure(figsize=(12, 5))
            ax = plt.gca()

        other_attrs = AttrFile._parse_otherattrs(otherAttrs)
        fattr = [self] + other_attrs

        alpha = max(0.5, min(1, 1.6 / len(fattr)))
        minwave = 100000
        maxwave = -1

        if not isinstance(colors, list):
            colors = [plt.cm.tab10(i) for i in range(len(fattr))]

        if not isinstance(alphas, list):
            alphas = [alpha for i in range(len(fattr))]

        for i, f in enumerate(fattr):
            wave = get_lambda(f.stacked_flux)
            stack = get_data_case_insensitive(f.stacked_flux, 'STACKED_FLUX')
            minwave = min(minwave, wave.min())
            maxwave = max(maxwave, wave.max())
            ax.plot(wave, stack - 1, '-', label=f.name, alpha=alphas[i],
                    color=colors[i])

        ax.set_ylabel("Stacked flux - 1")
        ax.axhline(0, c='k')
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylim(-0.05, 0.05)

        xtickmin = int(np.round(minwave / 400) * 400)
        xtickmax = int(np.round(maxwave / 400) * 400)
        nticks = (xtickmax - xtickmin) // 400
        ax.set_xticks(np.linspace(xtickmin, xtickmax, nticks + 1))

        add_minor_grid(ax)

        ax.ticklabel_format(
            style='sci', axis='y', scilimits=(-2, -2), useMathText=True)
        ax.legend()
        return ax

    def plot_all_varpipe_varobs(self):
        for i in range(self.varstats['NWBINS']):
            self.plot_varpipe_varobs(i)
            plt.show()

    def plot_all_varpipe_varobs_ratio(self):
        for i in range(self.varstats['NWBINS']):
            self.plot_varpipe_varobs_ratio(i)
            plt.show()

    def plot_all_covariances(self):
        for i in range(self.varstats['NWBINS']):
            self.plot_one_cov(i)
            plt.show()

    def plot_all_varpipe_meandelta(self, inone=False):
        if not inone:
            for i in range(self.varstats['NWBINS']):
                self.plot_varpipe_meandelta(i)
                plt.show()
            return

        ax = plt.gca()
        for iwave in range(self.varstats['NWBINS']):
            data = self.varstats['data'][iwave]
            meanl = np.mean(data['wave'])

            w2 = (data['num_qso'] >= 10) & (data['num_pixels'] >= 100)
            ax.errorbar(
                data['var_pipe'][w2], data['mean_delta'][w2],
                yerr=np.sqrt(data['var_delta'][w2] / data['num_pixels'][w2]),
                fmt='.', alpha=1, label=f"{meanl:.0f}")

        ax.set_xscale("log")
        ax.set_yscale("linear")
        ax.set_xlabel("Pipeline variance")
        ax.set_ylabel("Mean delta")
        ax.axhline(0, c='k')

        add_minor_grid(ax)
        ax.legend(loc="lower right")
        return ax
