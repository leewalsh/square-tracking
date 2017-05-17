# coding: utf-8

from __future__ import division

import sys
from functools import partial
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

import helpy
import tracks


save_figs = False

prefixes = sys.argv[1:]
pres = tuple([p.replace('ree_lower_lid', '').replace('_50Hz_MRG', '')
              for p in prefixes])
print '\n'.join([p.replace('/Users/leewalsh', '~') for p in pres])
fits = helpy.gather_fits(prefixes)

fit_config, fit_desc = tracks.make_fitnames()

all_colors = 'rgkbmyc'


def markers(fit, default='x'):
    """Set marker for fit based on fitting function"""
    if isinstance(fit, basestring):
        d = {'nn': 'o',
             'rn': 'v', 'rp': '>', 'ra': '^', 'rm': '<',
             'rr': (4, 1, 0), 'r0': 's',
             'pr': (6, 1, 0), 'p0': 5,#(6, 0, 0),
             'dr': (7, 1, 0), 'd0': 4,#(7, 0, 0),
            }
        return d.get(fit[:2], default)
    elif hasattr(fit, 'func'):
        return markers(fit_desc.get(fit), default)
    elif isinstance(fit, list):
        return map(partial(markers, default=default), fit)
    else:
        msg = 'Unknown fit {} {}'.format
        raise TypeError(msg(type(fit), fit))

# marker can be a tuple: (`numsides`, `style`, `angle`)
#       where `style` is {0: polygon, 1: star, 2: asterisk}

markersize = 4.0    # default is 6.0
markersizes = defaultdict(lambda: markersize)
markersizes['D'] -= 1
markersizes['s'] -= 0.5

colorbrewer = {c: plt.cm.Set1(i) for i, c in enumerate('rbgmoycpk')}
colorunbrewer = {plt.cm.Set1(i): c for i, c in enumerate('rbgmoycpk')}

def colors(fit, default='k'):
    """Set color for fit based on free parameters"""
    if isinstance(fit, basestring):
        d = {'T0': 'r',
             'Tm': 'r', 'Tm_Rn': 'r', 'Tm_Rn_Ln': 'r',
                                      'Tm_Rn_La': 'r',
             'Tf': 'brown', 'Tn_Rn': 'brown',
                        'Tm_Rf': 'b', 'Tm_Rn_Lr': 'b',
                                      'Tm_Rn_Lb': 'b',
                                      'Tm_Rn_Lf': 'green',
             'oo': 'brown', #'oo_DR', 'oo_Ds', 'oo_Da'
             'vo_T0': 'orange', 'vo_Tv': 'r', #'vo_T0_Dt', 'vo_Tv_Dt'
            }

        mpl2 = {'b': 'b',       # blue -> blue
                'brown': 'c',   # brown(c)
                'r': 'r',       # red -> red
                'orange': 'o',  # cyan -> orange
                'green': 'g',   # green
                'k': 'k',       # black -> grey(k)
                'y': 'y',       # yellow -> yellow
                'pink': 'p',    # pink
                'purple': 'm',  # purple
               }
        return colorbrewer[mpl2[d.get(fit[3:-3], default)]]
    elif hasattr(fit, 'func'):
        return colors(fit_desc.get(fit), default)
    elif isinstance(fit, list):
        return map(partial(colors, default=default), fit)
    else:
        msg = 'Unknown fit {} {}'.format
        raise TypeError(msg(type(fit), fit))


def labels(fit, default=None, fancy=True):
    """Set label for fit"""
    if isinstance(fit, basestring):
        dot = r'\langle {} \cdot {} \rangle'.format
        ms = r'\langle \left[ {} \right]^2 \rangle'.format
        free = r'{}^\mathrm{{ free}}'.format
        f = {'nn': dot('n(t)', 'n(0)'),
             'rn': dot('r(t)', 'n(0)'),
             'rp': dot('r(t>0)', 'n(0)'),
             'ra': dot('r(t)', 'n(0)') + '-' + dot('r(-t)', 'n(0)'),
             'rm': dot('r(t)', 'n'),
             'rr': ms('r(t) - r(0)'),
             'pr': ms(dot('r(t) - r(0)', r'\hat n(0)')),
             'dr': ms(dot('r(t) - r(0)', r'\hat t(0)')),
            }
        f.update(
            {'T0': r'\tau = 0',
             'Tm': r'\tau = \langle \tau \rangle',
             'Tn': r'\tau\left({}\right)'.format(f['nn']),
             'Rn': r'D_R\left({}\right)'.format(f['nn']),
             'Ln': r'\ell_p\left({}\right)'.format(f['rn']),
             'La': r'\ell_p\left({}\right)'.format(f['ra']),
             'Lr': r'\ell_p\left({}\right)$, ${}'.format(f['rn'], free(r'D_R')),
             'Tf': free(r'\tau'),
             'Rf': free(r'D_R'),
             'Lf': free(r'\ell_p'),
             'Df': free(r'D_T'),
             'r0': f['rr'],
             'p0': f['pr'],
             'd0': f['dr'],
            })
        ks = fit.split('_')
        if fancy:
            return '$' + '$, $'.join(f[k] for k in ks) + '$'
        else:
            return ' '.join(ks)
    elif hasattr(fit, 'func'):
        return labels(fit_desc.get(fit), default, fancy)
    elif isinstance(fit, list):
        return map(partial(labels, default=default, fancy=fancy), fit)
    else:
        msg = 'Unknown fit {} {}'.format
        raise TypeError(msg(type(fit), fit))


pargs = {}
scope = 'good'

pargs['DR'] = dict(
    param='DR',
    title='$D_R$',
    convert=None,
    xs='vo_Tv_Dt',
    ys={
        'good': [
            'nn_Tm_Rf',
        ],
        'maybe': [
            'nn_Tf_Rf',
            'rn_Tm_Rf_Lf',
            'ra_Tm_Rf_Lf',
        ],
        'bad': [
            'rn_Tn_Rf_Lf',
            'rp_Tn_Rf_Lf',
            'rp_Tm_Rf_Lf',
            'ra_Tn_Rf_Lf',
        ],
    }[scope],
    lims=(0, 0.14),
    legend={'loc':'lower right'},
)

pargs['DR_white'] = pargs['DR'].copy()
pargs['DR_white'].update(
    xs='vo_T0_Dt',
    ys=['nn_T0_Rf'],
    markerfacecolor='white',
)

for p in ('oo_DR', 'oo_Ds', 'oo_Da'):
    pargs['DR_'+p] = pargs['DR'].copy()
    pargs['DR_'+p]['xs'] = p
    pargs['DR_'+p]['color'] = colors('DR_'+p)

pargs['lp'] = dict(
    param='lp',
    title='$\\ell_p$',
    convert='y',
    xs='vn',
    ys={
        'good': [
            'r0_Tm_Rn_Lf_Df',
            'ra_Tm_Rn_Lf',
        ],
        'maybe': [
            'ra_Tm_Rf_Lf',
        ],
        'bad': [
            'ra_Tn_Rf_Lf',
            'ra_Tn_Rn_Lf',
            'rm_Tm_Rn_Lf',
            'rn_Tm_Rf_Lf',
            'rn_Tm_Rn_Lf',
            'rn_Tn_Rf_Lf',
            'rn_Tn_Rn_Lf',
            'rp_Tm_Rf_Lf',
            'rp_Tm_Rn_Lf',
            'rp_Tn_Rf_Lf',
            'rp_Tn_Rn_Lf',
            'rr_Tm_Rn_Lf_Df',
            'rr_Tn_Rn_Lf_Df',
        ],
    }[scope],
    lims=(0, 4.5),
    legend={'loc':'lower right'},
)


p = 'lp_msdvec'
pargs[p] = pargs[p[:2]].copy()
pargs[p]['ys'] = [
    'rr_Tn_Rn_Lf_Df',
    'dr_Tm_Rn_Lf_Df',
    'pr_Tm_Rn_Lf_Df',
    'r0_Tm_Rn_Lf_Df',
    'rr_Tm_Rn_Lf_Df',
    'p0_Tm_Rn_Lf_Df',
    'd0_Tm_Rn_Lf_Df',
]
pargs[p]['label'] = [l.replace('_', ' ') for l in pargs[p]['ys']]


pargs['v0'] = dict(
    param='v0',
    title='$v_0$',
    convert='y',
    xs='vn',
    ys={
        'good':[
            'r0_Tm_Rn_Lf_Df',
            'ra_Tm_Rf_Lf',
            'ra_Tm_Rn_Lf',
        ],
        'maybe': [
            'rr_Tm_Rn_Lf_Df',
        ],
        'bad': [
            'ra_Tn_Rf_Lf',
            'ra_Tn_Rn_Lf',
            'rm_Tm_Rn_Lf',
            'rn_Tm_Rf_Lf',
            'rn_Tm_Rn_Lf',
            'rn_Tn_Rf_Lf',
            'rn_Tn_Rn_Lf',
            'rp_Tm_Rf_Lf',
            'rp_Tm_Rn_Lf',
            'rp_Tn_Rf_Lf',
            'rp_Tn_Rn_Lf',
            'rr_Tn_Rn_Lf_Df',
        ],
    }[scope],
    lims=(0, 0.25),
    legend={'loc':'lower right'},
)


pargs['DT'] = dict(
    param='DT',
    title='$D_T$',
    convert=None,
    xs='vt',
    ys={
        'good': [
            'r0_Tm_Rn_Lf_Df',
            'r0_Tm_Rn_La_Df',
        ],
        'maybe':[
            'r0_Tm_Rn_Ln_Df',
            'r0_Tm_Rn_Lb_Df',
            'r0_Tm_Rn_Lq_Df',
        ],
        'bad': [
            'r0_Tm_Rn_Lr_Df',
            'rr_Tm_Rn_Lf_Df',
            'rr_Tm_Rn_Ln_Df',
            'rr_Tm_Rn_Lr_Df',
            'rr_Tn_Rn_Lf_Df',
            'rr_Tn_Rn_Ln_Df',
        ],
    }[scope],
    lims=(0, .013),
)


p = 'DT_msdvec'
pargs[p] = pargs[p[:2]].copy()
pargs[p]['ys'] = [
    'r0_Tm_Rn_Lf_Df',
    'r0_Tm_Rn_Ln_Df',
    'r0_Tm_Rn_La_Df',
    #'r0_Tm_Rn_Lq_Df',
    #'r0_Tm_Rn_Lb_Df',

    'rr_Tm_Rn_Lf_Df',
    'rr_Tm_Rn_Ln_Df',
    'rr_Tm_Rn_La_Df',
    #'rr_Tm_Rn_Lq_Df',
    #'rr_Tm_Rn_Lb_Df',

    'p0_Tm_Rn_La_Df',
    'p0_Tm_Rn_Lf_Df',
    'p0_Tm_Rn_Ln_Df',
    'pr_Tm_Rn_La_Df',
    'pr_Tm_Rn_Lf_Df',
    'pr_Tm_Rn_Ln_Df',
    'd0_Tm_Rn_La_Df',
    'd0_Tm_Rn_Lf_Df',
    'd0_Tm_Rn_Ln_Df',
    'dr_Tm_Rn_La_Df',
    'dr_Tm_Rn_Lf_Df',
    'dr_Tm_Rn_Ln_Df',
]
pargs[p]['label'] = [l.replace('_', ' ') for l in pargs[p]['ys']]

for p in pargs:
    ys = pargs[p]['ys']
    new = {
        'figsize': (5, 5),
        'color': colors(ys),
        'marker': markers(ys),
        'label': labels(ys, fancy=False),
        'linestyle': '',
    }
    new['markersize'] = [markersizes[m] for m in new['marker']]
    for k in new:
        pargs[p].setdefault(k, new[k])


param_xs = {
    'DR': ['DR',
           'DR_white',
           #'DR_oo_DR',# 'DR_oo_Ds', 'DR_oo_Da',
           ],
    'lp': ['lp'],
    'DT': ['DT'],
}

def trendline(xvals, yvals):
    slope = np.mean(yvals/xvals)
    print 'Slope: {:.4f}'.format(slope)
    xmax = max(yvals.max()/slope, xvals.max())
    xmin = min(yvals.min()/slope, xvals.min())
    xmax += xmin/4
    xmin *= 0.75
    return [xmin, xmax], [xmin*slope, xmax*slope]



def plot_param(fits, param, fitx, fity, convert=None, ax=None,
               tag=None, figsize=(4, 4), **kws):
    if ax is None:
        ax = plt.subplots(figsize=figsize)[1]
    resx, resy = fits[fitx], fits[fity]
    try:
        valx, valy = resx[param], resy[param]
    except KeyError as err:
        fitc = {'x': fitx, 'y': fity,
                'v': helpy.make_fit(func='vo', DR='var', w0='mean'),
                'a': helpy.make_fit(func='oo', TR='vac')}
        fitc = fitc.get(convert, convert)
        if param in ['v0', 'lp']:
            if param not in resx:
                print 'Convert: {} = {}.v0 / ({}.DR = {})'.format(
                    param, fit_desc[fitx], fit_desc[fitc],
                    fit_desc.get(fitc.DR, fitc.DR))
            elif param not in resy:
                print 'Convert: {} = {}.lp * ({}.DR = {})'.format(
                    param, fit_desc[fity], fit_desc[fitc],
                    fit_desc.get(fitc.DR, fitc.DR))
            else:
                print 'um Converting using D_R from', convert, fit_desc[fitc]
            if fitc is None:
                raise err
            #label += ' {}DR({}, {})'.format(
                #'lp(x)=v0(x)/' if param == 'lp' else 'v0(y)=lp(y)*',
                #fitc.func, fit_desc.get(fitc.DR, fitc.DR)
            DR = np.array(fits[fitc].get('DR') or fits[fitc.DR]['DR'])
            valx = resx.get(param) or resx['v0']/DR
            valy = resy.get(param) or resy['lp']*DR
        elif param in ['DR', 'var']:
            tau = np.array(fits[fitc]['TR'])
            valx = resx.get(param) or resx['var'] * tau
            valy = resy.get(param) or resy['DR'] / tau
    valx, valy = np.array(valx), np.array(valy)
    ax.plot(valx, valy, **kws)
    ax.plot(*trendline(valx, valy), color=kws['color'], linewidth=0.5, zorder=1)
    if tag:
        tag = [t.replace('ree_lower_lid', '').replace('_50Hz_MRG', '')
               for t in tag]
        tracks.plt_text(valx, valy, tag, fontsize='x-small')
    return ax


def plot_parametric(fits, param, xs, ys, scale='linear', lims=None,
                    ax=None, legend=None, savename='', title='', **kwargs):
    kwargs.update(xs=xs, ys=ys)
    kws = helpy.transpose_dict_of_lists(kwargs)
    for kw in kws:
        print '\n{}: {colcode}, {marker}'.format(
            param, colcode=colorunbrewer[kw['color']], **kw)
        x, y = kw.pop('xs'), kw.pop('ys')
        fitx, fity = fit_config[x], fit_config[y]
        print "y-axis:", y
        print helpy.fit_str(fity, remove_none=True)
        print "x-axis:", x
        print helpy.fit_str(fitx, remove_none=True)
        ax = plot_param(fits, param, fitx, fity, ax=ax, **kw)

    if scale == 'log' and lims[0] < 1e-3:
        lims[0] = 1e-3
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lims)
    ticks = ax.get_xticks() # must get_xticks after xlim
    ax.set_xticks(ticks)    # to prevent auto-changing later
    ax.set_yticks(ticks)
    ax.set_xlim(lims)       # must set_xlims again to fix
    ax.set_ylim(lims)
    ax.tick_params(direction='in', which='both')
    ax.plot(lims, lims, color=colorbrewer['k'], linewidth=0.5, zorder=1)
    if legend is False:
        if title:
            pad = 0.07
            ax.annotate(s=title, xy=(1-pad, pad), xycoords='axes fraction',
                        fontsize='x-large', ha='right', va='baseline')
    else:
        ax.legend(**dict(dict(loc='best', scatterpoints=1), **(legend or {})))
        if title:
            ax.set_title(title)
    if savename:
        ax.figure.savefig('~/Squares/colson/Output/stats/parameters/'
                          'parametric_{}.pdf'.format(savename))
    return ax

rcParams_for_context = {'text.usetex': True}
with plt.rc_context(rc=rcParams_for_context):
    fig, axes = plt.subplots(ncols=3, figsize=(7, 3))

    overrides = {'legend': False}

    params = ['DR', 'lp', 'DT']
    for p, ax in zip(params, axes):
        for px in param_xs[p]:
            kwargs = dict(pargs[px], **overrides)
            plot_parametric(fits, ax=ax, **kwargs)

    axes[0].set_ylabel('from correlation functions')
    axes[1].set_xlabel('from noise statistics')
    fig.tight_layout(w_pad=0.8)
    if save_figs:
        fig.savefig('parametric.pdf')

    # individual figures:
    #params = ['DR', 'lp', 'v0', 'DT', 'lp_msdvec', 'DT_msdvec']
    #for param in params:
    #    parg = pargs[param]
    #    ax = plot_parametric(fits, **parg)
    #    ax.set_xlabel('Noise statistics from velocity')
    #    ax.set_ylabel('Fits from correlation functions')

    #    if save_figs:
    #        ax.figure.savefig('parameters/parametric_{}.pdf'.format(param))

    plt.show()
