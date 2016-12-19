
# coding: utf-8

from __future__ import division

import sys
from functools import partial

from matplotlib import pyplot as plt

import helpy
import tracks


large = {'figure': {'figsize': (10, 12)},
         'legend': {'fontsize': 'small'},
         'font':   {'size': 12},
         'text':   {},
        }
small = {'figure': {'figsize': (5, 6)},
         'legend': {'fontsize': 'small',
                    'framealpha': 0.5},
         'font':   {'size': 10},
         'text':   {},
        }
default = {'figure': {'dpi': 120},
           'legend': {'loc': 'best'},
           'font':   {},
           'text':   {'usetex': True},
          }
rc = small
for grp in rc:
    rc[grp].update(default[grp])
    plt.rc(grp, **rc[grp])


save_figs = False


prefixes = sys.argv[1:]
print 'prefixes', prefixes
pres = tuple([p.replace('ree_lower_lid', '').replace('_50Hz_MRG', '')
              for p in prefixes])
fits = helpy.gather_fits(prefixes)


fit_config, fit_desc = tracks.make_fitnames()


all_colors = 'rgkbmyc'

def markers(fit, default='x'):
    if isinstance(fit, basestring):
        d = {'nn': 'o',
             'rn': 's', 'rp': '>', 'ra': 'D', 'rm': '^',
             'rr': (5, 1, 0), 'r0': (5, 0, 0),
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


def colors(fit, default='k'):
    if isinstance(fit, basestring):
        d = {'T0': 'k',
             'Tm': 'r', 'Tm_Rn': 'r', 'Tm_Rn_Ln': 'r',
             'Tf': 'g', 'Tn_Rn': 'g',
                        'Tm_Rf': 'b', 'Tm_Rn_Lr': 'b',
                                      'Tm_Rn_Lf': 'm',
            }

        o = {
            'Tn_Rn': 'b',
            'Tm': 'r', 'Tm_Rn': 'c',
             'Tn_Rf': 'm',
             'Tm_Rn_Lf': 'm', 'Tm_Rn_Ln': 'b', 'Tm_Rn_Lr':'c',
            }
        return d.get(fit[3:-3], default)
    elif hasattr(fit, 'func'):
        return colors(fit_desc.get(fit), default)
    elif isinstance(fit, list):
        return map(partial(colors, default=default), fit)
    else:
        msg = 'Unknown fit {} {}'.format
        raise TypeError(msg(type(fit), fit))


def labels(fit, default=None):
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
        return '$' + '$, $'.join(f[k] for k in ks) + '$'
    elif hasattr(fit, 'func'):
        return labels(fit_desc.get(fit), default)
    elif isinstance(fit, list):
        return map(partial(labels, default=default), fit)
    else:
        msg = 'Unknown fit {} {}'.format
        raise TypeError(msg(type(fit), fit))


pargs = {}
scope = 'good'


pargs['DR'] = dict(
    param='DR',
    title='$D_R$',
    convert=None,
    xs='vo',
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
            'nn_T0_Rf',
            'rn_Tn_Rf_Lf',
            'rp_Tn_Rf_Lf',
            'rp_Tm_Rf_Lf',
            'ra_Tn_Rf_Lf',
        ],
       }[scope],
    lims=(0, 0.13),
    legend={'loc':'lower right'},
)


pargs['lp'] = dict(
    param='lp',
    title='$\\ell_p$',
    convert='y',
    xs='vn',
    ys={
        'good': [
            'r0_Tm_Rn_Lf_Df',
            'ra_Tm_Rf_Lf',
            'ra_Tm_Rn_Lf',
        ],
        'maybe': [
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
    lims=(0, 4.0),
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
            'r0_Tm_Rn_Ln_Df',
        ],
        'maybe':[
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
    lims=(0, .025),
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
        'c': colors(ys),
        'marker': markers(ys),
        'label': labels(ys),
    }
    for k in new:
        pargs[p].setdefault(k, new[k])


fig, axes = plt.subplots(ncols=3, figsize=(14, 5))
params = ['DR', 'lp', 'DT']
for p, ax in zip(params, axes):
    tracks.plot_parametric(fits, ax=ax, **pargs[p])
if save_figs:
    fig.savefig('parameters/parametric_all.pdf')


param = 'DR'
parg = pargs[param]

ax = tracks.plot_parametric(fits, **parg)

if save_figs:
    ax.figure.savefig('parameters/parametric_{}.pdf'.format(param))


param = 'lp'
parg = pargs[param]

ax = tracks.plot_parametric(fits, **parg)

if save_figs:
    ax.figure.savefig('parameters/parametric_{}.pdf'.format(param))


param = 'v0'
parg = pargs[param]

ax = tracks.plot_parametric(fits, **parg)

if save_figs:
    ax.figure.savefig('parameters/parametric_{}.pdf'.format(param))


param = 'DT'
parg = pargs[param]

ax = tracks.plot_parametric(fits, **parg)

if save_figs:
    ax.figure.savefig('parameters/parametric_{}.pdf'.format(param))


pargs['lp_msdvec']['ys']


param = 'lp_msdvec'
parg = pargs[param]

ax = tracks.plot_parametric(fits, **parg)

if save_figs:
    ax.figure.savefig('parameters/parametric_{}.pdf'.format(param))


param = 'DT_msdvec'
parg = pargs[param].copy()

parg['figsize'] = (6, 6)
parg['legend'] = {'loc': 'upper right'}

ax = tracks.plot_parametric(fits, **parg)

ax.set_xlim(0, 0.025)
ax.set_ylim(0, 0.025)
#ax.legend(loc='lower right')
if save_figs:
    ax.figure.savefig('parameters/parametric_{}.pdf'.format(param))

plt.show()
