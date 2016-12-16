#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
from collections import namedtuple
from functools import partial
import re

import helpy


FitTuple = namedtuple('Fit', 'func TR DR lp DT v0 w0'.split())

def print_call(name, *args, **kwargs):
    s_args = map(repr, args)
    s_kwargs = map('{0[0]}={0[1]}'.format, kwargs.items())
    arglist = ', '.join(s_args + s_kwargs)
    print '{name}({arglist})'.format(name=name, arglist=arglist)

class Fit(FitTuple):
    """Fit"""

    def __init__(*args, **kwargs):
        print_call('__init__', *args, **kwargs)
        return FitTuple.__init__(*args, **kwargs)

    def __new__(*args, **kwargs):
        'Create new instance of Fit(func, TR, DR, lp, DT, v0, w0)'
        print_call('__new__', *args, **kwargs)
        return FitTuple.__new__(*args, **kwargs)

    def __getattribute__(*args, **kwargs):
        print_call('__getattribute__', *args, **kwargs)
        return FitTuple.__getattribute__(*args, **kwargs)



def make_fit(*dicts, **kwargs):
    """Generate a Fit instance from dicts and kwargs."""
    f = dict.fromkeys(Fit._fields)
    for d in dicts:
        f.update(d)
    f.update(kwargs)
    return Fit(**f)


def fit_dict(fit):
    """Convert a Fit instance to a plain dict."""
    if not hasattr(fit, '_asdict'):
        return fit
    return {k: fit_dict(v)
            for k, v in fit._asdict().iteritems() if v is not None}


def fit_items(fit):
    """Convert a Fit instance to a list of (k, v) pairs."""
    if not hasattr(fit, '_asdict'):
        return fit
    return [[k, fit_items(v)]
            for k, v in fit._asdict().iteritems() if v is not None]


def fit_str(fit, remove_none=False, indent=True):
    """convert a Fit instance into nicely formatted string"""
    tabs = []
    q = []
    fit = str(fit)
    if remove_none:
        fit = re.sub(', ..=None', '', fit)
    if not indent:
        return fit
    for s in fit.split(', '):
        q.append(' '*sum(tabs) + s)
        p = s.find('(') + 1
        if p:
            tabs.append(p)
        elif s.endswith(')'):
            tabs.pop(-1)
    return ',\n'.join(q)


def load_fits(prefix, new_fits=None):
    """load existing fits file, and update with new fits if given."""
    try:
        from ruamel import yaml     # prefer ruamel.yaml (updated PyYAML fork)
    except ImportError:
        import yaml                 # fall-back to PyYAML if unavailable

    try:
        path = helpy.with_suffix(prefix, '_FITS.yaml')
        with open(path, 'r') as f:
            all_fits = yaml.load(f, Loader=yaml.Loader)
    except IOError as ioe:
        if new_fits is None:
            raise ioe
        else:
            all_fits = {}

    return merge_fits(all_fits, new_fits)


def merge_fits(all_fits, new_fits):
    """merge new fits dict into existing dict of many fits."""
    if new_fits is None:
        return all_fits
    all_fits.update(new_fits)
    return all_fits


def save_fits(prefix, new_fits):
    """save dict of fits to yaml file.

    If file exists, load and merge them first
    """
    try:
        from ruamel import yaml     # prefer ruamel.yaml (updated PyYAML fork)
    except ImportError:
        import yaml                 # fall-back to PyYAML if unavailable
    path = helpy.with_suffix(prefix, '_FITS.yaml')
    fits = load_fits(path, new_fits)
    with open(path, 'w') as f:
        yaml.dump(fits, f)


def gather_fits(prefixes, key_by=('prefix', 'config', 'param')):
    """Load fits from several prefixes and merge them"""
    fits = helpy.dmap(load_fits, prefixes)
    if 'config' in key_by or 'param' in key_by:
        by_config = dict.fromkeys(f for fs in fits.values() for f in fs)
        for fit in by_config:
            by_config[fit] = {}
            fit_vars = fits[prefixes[0]][fit]
            for var in fit_vars:
                by_config[fit][var] = []
                for pre in prefixes:
                    by_config[fit][var].append(fits[pre][fit][var])
        fits['prefixes'] = prefixes
    if 'config' in key_by:
        fits.update(by_config)
    if 'param' in key_by:
        by_param = helpy.transpose_dict(by_config, missing=False)
        fits.update(by_param)
    if 'prefix' not in key_by:
        map(fits.pop, prefixes)
    return fits


def make_fitname(fit):
    """Attempt to create a nickname for a fit given a Fit instance"""
    # ('func', 'TR', 'DR', 'lp', 'DT', 'v0', 'w0')
    d = {'func':
         {'vo': 'vo', 'vn': 'vn', 'vt': 'vt',
          'nn': 'nn',
          'rn': 'rn', 'rp': 'rp', 'rs': 'rs', 'ra': 'ra', 'rm': 'rm',
          'rr': 'rr', 'r0': 'r0',
          'pr': 'pr', 'p0': 'p0', 'dr': 'dr', 'd0': 'd0',
         },
         'TR': {None    : '0',
                'free'  : 'f',
                1.8     : 'm',
               },
         'DR': {'free'  : 'f',

               },
        }

    fmt = '{func}_T{TR}_R{DR}_L{lp}_D{DT}'.format
    fmt(func=fit.func, TR=fit.TR or 0, DR=fit.DR, lp=fit.lp, DT=fit.DT)
    return


def make_fitnames():
    """Build up a mapping to nicknames for most conceivablel fits"""

    # from velocities
    mkf = helpy.make_fit
    cf = {'free': 'free', None: None}
    cf.update({
        'vn': mkf(func='vn', v0='mean', DT='var'),
        'vt': mkf(func='vt', DT='var'),
        'vo': mkf(func='vo', DR='var', w0='mean'),
    })

    # from orientation autocorrelation: nn
    mkf = partial(helpy.make_fit, func='nn', DR='free')
    cf.update({
        'nn_T0_Rf': mkf(TR=None),
        'nn_Tf_Rf': mkf(TR='free'),
        'nn_Tm_Rf': mkf(TR=1.8),
    })

    # from forward displacement: rn rp rs ra rm
    mkf = partial(helpy.make_fit, lp='free')
    for func, TR, DR in [('r'+r, 'T'+T, 'R'+R)
                         for r in 'npsam' for T in '0nm' for R in 'fn']:
        desc = '_'.join([func, TR, DR, 'Lf'])
        TRf = {'T0': None,
               'Tn': 'nn_Tf_Rf',
               'Tm': 'nn_Tm_Rf'}[TR]
        DRf = {'Rf': 'free',
               'Rn': TRf or 'nn_'+TR+'_Rf'}[DR]
        cf[desc] = mkf(func=func, TR=cf[TRf], DR=cf[DRf])

    # from mean squared displacement: rr pr dr r0 p0 d0
    mkf = partial(helpy.make_fit, DT='free')
    for func, TR, lp, DT in [(r+z, 'T'+T, 'L'+l, 'D'+D)
                             for r in 'rpd' for z in 'r0'
                             for T in '0nm' for l in 'fnard' for D in 'fd']:
        if func[0] in (lp[1], DT[1]):
            continue  # cannot source lp or DT from self
        desc = '_'.join([func, TR, 'Rn', lp, DT])
        TRf = {'T0': None,
               'Tn': 'nn_Tf_Rf',
               'Tm': 'nn_Tm_Rf'}[TR]
        DRf = TRf or 'nn_T0_Rf'
        lpf = {'Lf': 'free',
               'Ln': 'rn_'+TR+'_Rn_Lf',
               'La': 'ra_'+TR+'_Rn_Lf',
               'Lr': 'rn_'+TR+'_Rf_Lf',
               'Ld': d_desc}[lp]
        DTf = {'Df': 'free',
               'Dd': d_desc}[DT]
        cf[desc] = mkf(func=func, TR=cf[TRf], DR=cf[DRf],
                       lp=cf[lpf], DT=cf[DTf])

    del cf['free'], cf[None]
    return cf, {cf[k]: k for k in cf}
