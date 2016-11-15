#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import itertools as it
from math import log, sqrt
import sys
import os
import ntpath
from string import Formatter
import readline     # this is used by raw_input
import glob
import platform
import getpass
from time import strftime

import numpy as np

SYSTEM, HOST, USER = None, None, None
COMMIT = None


def replace_all(s, old, new=''):
    return reduce(lambda a, b: a.replace(b, new), old, s)


def getsystem():
    global SYSTEM
    if not SYSTEM:
        SYSTEM = platform.system()
    return SYSTEM


def gethost():
    getsystem()
    global HOST
    if not HOST:
        HOST = platform.node()
        if SYSTEM == 'Darwin':
            from subprocess import check_output, STDOUT, CalledProcessError
            try:
                HOST = check_output(('scutil', '--get', 'ComputerName'),
                                    stderr=STDOUT).strip()
            except CalledProcessError:
                pass
        bad_chars = """('",’)"""
        HOST = replace_all(HOST.partition('.')[0], bad_chars).replace(' ', '-')
    return HOST


def getuser():
    global USER
    if not USER:
        USER = getpass.getuser().replace(' ', '_')
    return USER


def getcommit():
    global COMMIT
    if not COMMIT:
        gitdir = os.path.dirname(__file__) or os.curdir
        if not os.path.exists(os.path.join(gitdir, '.git')):
            COMMIT = 'unknown'
            return COMMIT
        try:
            import git
            repo = git.Repo(gitdir)
            dirty = repo.is_dirty()
            commit = repo.commit().hexsha[:7]
            if repo.head.is_detached:
                branch = 'detached'
            else:
                branch = repo.active_branch.name
        except ImportError:
            from subprocess import check_output, STDOUT, CalledProcessError
            git = lambda cmd: check_output(('git', '-C', gitdir) + cmd,
                                           stderr=STDOUT).strip()
            status = ('status', '--short')
            commit = ('log', '-1', '--pretty=tformat:%h')
            branch = ('branch', '--contains', '@')
            try:
                dirty = bool(git(status))
                commit = git(commit)
                branch = git(branch).partition('*')[2].split()[0]
            except CalledProcessError:
                COMMIT = 'unknown'
                return COMMIT
        except git.GitCommandNotFound:
            COMMIT = 'unknown'
            return COMMIT
        COMMIT = '{}({}{})'.format(commit, branch, '+'*dirty)
    return COMMIT


def splitter(data, indices, method=None,
             ret_dict=False, noncontiguous=False):
    """Splits a dataset into subarrays with unique index value

    parameters:
    data:       dataset to be split along first axis
    indices:    values to group by, can be array of indices or field name.
                default is field name 'f'
    method:     either 'diff' or 'unique':
                diff is faster*, but
                unique returns the `index` value
    ret_dict:   whether to return a dict with indices as keys
    noncontiguous:  whether to assume indices array has matching values
                    separated by other values

    returns:
        a list or dict of subarrays of `data` split and grouped by unique values
        of `indices`:
        if `method` is 'unique',
            returns list of tuples of (index, section)
        if `ret_dict` is True,
            returns a dict of the form {index: section}
        *if method is 'diff',
            `indices` is assumed sorted and not missing any values

    examples:

        for f, fdata in splitter(data, 'f', method='unique')

        for fdata in splitter(data, 'f')

        fsets = splitter(data, 'f', method='unique', ret_dict=True)
        fset = fsets[f]

        for trackid, trackset in splitter(data, 't', noncontiguous=True)

        tracksets = splitter(data, 't', noncontiguous=True, ret_dict=True)
        trackset = tracksets[trackid]
    """
    multi = isinstance(data, tuple)
    if not multi:
        data = (data,)
    if isinstance(indices, basestring):
        indices = data[0][indices]
    if method is None:
        method = 'unique' if ret_dict or noncontiguous else 'diff'
    if method.lower().startswith('d'):
        di = np.diff(indices)
        if di.min() < 0:
            print "Warning: nonincreasing index, switching to method = 'unique'"
            method = 'unique'
        di = di.nonzero()[0] + 1
        sects = [np.split(datum, di) for datum in data]
        ret = [dict(enumerate(sect)) for sect in sects] if ret_dict else sects
    if method.lower().startswith('u'):
        u, i = np.unique(indices, return_index=True)
        if noncontiguous:
            # no nicer way to do this:
            uinds = [np.where(indices == ui) for ui in u]
            sects = [[datum[uind] for uind in uinds] for datum in data]
        else:
            sects = [np.split(datum, i[1:]) for datum in data]
        ret = [dict(it.izip(u, sect)) if ret_dict else it.izip(u, sect)
               for sect in sects]
    return ret if multi else ret[0]


def is_sorted(a):
    return np.all(np.diff(a) >= 0)


def groupby(arr, key, method, min_size=1):
    """method = 'counter,filterer,getter'
    """
    if method == 'bin,bool,bool':
        cs = np.bincount(key+1)
        us = np.where(cs >= min_size)[0] - 1
        return {u: arr[key == u] for u in us}
    elif method == 'uniq,bool,bool':
        us, cs = np.unique(key, return_counts=True)
        us = us[cs >= min_size]
        return {u: arr[key == u] for u in us}
    elif method == 'uniq,filt,bool':
        us, cs = np.unique(key, return_counts=True)
        return {u: arr[key == u] for u, c in it.izip(us, cs) if c >= min_size}
    elif method == 'bin,bool,sort':
        cs = np.bincount(key+1)
        us = np.where(cs >= min_size)[0] - 1
        sort = key.argsort(kind='mergesort')
        inds = np.split(sort, np.cumsum(cs[:-1]))[us+1]
        return {u: arr[i] for i, u in it.izip(inds, us)}
    elif method == 'uniq,bool,sort':
        us, cs = np.unique(key, return_counts=True)
        cbool = cs >= min_size
        us = us[cbool]
        sort = key.argsort(kind='mergesort')
        inds = np.split(sort, np.cumsum(cs[:-1]))[cbool]
        return {u: arr[i] for i, u in it.izip(inds, us)}
    elif method == 'uniq,where,sort':
        us, cs = np.unique(key, return_counts=True)
        where = np.where(cs >= min_size)
        us = us[where]
        sort = key.argsort(kind='mergesort')
        inds = np.split(sort, np.cumsum(cs[:-1]))[where]
        return {u: arr[i] for i, u in it.izip(inds, us)}
    elif method == 'uniq,filt,sort':
        us, cs = np.unique(key, return_counts=True)
        sort = key.argsort(kind='mergesort')
        inds = np.split(sort, np.cumsum(cs[:-1]))
        return {u: arr[i]
                for u, i, c in it.izip(us, inds, cs) if c >= min_size}
    elif method == 'uniq,comp,sort':
        us, cs = np.unique(key, return_counts=True)
        sort = key.argsort(kind='mergesort')
        inds = np.split(sort, np.cumsum(cs[:-1]))
        return {u: arr[i]
                for u, i in it.compress(it.izip(us, inds), cs >= min_size)}
    elif method == 'test':
        methods = ['bin,bool,bool', 'uniq,bool,bool', 'uniq,filt,bool',
                   # 'bin,bool,sort', 'uniq,bool,sort', 'uniq,where,sort',
                   'uniq,filt,sort', 'uniq,comp,sort']
        print 'grouping'
        groups = [groupby(arr, key, method=m, min_size=min_size)
                  for m in methods]
        print 'checking'
        keys = [sorted(group.keys()) for group in groups]
        for ks in it.izip(*keys):
            for i in xrange(len(methods) - 1):
                m, n = methods[i:i+2]
                k, l = ks[i:i+2]
                g, h = groups[i:i+2]
                msg = 'unique key mismatch: {}: {}, {}: {}'.format
                assert k == l, msg(m, k, n, l)
                msg = 'mismatch at {} between {} and {}'.format
                assert np.allclose(g[l], h[l]), msg(l, m, n)
        print 'Success!'
        return


def pad_uneven(lol, fill=0, return_mask=False, dtype=None,
               align=None, longest=None, shortest=None):
    """ take uneven list of lists
        return new 2d array with shorter lists padded with fill value
    """
    if hasattr(lol, 'shape'):
        origshape = lol.shape
        lol = lol.flatten()
    else:
        origshape = (len(lol),)
    if dtype is None:
        dtype = np.result_type(fill, np.array(lol[0][0]))
    lengths = np.array(map(len, lol), int)
    lengths[lengths < shortest] = 0
    length = min(longest or np.inf, lengths.max())
    shape = (len(lol), length) + np.shape(lol[0][0])
    if align is None:
        align = it.repeat(0)
    else:
        align = np.asarray(align, int).flatten()
        align = align.max() - align
    result = np.empty(shape, dtype)
    result[:] = fill  # this allows broadcasting unlike np.full
    if return_mask:
        mask = np.ones(shape, bool)
    for i, (lst, l, k) in enumerate(it.izip(lol, lengths, align)):
        result[i, k:k+l] = lst[:l and longest]
        if return_mask:
            mask[i, k:k+l] = False
    result = result.reshape(origshape + shape[1:])
    return (result, mask) if return_mask else result


def avg_uneven(arrs, min_added=3, weight=False, pad=None, align=None,
               ret_all=False):
    if pad is None:
        pad = np.any(np.diff(map(len, arrs)))
    if pad:
        # could return_mask=True and isfin = ~mask, but that misses input nans
        arrs = pad_uneven(arrs, np.nan, return_mask=False, align=align)
    isfin = np.isfinite(arrs)
    added = isfin.sum(0)
    enough = (added >= min_added).all(tuple(range(1, added.ndim))).nonzero()[0]
    arrs = arrs[:, enough]
    added = added[enough]
    if weight:
        lens = np.sum(isfin, 1, keepdims=True)
        weights = np.where(lens, np.sqrt(lens), np.nan)  # replace 0 with np.nan
        mean = np.nanmean(arrs*weights, 0)/np.nanmean(weights)
        stddev = np.nanstd(arrs*weights, 0, ddof=1)/sqrt(np.nanmean(weights**2))
    else:
        mean = np.nanmean(arrs, 0)
        stddev = np.nanstd(arrs, 0, ddof=1)
    stderr = stddev / np.sqrt(added)
    ret = (arrs, mean, stderr)
    if ret_all:
        ret += stddev, added, enough
    return ret


def nan_info(arr, verbose=False):
    isnan = np.isnan(arr)
    nnan = np.count_nonzero(isnan)
    if nnan:
        wnan = np.where(isnan)[0]
        if verbose:
            print nnan, 'nans at', wnan[:10],
            if nnan > 10:
                print '...', wnan[:10][-10:],
            print 'of', len(arr)
    else:
        wnan = []
        if verbose:
            print 'no nans'
    return isnan, nnan, wnan


def transpose_dict(outerdict={}, **innerdicts):
    """transpose a dict of dicts.

    Input is a dict of dicts and/or several dicts as keyword args.  Keys present
    in some innerdicts but missing from another innerdict are kept as value None

    return value is basically
        {inner_key: {outer_key: inner_dict[inner_key]
                     for outer_key, inner_dict in outer_dict.items()}
         for inner_key in set(inner_dict.keys for inner_dict in outer_dict)}
    """
    outerdict = dict(outerdict)
    outerdict.update(innerdicts)
    innerdicts = outerdict.viewvalues()
    innerkeys = {innerkey for innerdict in innerdicts for innerkey in innerdict}
    return {ki: {ko: di.get(ki) for ko, di in outerdict.iteritems()}
            for ki in innerkeys}


def dmap(f, d):
    """map function f to d returning a dict

    if d is a dict:
        apply f to values in d
        returns {k: f(v)}
    else:
        return new dict values mapped to functions output
        returns {d: f(d)}
    """
    try:
        target = d.iteritems()
        return {k: f(v) for k, v in target}
    except AttributeError:
        return dict(it.izip(d, it.imap(f, d)))


def dfilter(d, f=None, by='k'):
    if by.startswith('k'):
        return {k: d[k] for k in it.ifilter(f, d)}
    elif by.startswith('v'):
        f = f or bool
        return dict(it.izip(it.ifilter(lambda i: f(i[1]), d.iteritems())))


def arr_to_dict(a):
    """convert array with named fields to dict of simple arrays"""
    return {field: a[field] for field in a.dtype.names}


def dict_to_arr(d, fields=None):
    """convert dict of arrays to structured array with named fields"""
    fields = fields or d.keys()
    dtype = np.dtype([(f, d[f].dtype) for f in fields])
    arrs = np.broadcast_arrays(*(d[f] for f in fields))
    a = np.empty(arrs.shape, dtype)
    for f, arr in it.izip(fields, arrs):
        a[f] = arr
    return a


def str_union(a, b=None):
    if b is None:
        return reduce(str_union, a)
    if a == b:
        return a
    elif len(a) == len(b):
        l = [ac if ac == bc else '?' for ac, bc in it.izip(a, b)]
        return ''.join(l)
    else:
        msg = "Lengths differ, resulting pattern may not match.\na: {}\nb: {}"
        # this seems to cause exception, not just warn?
        # raise RuntimeWarning(msg.format(a, b))
        print '-'*79
        print 'RuntimeWarning:', msg.format(a, b)
        print '_'*79
        l = ''.join([ac if ac == bc else '?' for ac, bc in it.izip(a, b)])
        r = ''.join([ac if ac == bc else '?'
                     for ac, bc in reversed(zip(*map(reversed, (a, b))))])
        l = l.partition('?')[0]
        r = r.rpartition('?')[-1]
        return '*'.join((l, r))


def str_tree(strs, n=1):
    t = {}
    for s in strs:
        t.setdefault(s[:n], []).append(s)
    for p in sorted(t):
        if len(t[p]) == 1:
            t[p] = t[p].pop()
        else:
            tn = str_tree(t.pop(p), n+1)
            if len(tn) == 1:
                p, tn = tn.popitem()
            t[p] = tn
    return t


def eval_string(s, hashable=False):
    s = s.strip()
    try:
        return {'True': True, 'False': False, 'None': None}[s]
    except KeyError:
        pass
    first, last = s[0], s[-1]
    if '\\' in s:
        s = ntpath.normpath(s)
    if first == last and first in '\'\"':
        return s[1:-1]
    if first in '[(' and last in ')]':
        l = map(eval_string, s[1:-1].split(', '))
        return l if first == '[' else tuple(l)
    nums = '-0123456789'
    issub = set(s).issubset
    if issub(set(nums + 'L')):
        return int(s)
    if issub(set(nums + '.+eE')) and not hashable:
        return float(s)
    return s

DRIVES = {
        'Walsh_Lab':  'Seagate15T', 'colson': 'Seagate4T',
        'Seagate\\ Expansion\\ Drive': 'Seagate4T',
        'Users':  'Darwin', 'home':   'Linux',
        'C:': 'Windows', 'H:': 'hopper', }


def drive_path(path, local=False, both=False):
    """
    split absolute paths from nt, osx, or linux into (drive, path) tuple
    """
    if isinstance(path, basestring):
        if local:
            path = os.path.abspath(path)
        path = path.replace(ntpath.sep, '/')
        drive, path = ntpath.splitdrive(path)
        if ntpath.isabs(path):
            split = path.split('/', 3)
            if split[1] in ('Volumes', 'media', 'mnt'):
                drive = split[2]
                path = '/' + split[3]
            else:
                drive = drive or split[1]
        drive = DRIVES.get(drive, drive)
        if local and drive == getsystem():
            drive = gethost()
        path = drive, path
    elif isinstance(path, tuple):
        letters = {'hopper': 'H:', 'Windows': 'C:'}
        mount = {'Darwin': '/Volumes/{}'.format,
                 'Linux': '/media/{}'.format,
                 'Windows': letters.get}[getsystem()]
        path = mount(path[0]) + path[1]
    return drive_path(path, local=local, both=False) if both else path


def with_suffix(prefix, suffix):
    if isinstance(prefix, basestring):
        return prefix if prefix.endswith(suffix) else prefix+suffix
    else:
        return [with_suffix(p, suffix) for p in prefix]


def with_prefix(suffix, prefix):
    if isinstance(suffix, basestring):
        return suffix if suffix.startswith(prefix) else prefix+suffix
    else:
        return [with_prefix(s, prefix) for s in suffix]


def fio(path, content='', mode='w'):
    if content:
        with open(path, mode) as f:
            f.writelines(content)
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
        return lines


def timestamp():
    timestamp, timezone = strftime('%Y-%m-%d %H:%M:%S,%Z').split(',')
    if len(timezone) > 3:
        timezone = ''.join(s[0] for s in timezone.split())
    return timestamp+' '+timezone


def find_prefix(*patterns):
    """find all the prefixes, optionally only those matching patterns."""
    patterns = with_prefix(with_suffix(patterns or [''], '*_LOG.txt'), '*')
    filenames = it.chain(*it.imap(glob.iglob, patterns))
    return [f.partition('_LOG')[0] for f in filenames]


def load_meta(prefix):
    path = with_suffix(prefix, '_META.txt')
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        lines = f.readlines()
    return dict(map(eval_string, l.split(':', 1)) for l in lines)


def load_fits(prefix, new_fits=None):
    """load existing fits file, and update with new fits if given."""
    try:
        from ruamel import yaml     # prefer ruamel.yaml (updated PyYAML fork)
    except ImportError:
        import yaml                 # fall-back to PyYAML if unavailable

    try:
        path = with_suffix(prefix, '_FITS.yaml')
        with open(path, 'r') as f:
            all_fits = yaml.load(f)
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

    ks = ('fit', 'result')
    for m in new_fits:
        try:
            i = all_fits[m]['fit'].index(new_fits[m]['fit'])
            # update the result:
            all_fits[m]['result'][i] = new_fits[m]['result']
        except ValueError as ve:
            # otherwise, append new fit and result:
            if 'is not in list' not in ve[0]:
                raise ve
            for k in ks:
                all_fits[m][k].append(new_fits[m][k])
        except AttributeError as ae:
            # fit may not be in a list
            for k in ks:
                if isinstance(all_fits[m][k], dict):
                    all_fits[m][k] = [all_fits[m][k]]
                else:
                    raise ae
        except KeyError as ke:
            if m in ke.args:
                all_fits[m] = new_fits[m]
            elif ke.args[0] in 'fitresult':
                all_fits[m].update(new_fits[m])
            else:
                raise ke
        except TypeError as te:
            if all_fits is None:
                return merge_fits({}, new_fits)
            else:
                raise te

    return all_fits


def save_fits(prefix, new_fits):
    """save dict of fits to yaml file.

    If file exists, load and merge them first
    """
    try:
        from ruamel import yaml     # prefer ruamel.yaml (updated PyYAML fork)
    except ImportError:
        import yaml                 # fall-back to PyYAML if unavailable
    path = with_suffix(prefix, '_FITS.yaml')
    fits = load_fits(path, new_fits)
    with open(path, 'w') as f:
        yaml.dump(fits, f)


def merge_meta(metas, conflicts={}, incl=set(), excl=set(), excl_start=()):
    """Merges several meta dicts into single dict without loss of data.

    parameters
    ----------
    metas : any number of member meta dicts as separate arguments
    conflicts: dict with same keys as `metas`, with values:
            'join': values from all metas in a list in order given (default)
            'mean': mean of the list given above
            'drop': do not include this key in output
            'fail': raise RuntimeError

    returns
    -------
    merged : the result of the merger as a single dict.
        the keys are the union of the members' keys. If a key is present in
        only one member, the value is taken from that member. If the key is
        present in multiple members, but the values match, the key and value
        will be in the output. if any values differ, all values are combined in
        a list, in the same order as the member dicts were given as arugments,
        as the the new value for that key.
    """
    merged = {}
    keys = incl or {key for meta in metas for key in meta.iterkeys()
                    if not key.startswith(excl_start)}
    keys -= excl
    for key in keys:
        conflict = conflicts.get(key, 'join')
        vals = [meta[key] for meta in metas if key in meta]
        u = set(vals)
        if len(u) == 1:
            merged[key] = u.pop()
        else:
            if conflict == 'fail':
                msg = "Values " + "{!r} "*len(vals) + "conflict for key {!r}"
                raise RuntimeError(msg.format(vals, key))
            elif conflict == 'join':
                merged[key] = vals
            elif callable(conflict):
                merged[key] = conflict(vals)
            else:
                raise ValueError("Unknown conflict choice {}".format(conflict))
    return merged


def meta_meta(patterns, include=set(), exclude=set()):
    """build a dict and/or structured array to hold meta values across datasets
    """
    if isinstance(patterns, basestring):
        patterns = [patterns]
    meta_names = [filename.replace('_META.txt', '') for p in patterns
                  for filename in glob.iglob(with_suffix(p, '_META.txt'))]
    print '\n'.join(meta_names)
    if not meta_names:
        raise RuntimeError("No files found")
    bases = map(os.path.basename, meta_names)
    source = np.array(bases if len(set(bases)) == len(bases) else meta_names)
    # dict of meta dicts {meta_name: meta}
    metas = dmap(load_meta, dict(zip(source, meta_names)))
    # all values keyed by meta key {key: {meta_name: meta[key]}}
    fields = transpose_dict(metas)
    keys = include or set(fields)
    keys -= exclude
    map(fields.pop, set(fields) - keys)
    dtypes = {k: np.array([v for v in f.itervalues() if v is not None]).dtype
              for k, f in fields.iteritems()}
    spaces = [replace_all(s.lower(), '_,', ' ') for s in source]
    mv = [int(s.partition('mv')[0].split()[-1]) for s in spaces]
    hz = [int(s.partition('hz')[0].split()[-1]) for s in spaces]
    dtype = [('source', source.dtype), ('mv', 'u2'), ('hz', 'u2')]
    dtype = np.dtype(dtype + dtypes.items())
    meta_array = np.empty(source.shape, dtype=dtype)
    meta_array['source'][:] = source
    meta_array['mv'][:] = mv
    meta_array['hz'][:] = hz
    bad = {'i': -1, 'f': np.nan, 'S': 'None', 'b': False, 'O': None, 'u': -1}
    for key, field in fields.iteritems():
        r = bad[dtypes[key].kind]
        meta_array[key][:] = [r if field[s] is None else field[s]
                              for s in source]
    return metas, fields, meta_array


def sync_args_meta(args, meta, argnames, metanames, defaults=None):
    """synchronize values between the args namespace and saved metadata dict

    Any non-`None` value in `args` overwrites the value (if any) in `meta`.
    Values of `None` in `args` are overwritten by either the value in `meta` if
    it exists, otherwise with the `default` if provided.

    Both `args` and `meta` are modified in-place

    parameters:
        args:       the argparse args namespace
        meta:       the metadata dict
        argnames:   keys to the entries in args to sync
        metanames:  corresponding keys to the entries in meta
        defaults:   value used if args value is None and missing from meta

    returns and modifies:
        args, meta: the modified mappings
    """
    argdict = args.__dict__
    if isinstance(argnames, basestring):
        argnames = argnames.split()
    if isinstance(metanames, basestring):
        metanames = metanames.split()
    if defaults is None:
        defaults = it.repeat(None)
    for argname, metaname, default in it.izip(argnames, metanames, defaults):
        if argdict[argname] is None:
            argdict[argname] = meta.get(metaname, default)
        else:
            meta[metaname] = argdict[argname]
    return args, meta


def change_meta(prefix, old_key, new_key=None, new_val=None, bak_key=None):
    """change keys and/or values in metadata files

    Given an old_key and optionally new_key or new_val, replace the old_key or
    old_val with whichever new items are given.

    Inputs may all be single value or lists. length of key/val lists must match.

    parameters:
        prefix:     prefix or list of prefixes
        old_key:    key (or list of keys) to meta dict to change
        new_key:    if not None, replace old_key with new_key
        new_val:    if not None, replace meta[new_key or old_key] with either
                    new_val or if a function is provided new_val(old_val)
        bak_key:    save old_val under bak_key as backup
    """
    if isinstance(prefix, basestring):
        prefix = [prefix]
    if not isinstance(old_key, list):
        old_key = [old_key]
    if not isinstance(old_key[0], basestring):
        raise TypeError("`old_key` must be string or list of strings, but "
                        "type(old_key) is {}".format(type(old_key[0])))
    if not isinstance(new_key, list):
        new_key = it.repeat(new_key)
    if not isinstance(new_val, list):
        new_val = it.repeat(new_val)
    if not isinstance(bak_key, list):
        bak_key = it.repeat(bak_key)
    for pfx in prefix:
        meta = load_meta(pfx)
        for ok, nk, nv, bk in it.izip(old_key, new_key, new_val, bak_key):
            meta = modify_dict(meta, ok, new_key=nk, new_val=nv, bak_key=bk)
        write_meta(pfx, meta)


def modify_dict(d, old_key, new_key=None, new_val=None, bak_key=None):
    """modify (inplace) a key and/or value in dict"""
    if new_key is not None:
        old_val = d.pop(old_key)
    elif new_val is None:
        raise ValueError("Provide either new_key or new_value. Both are None.")
    else:
        new_key = old_key
        old_val = d[old_key]
    if callable(new_val):
        new_val = new_val(old_val)
    elif new_val is None:
        new_val = old_val
    d[new_key] = new_val
    if bak_key is not None:
        d[bak_key] = old_val
    return d


def save_meta(prefix, meta_dict=None, **meta_kw):
    """append keys and values to meta file '{prefix}_META.txt'

    For keys that exist in more than one place, precedence is:
        keyword args, dict arg, file on disk

    parameters:
        prefix:         filename is '{prefix}_META.txt'
        meta_dict:      a dict with which to update file
        keyword args:   key-value pairs to update file and meta_dict
    """
    meta = load_meta(prefix)
    meta.update(meta_dict or {}, **meta_kw)
    write_meta(prefix, meta)


def write_meta(prefix, meta):
    """write keys and values to meta file '{prefix}_META.txt'

    Will overwrite any existing file without reading.

    parameters:
        prefix:     filename is '{prefix}_META.txt'
        meta:       dict to write to file
    """
    fmt = '{0[0]!r:18}: {0[1]!r}\n'.format
    lines = sorted(map(fmt, meta.iteritems()))
    path = with_suffix(prefix, '_META.txt')
    with open(path, 'w') as f:
        f.writelines(lines)


def swap_ij(a, i, j):
    """ad-hoc function for use to fix x-y transposition error"""
    b = list(a)
    b[i], b[j] = a[j], a[i]
    return type(a)(b)


def swapper(i, j):
    """ad-hoc function for use to fix x-y transposition error"""
    return lambda a: swap_ij(a, i, j)


def save_log_entry(prefix, entries, mode='a'):
    """save an entry to log file '{prefix}_LOG.txt'

    parameters:
        entries:    string or list of strings to write to file (may contain'\\n'
                    newlines), or the string 'argv' to write the `sys.argv`
        mode:       mode with which to write. only use 'a'
    """
    pre = '{} {}@{}/{}: '.format(timestamp(), getuser(), gethost(), getcommit())
    path = with_suffix(prefix, '_LOG.txt')
    if entries == 'argv':
        entries = [' '.join([os.path.basename(sys.argv[0])] + sys.argv[1:])]
    elif isinstance(entries, basestring):
        entries = entries.split('\n')
    entries = it.ifilter(None, it.imap(str.strip, entries))
    entries = pre + ('\n' + pre).join(entries) + '\n'
    with open(path, mode) as f:
        f.write(entries)


def clear_execution_counts(nbpath, inplace=False):
    import nbformat
    nb = nbformat.read(nbpath, nbformat.current_nbformat)
    for cell in nb['cells']:
        if 'execution_count' in cell:
            cell['execution_count'] = None
    out = nbpath if inplace else nbpath.replace('.ipynb', '.nulled.ipynb')
    nbformat.write(nb, out)


def load_data(fullprefix, sets='tracks', verbose=False):
    """ Load data from an npz file

        Given `fullprefix`, returns data arrays from a choice of:
            tracks, orientation, position, corner
    """
    sets = [s[0].lower() for s in sets.replace(',', ' ').strip().split()]

    # sort for optimal loading order
    allsets = [('t', 'tracks'), ('p', 'positions'), ('o', 'orientation'),
               ('c', 'corner_positions'), ('m', 'melt')]
    allsets = [(s, setname) for (s, setname) in allsets if s in sets]

    npzs = {}
    data = {}
    needs_initialize = False
    for s, setname in allsets:
        suffix = setname.upper()
        datapath = fullprefix+'_'+suffix+'.npz'
        try:
            npzs[s] = np.load(datapath)
        except IOError as e:
            if s == 'p':
                if verbose:
                    print "No positions file, loading from tracks"
                data[s] = data['t'] if 't' in data else load_data(fullprefix,
                                                                  't', verbose)
            elif s == 't':
                if verbose:
                    print "No tracks file, loading from positions"
                if 'p' not in sets:
                    data['p'] = load_data(fullprefix, 'p', verbose)
                needs_initialize = True
                t = -1
            else:
                print e
                cmd = '`tracks -{}`'.format(
                    {'t': 't', 'o': 'o', 'p': 'l', 'c': 'lc'}[s])
                print ("Found no {} npz file. Please run ".format(setname) +
                       ("{0} to convert {1}.txt to {1}.npz, " +
                        "or run `positions` on your tiffs" if s in 'pc' else
                        "{0} to generate {1}.npz").format(cmd, suffix))
                raise
        else:
            if verbose:
                print "Loaded {} data from {}".format(setname, datapath)
            data[s] = npzs[s][s*(s == 'o') + 'data']
            if s == 't' and 'trackids' in npzs['t'].files:
                needs_initialize = True
                t = npzs['t']['trackids']
                if 'p' not in sets:
                    data['p'] = data[s]
    if needs_initialize:
        # separate `trackids` array means this file is an old-style
        # TRACKS.npz which holds positions and trackids but no orient
        if verbose:
            print "Converting to TRACKS array from positions, trackids, orient"
        if 'o' not in data:
            data['o'] = load_data(fullprefix, 'o', verbose)
        data['t'] = initialize_tdata(data['p'], t, data['o']['orient'])
    if 't' in sets:
        data['t'] = add_self_view(data['t'], ('x', 'y'), 'xy')
    ret = [data[s] for s in sets]
    return ret if len(ret) > 1 else ret[0]


def load_MSD(fullprefix, pos=True, ang=True):
    """ Loads ms(a)ds from an MS(A)D.npz file

        Given `fullprefix`, and choice of position and angular

        Returns [`msds`, `msdids`,] [`msads`, `msadids`,] `dtau`, `dt0`
    """
    ret = ()
    if pos:
        msdnpz = np.load(fullprefix+'_MSD.npz')
        ret += msdnpz['msds'], msdnpz['msdids']
        dtau = msdnpz['dtau'][()]
        dt0 = msdnpz['dt0'][()]
    if ang:
        msadnpz = np.load(fullprefix+'_MSAD.npz')
        ret += msadnpz['msds'], msadnpz['msdids']
        if pos:
            assert dtau == msadnpz['dtau'][()] and dt0 == msadnpz['dt0'][()], \
                'dt mismatch'
        else:
            dtau = msadnpz['dtau'][()]
            dt0 = msadnpz['dt0'][()]
    ret += dtau, dt0
    print 'loading MSDs for', fullprefix
    return ret


def load_tracksets(data, trackids=None, min_length=10, max_length=None,
                   reverse=False, verbose=False, run_remove_dupes=False,
                   run_repair=False, run_track_orient=False):
    """Build a dict of slices into data based on trackid

    parameters:
        data:               array to group by trackid
        trackids:           values by which to split. if None, uses data['t']
        min_length:         only include tracks with at least this many members
                        or, include the longest N tracks with min_length = -N
        run_remove_dupes:   whether to run tracks.remove_dupes on the data
        run_repair:         whether to fill/interp gaps (see tracks -h (--gaps))
        run_track_orient:   whether to track orients so angles are not mod 2pi

    returns:
        tracksets:          a dict of {trackid: subset of `data`}
    """
    if trackids is None:
        # copy actually speeds it up by a factor of two
        trackids = data['t'].copy()
    elif not trackids.flags.owndata:
        # copy in case called as ...(data, data['t'])
        trackids = trackids.copy()
    if min_length:
        lengths = np.bincount(trackids+1)[1:]
        if min_length < 0:
            ts = lengths.argsort()[min_length:]
        else:
            ts = np.where(lengths >= min_length)[0]
    else:
        ts = np.unique(trackids)
        if ts[0] == -1:
            ts = ts[1:]
    reverse = int(reverse)
    step = 1 - 2*reverse
    tracksets = {t: data[trackids == t][::step] for t in ts}
    if reverse:
        for t in tracksets:
            tracksets[t]['f'] = tracksets[t]['f'].max() - tracksets[t]['f']
    if run_remove_dupes:
        from tracks import remove_duplicates
        remove_duplicates(tracksets=tracksets, inplace=True, verbose=verbose)
    if run_track_orient:
        from orientation import track_orient
        for track in tracksets:
            tracksets[track]['o'] = track_orient(tracksets[track]['o'])
    if run_repair and run_repair != 'leave':
        from tracks import repair_tracks
        fs = () if run_repair == 'nans' else ('xy', 'o')
        repair_tracks(tracksets, interp=fs, inplace=True, verbose=verbose)
    start = max_length and reverse and -max_length
    stop = max_length and max_length*(1 - reverse) or None
    if start or stop:
        tracksets = {t: tracksets[t][start:stop] for t in ts}
    return tracksets


def load_framesets(data_or_tracksets, indices='f', ret_dict=True, **tset_args):
    if tset_args:
        data_or_tracksets = load_tracksets(data_or_tracksets, **tset_args)
    if not isinstance(data_or_tracksets, tuple):
        data_or_tracksets = (data_or_tracksets,)
    data = []
    splitargs = {'ret_dict': ret_dict}
    for datum in data_or_tracksets:
        if hasattr(datum, 'values'):
            splitargs['noncontiguous'] = True
            datum = np.concatenate([datum[k] for k in sorted(datum)])
        data.append(datum)
    return splitter(tuple(data), indices, **splitargs)


def loadall(fullprefix, ret_msd=True, ret_fsets=False):
    """ returns data, tracksets, odata, otracksets,
         + (msds, msdids, msads, msadids, dtau, dt0) if ret_msd
    """
    print "warning, this may not return what you want"
    data = load_data(fullprefix, 'tracks')
    tracksets = load_tracksets(data, odata, omask)
    ret = data, tracksets, odata, otracksets
    if ret_msd:
        ret += load_MSD(fullprefix, True, True)
    if ret_fsets:
        fsets = splitter(data, ret_dict=True)
        fosets = splitter(odata[omask], data['f'][omask], ret_dict=True)
        ret += (fsets, fosets)
    return ret


def fields_view(arr, fields):
    """ by [HYRY](https://stackoverflow.com/users/772649/hyry)
        at http://stackoverflow.com/a/21819324/
    """
    dtype2 = np.dtype({name: arr.dtype.fields[name] for name in fields})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)


def quick_field_view(arr, field, careful=False):
    """quickly return a view onto a field in a structured array

    Not sure why but this turns out to be much faster than get-item lookup, good
    for accessing fields inside loops

    parameters:
        arr:    a structured array
        field:  a single field name
    """
    dt, off = arr.dtype.fields[field]
    out = np.ndarray(arr.shape, dt, arr, off, arr.strides)
    if careful:
        i = arr.dtype.names.index(field)
        a, o = arr.item(-1)[i], out[-1]
        if dt.shape is ():
            assert a == o or np.isnan(a) and np.isnan(o)
        else:
            eq = a == o
            assert np.all(eq) or np.all(eq[np.isfinite(a*o)])
    return out


def consecutive_fields_view(arr, fields, careful=False):
    """quickly return a view of consecutive fields in a structured array.

    The fields must be consecutive and must all have the same itemsize.

    parameters:
        arr:      a structured array
        fields:   list of field names (1 string ok for one-char field names)
        careful:  be careful (check shapes/values). default is False.

    returns:
        view:     a view onto the fields
    """
    shape, j = arr.shape, len(fields)
    df = arr.dtype.fields
    dt, offset = df[fields[0]]
    strides = arr.strides
    if j > 1:
        shape += (j,)
        strides += (dt.itemsize,)
    out = np.ndarray(shape, dt, arr, offset, strides)
    if careful:
        names = arr.dtype.names
        i = names.index(fields[0])
        assert tuple(fields) == names[i:i+j], 'fields not consecutive'
        assert all([df[f][0] == dt for f in fields[1:]]), 'fields not same type'
        l, r = arr.item(-1)[i:i+j], out[-1]
        assert all([a == o or np.isnan(a) and np.isnan(o)
                    for a, o in it.izip(l, r)]), \
            "last row mismatch\narr: {}\nout: {}".format(l, r)
    return out


def add_self_view(arr, fields, name):
    dtype = dict(arr.dtype.fields)
    dt, off = dtype[fields[0]]
    dtype[name] = np.dtype((dt, (len(fields),))), off
    return np.ndarray(arr.shape, dtype, arr, 0, arr.strides)


track_dtype = np.dtype({'names': '  id f  t  x  y  o'.split(),
                        'formats': 'u4 u2 i4 f4 f4 f4'.split()})
pos_dtype = np.dtype({'names': '    f  x  y  lab ecc area id'.split(),
                      'formats': '  i4 f8 f8 i4  f8  i4   i4'.split()})
vel_dtype = np.dtype({'names': 'o v x y par perp eta etax etay etapar'.split(),
                      'formats': ['f4']*10})


def initialize_tdata(pdata, trackids=-1, orientations=np.nan):
    if pdata.dtype == track_dtype:
        data = pdata
    else:
        data = np.empty(len(pdata), dtype=track_dtype)
        for field in pdata.dtype.names:
            if field in track_dtype.names:
                data[field] = pdata[field]
    if trackids is not None:
        data['t'] = trackids
    if orientations is not None:
        data['o'] = orientations
    return data


def dtype_info(dtype='all'):
    if dtype == 'all':
        [dtype_info(s+b) for s in 'ui' for b in '1248']
        return
    dt = np.dtype(dtype)
    bits = 8*dt.itemsize
    if dt.kind == 'f':
        print np.finfo(dt)
        return
    if dt.kind == 'u':
        mn = 0
        mx = 2**bits - 1
    elif dt.kind == 'i':
        mn = -2**(bits-1)
        mx = 2**(bits-1) - 1
    print "{:6} ({}{}) min: {:20}, max: {:20}".format(
                dt.name, dt.kind, dt.itemsize, mn, mx)


def txt_to_npz(datapath, verbose=False, compress=True):
    """ Reads raw txt positions data into a numpy array and saves to an npz file

        `datapath` is the path to the output file from finding particles
        it must end with "results.txt" or "POSITIONS.txt", depending on its
        source, and its structure is assumed to match a certain pattern
    """
    if not os.path.exists(datapath):
        if os.path.exists(datapath+'.gz'):
            datapath += '.gz'
        elif os.path.exists(datapath[:-3]):
            datapath = datapath[:-3]
        else:
            raise IOError('File {} not found'.format(datapath))
    if verbose:
        print "loading positions data from", datapath,
    from numpy.lib.recfunctions import append_fields
    # successfully tested the following reduced datatype sizes on
    # Squares/diffusion/orientational/n464_CORNER_POSITIONS.npz
    # with no real loss in float precision nor cutoff in ints
    # names:   'f  x  y  lab ecc area id'
    # formats: 'u2 f4 f4 i4  f2  u2   u4'
    data = np.genfromtxt(datapath, skip_header=1,
                         names="f,x,y,lab,ecc,area",
                         dtype="u2,f4,f4,i4,f2,u2")
    ids = np.arange(len(data), dtype='u4')
    data = append_fields(data, 'id', ids, usemask=False)
    if verbose:
        print '...',
    outpath = datapath[:datapath.rfind('txt')] + 'npz'
    (np.savez, np.savez_compressed)[compress](outpath, data=data)
    if verbose:
        print 'and saved to', outpath
    return


def compress_existing_npz(path, overwrite=False, careful=False):
    orig = np.load(path)
    amtime = os.path.getatime(path), os.path.getmtime(path)
    arrs = {n: orig[n] for n in orig.files}
    if careful or not overwrite:
        out = path[:-4]+'_compressed.npz'
    else:
        out = path
    np.savez_compressed(out, **arrs)
    os.utime(out, amtime)
    if careful:
        comp = np.load(out)
        assert comp.files == orig.files
        for n in comp.files:
            assert np.all(np.nan_to_num(orig[n]) == np.nan_to_num(comp[n])),\
                    'FAIL {}[{}] --> {}'.format(path, n, out)
        if overwrite:
            os.rename(out, path)
        print 'Success!'


def merge_data(members, savename=None, dupes=False, do_orient=False):
    """ returns (and optionally saves) new `data` array merged from list or
        tuple of individual `data` arrays or path prefixes.

        parameters
        ----------
        members : list of arrays, prefixes, or single prefix with wildcards
        savename : path prefix at which to save the merged data,
            saved as "<savename>_MRG_<TRACKS|ORIENTATION>.npz"
        do_orient : True or False, whether to merge the orientation data as
            well. default is False, NOT IMPLEMENTED

        returns
        -------
        merged : always returned, the main merged data array

        if orientational data is to be merged, then a list of filenames or
        prefixes must be given instead of data objects.

        only data is returned if array objects are given.
    """
    if do_orient:
        raise ValueError('do_orient is not possible')
    if isinstance(members, basestring):
        pattern = members
        suf = '_TRACKS.npz'
        l = len(suf)
        pattern = pattern[:-l] if pattern.endswith(suf) else pattern
        members = [s[:-l] for s in glob.iglob(pattern+suf)]

    n = len(members)
    assert n > 1, "need more than {} file(s)".format(n)

    if isinstance(members[0], basestring):
        members.sort()
        print '\n\t'.join(['Merging:'] + members)
        datasets = map(load_data, members)
    else:
        datasets = members

    track_increment = 0
    for dataset in datasets:
        ts = quick_field_view(dataset, 't', False)
        if dupes:
            from tracks import remove_duplicates
            ts[:] = remove_duplicates(ts, dataset)
        ts[ts >= 0] += track_increment
        track_increment = ts.max() + 1

    merged = np.concatenate(datasets)

    if savename:
        meta_conflicts = dict(track_cut='fail', fps='fail', sidelength='fail',
                              path_to_tiffs=str_union)
        merged_meta = merge_meta(map(load_meta, members),
                                 conflicts=meta_conflicts,
                                 excl_start=('fit_',))
        savedir = os.path.dirname(savename) or os.path.curdir
        if not os.path.exists(savedir):
            print "Creating new directory", savedir
            os.makedirs(savedir)
        if n*len(members[0]) > 200:
            pattern = pattern or str_union(members)
        else:
            pattern = members
        args = ', dupes=True'*dupes + ', do_orient=True'*do_orient
        entry = 'merge_data(members={!r}, savename={!r}{})'
        entry = entry.format(pattern, savename, args)
        suffix = '_MRG'
        if not (savename.endswith(suffix) or savename.endswith('_MERGED')):
            savename += suffix
        save_log_entry(savename, entry)
        save_meta(savename, merged_meta, merged=members)
        savename += '_TRACKS.npz'
        # if the 'xy' view field exists in the dtype, remove it before saving:
        dt = dict(merged.dtype.fields)
        dt.pop('xy', None)
        np.savez_compressed(savename, data=merged.view(dt))
        print "saved merged tracks to", savename
    return merged


def bool_input(question='', default=None):
    "Returns True or False from yes/no user-input question"
    if question and question[-1] not in ' \n\t':
        question += ' '
    answer = raw_input(question).strip().lower()
    if answer == '':
        if default is None:
            # ask again
            return bool_input(question, default)
        else:
            return default
    return answer.startswith('y') or answer.startswith('t')


def farange(start, stop, factor):
    start_power = log(start, factor)
    stop_power = log(stop, factor)
    dt = np.result_type(start, stop, factor)
    return factor**np.arange(start_power, stop_power, dtype=dt)


def mode(x, count=False):
    """Find the mode of an integer array.

    parameters
    x:      array of integers
    count:  if True, will return the mode of frequencies of x instead of values

    returns
    mode:   a scalar
    """
    if count:
        x = np.bincount(x)
        m = 0
    else:
        m = x.min()
    return np.argmax(np.bincount(x-m)) + m


def decade(n, m=None):
    if m in (None, 'down'):
        return 10**int(np.log10(n))
    elif m == 'up':
        return 10**int(1 + np.log10(n))
    return decade(n, 'down'), decade(m, 'up')


def loglog_slope(x, y, smooth=0):
    dx = 0.5*(x[1:] + x[:-1])
    dy = np.diff(np.log(y)) / np.diff(np.log(x))

    if smooth:
        from scipy.ndimage import gaussian_filter1d
        dy = gaussian_filter1d(dy, smooth, mode='reflect')
    return dx, dy


def dist(a, b):
    """ The 2d distance between two arrays of shape (N, 2) or just (2,)
    """
    return np.hypot(*(a - b).T)


def rotate(v, theta, out=None):
    """rotate a 2-d vector into the basis defined by theta

    parameters
    ----------
    v:      vector(s) with shape (2, ...)
    theta:  angle(s) with shape (...,)

    returns
    -------
    vout:   v rotated along theta

    take ihat =  cos(theta), sin(theta)
    thus jhat = -sin(theta), cos(theta)
    then vout =  v dot ihat, v dot jhat
    """
    if out is None:
        out = np.empty_like(v)
    cos, sin = np.cos(theta), np.sin(theta)
    rot = np.array([[cos, sin],
                    [-sin, cos]])
    return np.einsum('ij...,...j', rot, v, out=out)


def circle_three_points(*xs):
    """ With three points, calculate circle
        e.g., see paulbourke.net/geometry/circlesphere
        returns center, radius as (xo, yo), r
    """
    xs = np.squeeze(xs)
    if xs.shape == (3, 2):
        xs = xs.T
    (x1, x2, x3), (y1, y2, y3) = xs

    ma = (y2-y1)/(x2-x1)
    mb = (y3-y2)/(x3-x2)
    xo = ma*mb*(y1-y3) + mb*(x1+x2) - ma*(x2+x3)
    xo /= 2*(mb-ma)
    yo = (y1+y2)/2 - (xo - (x1+x2)/2)/ma
    r = ((xo - x1)**2 + (yo - y1)**2)**0.5

    return xo, yo, r


def parse_slice(desc, shape=0, index_array=False):
    if desc is True:
        print "enter number or range as slice start:end",
        if shape:
            print "for shape {}".format(shape),
        desc = raw_input('\n>>> ')
    if isinstance(desc, slice):
        slice_obj = desc
    else:
        if isinstance(desc, basestring):
            args = [int(s) if s else None for s in desc.split(':')]
        else:
            args = np.atleast_1d(desc)
        if len(args) <= 3:
            slice_obj = slice(*args)
        elif index_array:
            return args
        else:
            raise ValueError("too many args for slice")
    if index_array:
        if shape:
            return np.arange(*slice_obj.indices(shape))
        else:
            return np.r_[slice_obj]
    else:
        return slice_obj


def find_tiffs(path='', prefix='', meta='', frames='', single=False,
               load=False, verbose=False):
    meta = meta or load_meta(prefix)
    path = path or meta.get('path_to_tiffs', prefix)
    path = drive_path(path, both=True)
    if frames in (0, '0', 1, '1'):
        single = True
        frames = 0
        prefix_dir, prefix_base = os.path.split(prefix)
        imnames = it.product([prefix_base + '_', '', '_'],
                             ['image_', 'image', ''],
                             ['000', '00'], '01', ['.tif'])
        for first_frame in imnames:
            first_frame = os.path.join(prefix_dir, ''.join(first_frame))
            if os.path.isfile(first_frame):
                path = first_frame
                break
    tar = path.endswith(('.tar', '.tbz', '.tgz')) and os.path.isfile(path)
    if tar:
        if verbose:
            print 'loading tarfile', os.path.basename(path)
        import tarfile
        tar = tarfile.open(path)
        fnames = [f for f in tar if f.name.endswith('.tif')]
    else:
        if os.path.isdir(path):
            path = os.path.join(path, '*.tif')
        if glob.has_magic(path):
            if verbose:
                print 'seeking matches to', path
            fnames = glob.glob(path)
        elif os.path.isfile(path):
            fnames = [path]
        else:
            fnames = []
    if fnames:
        nfound = len(fnames)
        if verbose or frames is True:
            print 'found {} images'.format(nfound)
        if single:
            frames = slice(int(frames), int(frames)+1)
        else:
            frames = parse_slice(frames, nfound)
        fnames.sort()
        fnames = fnames[frames]
        if load:
            from scipy.ndimage import imread
            fnames = fnames[:100]
            if verbose:
                print '. . .',
            imfiles = map(tar.extractfile, fnames) if tar else fnames
            fnames = np.squeeze(map(imread, imfiles))
            if verbose:
                print 'loaded'
        if tar:
            tar.close()
        return path, fnames, frames.indices(nfound)
    else:
        print "No files found; please correct the path"
        print "the following substitutions will be made:"
        substr = ("{{prefix}} --> {prefix}\n"
                  "{{meta}}   --> {meta}")
        subval = dict(prefix=prefix, meta=meta.get('path_to_tiffs', ''))
        print substr.format(**subval)
        print '    {}'.format(path)
        new_path = raw_input('>>> ').format(**subval)
        return find_tiffs(path=new_path, prefix=prefix, meta=meta,
                          frames=frames, single=single, load=load, verbose=True)


def circle_click(im):
    """saves points as they are clicked, then find the circle that they define

    To use:
    when image is shown, click three non-co-linear points along the perimeter.
    neither should be vertically nor horizontally aligned (gives divide by zero)
    when three points have been clicked, a circle should appear.
    Then close the figure to allow the script to continue.
    """
    import matplotlib
    if matplotlib.is_interactive():
        raise RuntimeError("Cannot do circle_click in interactive/pylab mode")
    from matplotlib import pyplot as plt

    print ("Please click three points on circumference of the boundary, "
           "then close the figure")
    if isinstance(im, basestring):
        im = plt.imread(im)
    xs = []
    ys = []
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(im)

    def circle_click_connector(click):
        # convert to image coordinates from cartesian
        x, y = click.ydata, click.xdata
        xs.append(x)
        ys.append(y)
        print 'click {}: x: {:.2f}, y: {:.2f}'.format(len(xs), x, y)
        if len(xs) == 3:
            # With three points, calculate circle
            print 'got three points'
            global xo, yo, r  # can't access connector function's returned value
            xo, yo, r = circle_three_points(xs, ys)
            cpatch = matplotlib.patches.Circle([yo, xo], r, linewidth=3,
                                               color='g', fill=False)
            ax.add_patch(cpatch)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', circle_click_connector)
    plt.show()
    return xo, yo, r


def draw_circles(centers, rs, ax=None, fig=None, **kwargs):
    """draw circles on an axis

    parameters:
        centers:    one or a list of (x, y) pairs
        rs:         one or a list of radii (in data units)
        ax or fig:  axis or figure on which to draw
        kwargs:     arguments passed to the patch (e.g.: color, fill, zorder)

    returns:
        patches:    a list of the patch objects
    """
    from matplotlib.patches import Circle
    if np.isscalar(rs):
        rs = it.repeat(rs)
    centers = np.atleast_2d(centers)
    patches = [Circle(c, abs(r), **kwargs) for c, r in it.izip(centers, rs)]
    if ax is None:
        if fig is None:
            from matplotlib.pyplot import gca
            ax = gca()
        else:
            ax = fig.gca()
    map(ax.add_patch, patches)
    ax.figure.canvas.draw()
    return patches


def check_neighbors(prefix, frame, data=None, im=None, **neighbor_args):
    import matplotlib.pyplot as plt
    from correlation import neighborhoods
    fig, ax = plt.subplots()

    if im is None:
        im = find_tiffs(prefix=prefix, frames=frame, single=True, load=True)[1]
    ax.imshow(im, cmap='gray', origin='lower')

    if data is None:
        data = load_data(prefix)
        data = data[data['t'] >= 0]
    fdata = splitter(data, 'f')
    frame = fdata[frame]
    positions = frame['xy']
    neighs, mask, dists = neighborhoods(positions, **neighbor_args)
    fmt = 'particle {} track {}: {} neighbors at dist {} - {} ({})'.format
    for pt, ns, m, ds, d in zip(positions, neighs, mask, dists, frame):
        ns, ds = ns[~m], ds[~m]
        if len(ns) == 0:
            continue
        print fmt(d['id'], d['t'], len(ns), ds.min(), ds.max(), ds.mean())
        # print '\tneighbors:', (len(ns)*'{:5d} ').format(*ns)
        # print '\tdistances:', (len(ns)*'{:5.2f} ').format(*ds)
        ax.scatter(*positions.T[::-1], c='k', marker='o')
        ax.scatter(*positions[ns].T[::-1], c='w', marker='o')
        ax.scatter(pt[1], pt[0], c='r', marker='o')
        plt.waitforbuttonpress()


def derivwidths(w, dx, sigmas, order):
    from scipy.ndimage import gaussian_filter1d
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.arange(-w, w+1)*dx
    y = x.max() - 2*np.abs(x)
    np.clip(y, 0, None, out=y)
    dy = -2*np.sign(x)
    dy[y == 0] = 0

    ax.plot(x, y, '-k', zorder=10)
    ax.plot(x, dy, '-k', zorder=10)
    sigmin, sigptp = min(sigmas), np.ptp(sigmas)
    print 'sigma\tsum(g)\tsum(abs(g))\tptp\tchi2'
    for sigma in sigmas:
        c = plt.cm.jet((sigma-sigmin)/sigptp)
        g = gaussian_filter1d(y, sigma, order=order, truncate=w/sigma)/dx
        # gi = gaussian_filter1d(y, sigma, order=order-1, truncate=w/sigma)
        ax.plot(x, g, alpha=.5, c=c)
        # ax.plot(x, gi, ':', c=c)
        ax.plot(x, np.cumsum(g)*dx, '--', alpha=.75, c=c)
        print ('{:.3f}\t'*5).format(
            sigma, g.sum(), np.abs(g).sum(), g.ptp(), ((g-dy)**2).sum())


class SciFormatter(Formatter):
    def format_field(self, value, format_spec):
        if format_spec.endswith('T'):
            s = format(value, format_spec[:-1])
            if 'e' in s:
                s = s.replace('e', r'\times10^{') + '}'
        elif format_spec.endswith('t'):
            s = format(value, format_spec[:-1]).replace('e', '*10**')
        else:
            s = format(value, format_spec)
        return s

# Pixel-Physical Unit Conversions
mm_per_inch = 25.4

# Physical measurements
R_inch = 4.0           # as machined
R_mm = R_inch * mm_per_inch
S_inch_caliper = np.array([4, 3, 6, 7, 9, 1, 9, 0, 0, 4, 7, 5, 3, 6, 2, 6, 0, 8,
                           8, 4, 3, 4, 0, -1, 0, 1, 7, 7, 5, 7])*1e-4 + .309
S_inch = S_inch_caliper.mean()
S_mm = S_inch * mm_per_inch
R_S = R_inch / S_inch

# Digital measurements (pixels)
# Still (D5000 SLR)
R_slr = 2459 / 2
# instead use S_slr = R_slr/R_S
# S_slr_m = np.array([3.72, 2.28, 4.34, 3.94, 2.84, 4.23, 4.87, 4.73, 3.77])
# S_slr = S_slr_m.mean() + 90

# Video (Phantom)
R_vid = 585.5 / 2
# instead use S_vid = R_vid/R_S
# S_vid_m = 22  # ish

# What we'll use:
R = R_S             # radius in particle units
S_slr = R_slr/R     # particle in still pixels
S_vid = R_vid/R     # particle in video pixels
A_slr = S_slr**2    # particle area in still pixels
A_vid = S_vid**2    # particle area in video pixels


def Nb(margin):
    """N = max number of particles (πR^2)/S^2 where S = 1
    """
    return np.pi * (R - margin)**2

N = Nb(0)
