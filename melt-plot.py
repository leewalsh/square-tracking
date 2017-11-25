"""Plot global mean vs time from each configuration."""

save = False
# 1 - edge inward, with timescale and background levels marked, colored by size, cluster size
stats = ['phi', 'psi', 'dens', 'rad','clust']
nruns = 5
config_names = [('pdms', 'in', n) for n in range(7, 12)]
xmax = 1200
colorby = 'W'
## 2 - all 11x1, with timescale marked
#stats = ['phi', 'psi', 'dens', 'clust']
#nruns = 4
#config_names = [('pdms', 'in', 11), ('pdms', 'rand', 11), ('lego', '', 11)]
#xmax = 1200
#colorby = 'arrange'
## 3 - edge inward, with timescale marked, phi-psi parametric
#stats = ['phi', 'psi', 'psi/phi']
#nruns = 5
#config_names = [('pdms', 'in', n) for n in range(7,12)]
#xmax = 680
#colorby = 'W'
## 4 - edge random, with timescale marked, cluster size
#stats = ['phi', 'psi', 'dens', 'rad', 'clust']
#nruns = 2
#config_names = [('pdms', 'rand', n) for n in range(7,12)]
#xmax = 200
#colorby = 'W'

ncols, nrows = nruns + 1, len(stats)
figsize = np.array(plt.rcParams['figure.figsize'])*3/4.
figsize *= [ncols, nrows]

fig, axeses = plt.subplots(figsize=figsize, squeeze=False,
                           ncols=ncols, nrows=nrows,
                           sharex='row', sharey='row',
                           dpi=120,
                          )

for config in config_names:
    print config
    config = config_dict[config]
    runs = ['MRG'] + range(1, 1+min(config['runs'], nruns))
    N = config['W']**2

    for run, axes in zip(runs, axeses.T):
        print str(run).center(3),
        loaded = load_melting_stuff(run=run, **config)
        prefix, meta, data, mdata, frames, mframes, means, plot_args_orig = loaded
        plot_args = plot_args_orig.copy()

        all_sh = len(means.keys()) - 1

        color = colors[config[colorby]]
        plot_args['line_props'] = list(plot_args_orig['line_props'])
        plot_args['line_props'][all_sh] = plot_args['line_props'][all_sh].copy()
        plot_args['line_props'][0] = plot_args['line_props'][0].copy()

        plot_args['line_props'][all_sh]['c'] = color
        plot_args['line_props'][all_sh]['label'] = config['label']
        plot_args['line_props'][all_sh]['ls'] = '--'

        plot_args['line_props'][0]['c'] = color
        plot_args['line_props'][0]['label'] = config['label']
        plot_args['line_props'][0]['lw'] = 1

        axes[0].set_title("run {}".format(run))

        fs = means['c']['f']
        f0 = meta['start_frame']
        ff = meta.get('end_frame') or len(fs)
        ts = (fs[:ff] - f0)/meta['fps']
        pos_ts = fs[:ff-f0]/meta['fps']
        frame_size = np.array(map(len, mframes[:ff]))
        raw_cluster_size = frame_size - np.array([
            np.count_nonzero(mframe['clust'])
            for mframe in mframes[:ff]])
        cluster_size = N * raw_cluster_size / frame_size
        min_size = 16
        cluster_death = (cluster_size < min_size).argmax() or cluster_size.argmin()
        cluster_min = cluster_size[cluster_death]

        for stat, ax in zip(stats, axes):
        print

fig.tight_layout(h_pad=0, w_pad=0)

if save:
    save_name = savedir + 'configs_dens_rad_v_time.pdf'
    print 'saving to', save_name
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)
    #plt.close(fig.number)


def plot_by_cluster(x, y, smooth):
    unit = plot_args.get('unit', {x: 1, y: 1})
    ax = plot_args.get('ax')
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    smooth = meta['fps']

    xs = cluster_means[x] - start
    ys = cluster_means[y]
    if np.issubdtype(vs.dtype, complex):
        ys = np.abs(ys)
    if smooth:
        ys = gaussian_filter1d(ys, smooth, mode='nearest', truncate=2)

    if y == 'clust':
        ax.plot(ts, cluster_size,
                color=color, lw=2, label=config['label'])
        helplt.axline(ax, 'h', min_size, coords='ax', color='k', lw=2, zorder=0.5)
        helplt.axline(ax, 'h', cluster_min, stop=ts[cluster_death], coords='data', color=color)
        helplt.axline(ax, 'v', ts[cluster_death], stop=cluster_min, coords='data', color=color)
        ax.set_ylabel('cluster size')

    if x == 'f':
        ax.plot(xs*unit[x], ys*unit[y], **line_props)
        ax.set_xlim(plot_args['xylim'][x])
        ax.set_ylim(plot_args['xylim'][y])
    elif x in ('dens', 'phi', 'psi'):
        xs, ys = cluster_means[x], cluster_means[y]
        if np.issubdtype(xs.dtype, complex):
            xs = np.abs(xs)
        if np.issubdtype(ys.dtype, complex):
            ys = np.abs(ys)
        if smooth:
            xs = gaussian_filter1d(xs, smooth, mode='nearest', truncate=2)
            ys = gaussian_filter1d(ys, smooth, mode='nearest', truncate=2)
        ax.plot(xs*unit[x], ys*unit[y], **line_props)
    else:
        raise ValueError("Unknown x-value `{!r}`.".format(x))

    ax.legend(fontsize='small')
    ax.set_xlabel(plot_args['xylabel'][x])
    ax.set_ylabel(plot_args['xylabel'][y])

    if x == 'f' and y in ('phi', 'psi'):
        style = dict(linestyle=':', color=color)
        op = np.abs(means['c'][y])[f0:ff]
        nans = np.isnan(op)
        if nans.any():
            first_nan = nans.nonzero()[0][0]
            op = op[:first_nan]
        t = pos_ts[:len(op)]

        op_bg = random_orient_OP(y, N=N)
        timescale = curve.decay_scale(op - op_bg, x=1/meta['fps'], method='mean', smooth='', rectify=False)
        assert np.all(np.isfinite(timescale)), 'timescale not finite'
        helplt.mark_value(ax, timescale, str(int(round(timescale))), method='vline', line=dict(style, stop=1))#, annotate=dict(xy=(timescale, .9)))
        ax.plot(t, np.ones_like(op)*op_bg, **style)
        print '\t', y, '{:4d}'.format(int(round(timescale))),

        op_bg = random_orient_OP(y, N=raw_cluster_size[f0:f0+len(op)])
        timescale = curve.decay_scale(op - op_bg, x=1/meta['fps'], method='mean', smooth='', rectify=False)
        style['linestyle'] = '--'
        helplt.mark_value(ax, timescale, 0*str(int(round(timescale))), method='vline', line=dict(style, stop=1))#, annotate=dict(xy=(timescale, .9)))
        ax.plot(t, np.ones_like(op)*op_bg, **style)
        print '{:4d}'.format(int(round(timescale))),

        if cluster_death - f0 < len(op):
            timescale = curve.decay_scale((op - op_bg)[:cluster_death-f0], x=1/meta['fps'], method='mean', smooth='', rectify=False)
            style['linestyle'] = '-'
            helplt.axline(ax=ax, coords='data', orient='v', x=t[cluster_death-f0], **style)
            helplt.mark_value(ax, timescale, 0*str(int(round(timescale))), method='vline', line=dict(style, stop=1))#, annotate=dict(xy=(timescale, .9)))
            print '{:4d}'.format(int(round(timescale))), '[{:4.0f}]'.format(t[cluster_death-f0]),
        else:
            print "cluster lives {:4.0f} > {:4.0f}".format((cluster_death-f0)/meta['fps'], t[-1]),

        ax.set_xlim(None, xmax)
    if stat == 'rad':
        ax.set_ylim(1, 14)
    ax.legend()
    ax.grid(True)
