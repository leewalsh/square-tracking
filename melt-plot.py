save = False

params = dict(
    stat = [
#         'phi', 'psi', 'dens', 'clust',
#         'phi', 'psi',
        'dens', 'clust',
#         'clust'
#         'phi/dens', 'psi/dens', 'psi/phi',
#         'phi/dens',
#         'phi/psi',
    ],
    W = range(7, 12),
    arrange = ('in', 'rand', ''),
)
rescale = False and ('phi' in params['stat'] or 'psi' in params['stat'])

columns_by = 'W'
rows_by = 'stat'
color_by = 'arrange'

nruns = 5  # maximum number of runs shown, 0 for MRG only

save_name = '_'.join(filter(None, [
    'parametric'*any('/' in stat for stat in params['stat']),
    'decays'*any('/' not in stat for stat in params['stat']),
    'timescaled'*rescale,
    'by_{}'.format({'W': 'size', 'arrange': 'config'}[color_by]),
    str(params[columns_by][0]) if len(params[columns_by]) == 1 else None,
    'MRG'*(nruns == 0),
]))

ncols = len(params[columns_by])
nrows = len(params[rows_by])

figsize = np.array(plt.rcParams['figure.figsize'])
if all('/' in stat for stat in params['stat']):
    figsize = figsize.mean(keepdims=True)
figsize = figsize * [ncols, nrows]

fig, axeses = plt.subplots(
    figsize=figsize, squeeze=False,
    ncols=ncols, nrows=nrows,
    #sharex='row', sharey='row',
    dpi=120,
)

square_stats = ('phi', 'psi', 'dens')

# loops must go in order: config, run, stat;
# config may be two loops (two criteria: size & arrangement, e.g.)
# but can loop through axes (cols, rows) in any order.

# loop data through config_filters
# loop axes through columns
for col_by, axes_col in zip(params[columns_by], axeses.T):
    print '\n' + '='*30 + 'NEW COLUMN'.center(12) + '='*30
    axes_col[0].set_title(labels[col_by])
    config_filter = {'particle': ['pdms', 'lego'], columns_by: col_by}
    config_timescales = filter_timescales(**config_filter)
    times = helpy.consecutive_fields_view(config_timescales, ['phi', 'psi'])
    tmax = 2.5 * np.percentile(times, 80)

    # loop data through configs
    for config in filter_configs(config_filter):
        print config,
        runs = range(1, 1+min(config['runs'], nruns)) + ['MRG']

        # loop data through runs
        for run in runs:
            timescale = filter_timescales(run=run, **config)
            print str(run).center(3),
            loaded = load_melting_stuff(run=run, **config)
            prefix, meta, mdata, mframes, means = loaded
            plot_args = melt.make_plot_args(meta)

            plot_args['line_props'] = {
                'c': colors[config[color_by[0]]],
                'label': labels[config[color_by[0]]],
                'lw': 1.5,
                'alpha': 1,
            } if run == 'MRG' else {
                'c': colors[config[color_by[0]]],
                'label': None,
                'lw': 0.5,
                'alpha': 0.25,
            }

            # loop data through rows_by
            # loop axes through rows
            for row_val, ax in zip(params[rows_by], axes_col):
                #print '\n' + '-'*30 + 'NEW ROW'.center(12) + '-'*30
                y, x = (stat.split('/') + ['f'])[:2]
                smooth = xscale = 1
                xlabel = xlim = None
                if x == 'f':
                    xlabel = r'time $t \, f$'
                    end_index = None
                    xlim = None, tmax
                    if rescale and y in 'phi psi':
                        xscale = timescale[y]
                        xlabel = r'rescaled time $t / \tau_\{}$'.format(y)
                        smooth = max(1, xscale/50)
                        xlim = -0.1, 3.1
                elif x in square_stats and y in square_stats:
                    ax.set_aspect('equal', adjustable='box-forced')
                    end_index = int(1.2*(timescale['phi']+timescale['psi'])*meta['fps'])
                else:
                    raise ValueError('cannot plot {} vs {}'.format(y, x))

                plot_by_cluster(means['c'][:end_index], x, y, meta, ax=ax,
                                smooth=smooth, xscale=xscale, **plot_args)
                if xlim:
                    ax.set_xlim(xlim)
                if xlabel:
                    ax.set_xlabel(xlabel)
        print

fig.tight_layout(pad=1.1, h_pad=0, w_pad=0)
if save:
    save_name = savedir + save_name + '.pdf'
    print 'saving to', save_name
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0)

plt.show()
plt.close(fig)
