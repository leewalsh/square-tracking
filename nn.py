def correlation_xx(fitstr, form):
    print "====== <xx> ======"
    # Calculate the <xx> correlation for all the tracks in a given dataset

    corr_func = partial(corr.autocorr, cumulant=False, norm=False)
    if args.verbose:
        print 'calculating <xx> correlations'
    if args.dot:
        all_corrs = [corr_func(np.cos(ts['o'])) + corr_func(np.sin(ts['o']))
                     for ts in tracksets.itervalues()]
    else:
        all_corrs = [c for ts in tracksets.itervalues() for c in
                     [corr_func(np.cos(ts['o'])), corr_func(np.sin(ts['o']))]]

    all_corrs, meancorr, errcorr = helpy.avg_uneven(all_corrs, pad=True)
    taus = np.arange(len(meancorr))/args.fps

    params = {}
    params['DR'] = meta.get('fit_xx_DR', 1/16)
    if args.colored:
        params['tau_xi'] = meta.get('fit_xx_TR', 1/(16*params['DR']))
    else:
        params['tau_xi'] = 0

    tmax = 3*args.zoom/params['DR']     # in physical units

    if verbose > 1:
        errfig, errax = plt.subplots()
        errax.set_yscale('log')
    else:
        errax = False
    uncert = args.dtheta/rt2
    sigma = curve.sigma_for_fit(meancorr, errcorr, x=taus, plot=errax, const=uncert,
                                ignore=[0, tmax],
                                verbose=verbose)
    if errax:
        errax.legend(loc='best', fontsize='x-small')
        if args.save:
            errfig.savefig(saveprefix+'_xx-corr_sigma.pdf')

    # set free parameters
    vary = {'tau_xi': args.colored,
            'DR': True}

    model = fit.Model(form)
    for param in vary:
        model.set_param_hint(param, min=0, vary=vary[param])
    result = model.fit(meancorr, s=taus, weights=1/sigma)

    best_fit = result.eval(s=taus)
    print "Fits to <xx> (free params:", ', '.join(result.var_names)+'):'
    for p in result.var_names:
        print '{:6}: {:.4g}'.format(p, result.best_values[p])
        fit_source[p] = 'xx'
        meta_fits = {'fit_xx_'+p: result.best_values[p]}
    if args.save:
        helpy.save_meta(saveprefix, meta_fits)

    fig, ax = plt.subplots(figsize=(5, 4) if args.clean else (8, 6))
    if args.showtracks:
        ax.plot(taus, all_corrs.T, 'b', alpha=.2, lw=0.5)
    ax.errorbar(taus, meancorr, errcorr, None, c=vcol, lw=3,
                label="Mean Orientation Autocorrelation"*labels,
                capthick=0, elinewidth=1, errorevery=3)
    fitinfo = sf("$D_R={0:.4T}=1/{1:.3T}$", result.best_values['DR'], 1/result.best_values['DR'])
    if args.colored:
        fitinfo += sf(", $\\tau_R={0:.4T}$", result.best_values['tau_xi'])
    ax.plot(taus, best_fit, c=pcol, lw=2,
            label=labels*(fitstr + '\n') + fitinfo)
    tmax = 3*args.zoom/result.best_values['DR'] + result.best_values['tau_xi']
    ax.set_xlim(0, tmax)
    ax.set_ylim(exp(-3*args.zoom), 1)
    ax.set_yscale('log')

    if labels:
        ax.set_title("Orientation Autocorrelation\n"+prefix)
    ax.set_ylabel(r"$\langle \hat n(t) \hat n(0) \rangle$")
    ax.set_xlabel("$tf$")
    ax.legend(loc='upper right' if args.zoom <= 1 else 'lower left',
              framealpha=1)

    if args.save:
        save = saveprefix + '_xx-corr.pdf'
        print 'saving <xx> correlation plot to',
        print save if verbose else os.path.basename(save)
        fig.savefig(save)
