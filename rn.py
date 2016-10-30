def correlation_xx(fitstr, form):
    print "====== <xx> ======"
    # Calculate the <xx> correlation for all the tracks in a given dataset

    corr_func = partial(corr.crosscorr, side='both', ret_dx=True,
                        cumulant=(True, False), norm=0)

    # shape (track, x_or_y, time_or_correlation, time)
    all_corrs = np.array([[corr_func(ts['x']/args.side, np.cos(ts['o'])),
                          corr_func(ts['y']/args.side, np.sin(ts['o']))]
                         for ts in tracksets.itervalues()])
    # Align and merge them
    taus = all_corrs[:, :, 0]/args.fps
    if all_corrs.ndim == 4:
        if verbose:
            print "Already aligned: all tracks have same length"
        taus = taus[0, 0]
        all_corrs = all_corrs[:, :, 1]
    else:
        if verbose:
            print "Aligning tracks around tau=0"
        tau0 = np.array(map(partial(np.searchsorted, v=0), taus.flat))
        taus = taus.flat[tau0.argmax()]
        all_corrs = helpy.pad_uneven(all_corrs[:, :, 1], np.nan, align=tau0)
    if args.dot:
        all_corrs = all_corrs.sum(1)  # sum over x and y components
    else:
        all_corrs = all_corrs.reshape(-1, len(taus))
    all_corrs, meancorr, errcorr, stddev, added, enough = helpy.avg_uneven(
        all_corrs, pad=False, ret_all=True, weight=False)
    taus = taus[enough]

    params = {}
    if not args.nn:
        params['DR'] = meta.get('fit_nn_DR', meta.get('fit_xx_DR', 1/16))
    v0 = meta.get('fit_xx_v0', 0.1)  # if dots on back, use v0 < 0
    params['lp'] = v0/params['DR']

    tmax = 3*args.zoom/params['DR']     # in physical units

    if verbose > 1:
        errfig, errax = plt.subplots()
    else:
        errax = False
    uncert = np.hypot(args.dtheta, args.dx)/rt2
    sigma = curve.sigma_for_fit(meancorr, errcorr, x=taus, plot=errax, const=uncert,
                                ignore=[0, -tmax, tmax],
                                verbose=verbose)
    if errax:
        errax.legend(loc='best', fontsize='x-small')
        if args.save:
            errfig.savefig(saveprefix+'_xx-corr_sigma.pdf')

    # set free parameters
    vary = {'tau_xi': args.fittr or (args.colored and not args.nn),
            'DR': args.fitdr or not args.nn,
            'lp': True}

    model = fit.Model(form)
    for param in vary:
        model.set_param_hint(param, min=0, vary=vary[param])
    result = model.fit(meancorr, s=taus, weights=1/sigma)

    print "Fits to <xx> (free params:", ', '.join(result.var_names)+'):'
    v0 = result.best_values['DR']*result.best_values['lp']
    fit_source['v0'] = 'xx'
    print ' v0/D_R: {:.4g}'.format(result.best_values['lp'])
    if vary['DR']:
        fit_source['DR'] = 'xx'
        print '    D_R: {:.4g}'.format(result.best_values['DR'])
    if vary['tau_xi']:
        fit_source['tau_xi'] = 'xx'
        print '  tau_R: {:.4g}'.format(result.best_values['tau_xi'])
    print "Giving:"
    print '     v0: {:.4f}'.format(v0)
    if args.save:
        if vary['DR']:
            psources = ''
            meta_fits = {'fit_xx_v0': v0,
                         'fit_xx_DR': result.best_values['DR']}
        else:
            psources = '_nn'
            meta_fits = {'fit'+psources+'_xx_v0': v0}
        if vary['tau_xi']:
            meta_fits = {'fit_xx_TR': result.best_values['tau_xi']}
        helpy.save_meta(saveprefix, meta_fits)

    fig, ax = plt.subplots(figsize=(5, 4) if args.clean else (8, 6))
    sgn = np.sign(v0)
    if args.showtracks:
        ax.plot(taus, sgn*all_corrs.T, 'b', alpha=.2, lw=0.5)
    ax.errorbar(taus, sgn*meancorr, errcorr, None, c=vcol, lw=3,
                label="Mean Position-Orientation Correlation"*labels,
                capthick=0, elinewidth=0.5, errorevery=3)
    fitinfo = sf('$v_0={0:.3T}$', abs(v0))
    if vary['DR']:
        fitinfo += sf(", $D_R={0:.3T}$", result.best_values['DR'])
    if vary['tau_xi']:
        fitinfo += sf(", $\\tau_R={0:.4T}$", result.best_values['tau_xi'])
    ax.plot(taus, sgn*result.best_fit, c=pcol, lw=2,
            label=labels*(fitstr + '\n') + fitinfo)

    ylim_buffer = 1.5
    ylim = ax.set_ylim(ylim_buffer*result.best_fit.min(),
                       ylim_buffer*result.best_fit.max())
    xlim = ax.set_xlim(-tmax, tmax)
    tau_R = 1/result.best_values['DR']
    if xlim[0] < tau_R < xlim[1]:
        ax.axvline(tau_R, 0, 2/3, ls='--', c='k')
        ax.text(tau_R, 1e-2, ' $1/D_R$')

    if labels:
        ax.set_title("Position - Orientation Correlation")
    ax.set_ylabel(r"$\langle \vec r(t) \hat n(0) \rangle / \ell$")
    ax.set_xlabel("$tf$")
    ax.legend(loc='upper left', framealpha=1)

    if args.save:
        save = saveprefix + psources + '_xx-corr.pdf'
        print 'saving <xx> correlation plot to',
        print save if verbose else os.path.basename(save)
        fig.savefig(save)
