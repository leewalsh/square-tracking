def correlation_xx(fitstr, form):
    print "====== <xx> ======"
    # Calculate the <xx> correlation for all the tracks in a given dataset
    fig, ax, taus, meancorr, errcorr = plot_msd(
        msds, msdids, args.dtau, args.dt0, data['f'].max()+1, save=False,
        show=False, tnormalize=0, errorbars=3, prefix=saveprefix, meancol=vcol,
        show_tracks=args.showtracks, lw=3, labels=labels, S=args.side,
        singletracks=args.singletracks, fps=args.fps, kill_flats=args.killflat,
        kill_jumps=args.killjump*args.side**2, title='' if args.clean else None,
        figsize=(5, 4) if args.clean else (8, 6))

    msdvec = {'displacement': 0, 'progression': 1, 'diversion': -1}[args.msdvec]
    if meancorr.ndim == 2:
        progression, diversion = meancorr.T
        msdisp = meancorr.sum(1)
        errcorr_prog, errcorr_div = errcorr.T
        errcorr_disp = np.hypot(*errcorr.T)
        ax.plot(taus, msdisp, '-', c=vcol, lw=3)
        meancorr = (msdisp, progression, diversion)[msdvec]
        errcorr = (errcorr_disp, errcorr_prog, errcorr_div)[msdvec]

    params = {}
    params['DT'] = meta.get('fit_xx_DT', 0.01)
    if args.fitv0 or not args.rn:
        params['v0'] = meta.get('fit_rn_v0', 0.1)
        sgn = np.sign(params['v0'])
    if not (args.nn or args.rn):
        params['DR'] = meta.get('fit_nn_DR', meta.get('fit_rn_DR', 1/16))

    tmax = 12*args.zoom/params['DR']    # in physical units

    if verbose > 1:
        errax = errfig.axes[0]
        errax.set_yscale('log')
        errax.set_xscale('log')
    else:
        errax = False
    uncert = args.dx*rt2
    sigma = curve.sigma_for_fit(meancorr, errcorr, x=taus, plot=errax, const=uncert,
                                ignore=[0, tmax],
                                xnorm=1,
                                verbose=verbose)
    if errax:
        errax.legend(loc='best', fontsize='x-small')
        if args.save:
            errfig.savefig(saveprefix+'_xx-corr_sigma.pdf')

    def limiting_regimes(s, DT=params['DT'], v0=params['v0'], DR=params['DR'], TR=0, msdvec=0):
        vv = v0*v0  # v0 is squared everywhere
        tau_T = DT/vv
        tau_R = 1/DR
        if tau_T > tau_R:
            return np.full_like(s, np.nan)
        taus = (tau_T, tau_R)

        early = 2*DT*s       # s < tau_T
        middle = vv*s**2         # tau_T < s < tau_R
        late = 2*(vv/DR + DT)*s               # tau_R < s
        lines = np.choose(np.searchsorted(taus, s), [early, middle, late])

        taus_f = np.clip(np.searchsorted(s, taus), 0, len(s)-1)
        lines[taus_f] = np.nan
        return lines

    # set free parameters
    vary = {'tau_xi': args.fittr or (args.colored and not (args.nn or args.rn)),
            'DR': args.fitdr or not (args.nn or args.rn),
            'v0': args.fitv0 or not args.rn,
            'DT': True}

    model = fit.Model(form)
    for param in vary:
        model.set_param_hint(param, min=0, vary=vary[param])
    result = model.fit(meancorr, s=taus, weights=1/sigma)

    print "Fits to <xx> (free params:", ', '.join(result.var_names)+'):'
    fit_source['DT'] = 'xx'
    print '   D_T: {:.3g}'.format(result.best_values['DT'])
    fitinfo = sf("$D_T={0:.3T}$", result.best_values['DT'])
    if vary['v0']:
        v0 = result.best_values['v0']
        fit_source['v0'] = 'xx'
        print 'v0(xx): {:.3g}'.format(result.best_values['v0'])
        fitinfo += sf(", $v_0={0:.3T}$", result.best_values['v0']*sgn)
    if vary['DR']:
        fit_source['DR'] = 'xx'
        print '   D_R: {:.3g}'.format(result.best_values['DR'])
        fitinfo += sf(", $D_R={0:.3T}$", result.best_values['DR'])
    if vary['tau_xi']:
        tau_R = result.best_values['tau_xi']
        fit_source['tau_xi'] = 'xx'
        print ' tau_R: {:.3g}'.format(tau_R)
        fitinfo += sf(r", $\tau_R={0:.3T}$", tau_R)
    if vary['v0'] or vary['DR']:
        print "Giving:"
        print "v0/D_R: {:.3g}".format(result.best_values['v0']/result.best_values['DR'])
    if args.save:
        if vary['v0'] and vary['DR']:
            psources = ''
            meta_fits = {'fit_xx_DT': result.best_values['DT'],
                         'fit_xx_v0': result.best_values['v0'],
                         'fit_xx_DR': result.best_values['DR']}
        elif vary['v0']:
            psources = '_{DR}'.format(**fit_source)
            meta_fits = {'fit'+psources+'_xx_DT': result.best_values['DT'],
                         'fit'+psources+'_xx_v0': result.best_values['v0']}
        else:
            psources = '_{DR}_{v0}'.format(**fit_source)
            meta_fits = {'fit'+psources+'_xx_DT': result.best_values['DT']}
        helpy.save_meta(saveprefix, meta_fits)

    label = ''.join(fitstr) if labels else ''

    best_fit = result.eval(s=taus)
    ax.plot(taus, best_fit, c=pcol, lw=2, label=label+fitinfo)

    guide = limiting_regimes(taus, **result.best_values)
    ax.plot(taus, guide, '-k', lw=2)

    ylim = ax.set_ylim(min(best_fit[0], meancorr[0]), best_fit[-1])
    xlim = ax.set_xlim(taus[0], tmax)
    if verbose > 1:
        errax.set_xlim(taus[0], taus[-1])
        map(errax.axvline, xlim)
    ax.legend(loc='upper left')

    tau_T = result.best_values['DT']/result.best_values['v0']**2
    tau_R = 1/result.best_values['DR']
    if xlim[0] < tau_T < xlim[1]:
        ax.axvline(tau_T, 0, 1/3, ls='--', c='k')
        ax.text(tau_T, 2e-2, ' $D_T/v_0^2$')
    if xlim[0] < tau_R < xlim[1]:
        ax.axvline(tau_R, 0, 2/3, ls='--', c='k')
        ax.text(tau_R, 2e-1, ' $1/D_R$')

    if args.save:
        save = saveprefix + psources + '_xx-corr.pdf'
        print 'saving <xx> correlation plot to',
        print save if verbose else os.path.basename(save)
        fig.savefig(save)
