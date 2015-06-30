

If we make certain assumptions about the immediate motion (translational and rotational) of a particle, we can calculate various statistical behaviors of the particle over long times. This allows the model to be used in simulations and calculations for collective systems as well as individual particles.

These assumptions, in our case, are as follows.
-----------------------------------------------

1) The velocity of the particle at a given time is some constant in the direction of the particle's orientation plus a translational noise: $\vec v(t) = v_0 \hat n(t) + \vec\eta(t)$. The noise term is drawn from a normal (Gaussian) distribution with no correlation over time. The mean of the distribution is zero, and its variance is $D_T$ (the translational coefficient of diffusion).

2) The rotational speed has no constant component but it simply a noise term: $\omega(t) = \xi(t)$. Where the noise has zero mean and variance $D_R$.

We can check these assumptions in two distinct ways:
----------------------------------------------------

1) Calculate out the statistical behavior of a particle over time, averaged over all the possible values of the noise, and compare that behavior to the measured statistics of an experimental particle, averaged over several different particles at several different times.

2) Measure the velocity of the particle, extract the noise term by removing the constant component, and measure the three noise assumptions: zero mean, variance of D, and no time correlations.
