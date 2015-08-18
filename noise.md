
Noise in vibrated particles
===========================

If we make certain assumptions about the immediate motion (translational and rotational) of a particle, we can calculate various statistical behaviors of the particle over long times. This allows the model to be used in simulations and calculations for collective systems as well as individual particles.

These assumptions, in our case, are as follows.
-----------------------------------------------

1) The velocity of the particle at a given time is some constant in the direction of the particle's orientation plus a translational noise: $\vec v(t) = v_0 \hat n(t) + \vec\eta(t)$. The noise term is drawn from a normal (Gaussian) distribution with no correlation over time. The mean of the distribution is zero, and its variance is $D_T$ (the translational coefficient of diffusion).

2) The rotational speed has no constant component but it simply a noise term: $\omega(t) = \xi(t)$. Where the noise has zero mean and variance $D_R$.

We can check these assumptions in two distinct ways:
----------------------------------------------------

1) Calculate out the statistical behavior of a particle over time, averaged over all the possible values of the noise, and compare that behavior to the measured statistics of an experimental particle, averaged over several different particles at several different times. These statistics, in our case, are $\langle \hat n(0) \hat n(t) \rangle$, $\langle \hat n(0) \vec r(t) \rangle$, $\langle \vec r(0) \vec r(t) \rangle$.

This has been discussed in more detail elsewhere [e.g., Balasubramanian, Baskaran].

2) Measure the velocity of the particle, extract the noise term by removing the constant component in the parallel direction, and measure the three noise assumptions: zero mean, variance of D, and no time correlations.

To measure the velocity, we take a derivative of the position and orientation of the particle over time. Derivative is taken as the convolution with the derivative of a gaussian kernel, which is equal to the convolution of the derivative with a gaussian kernel, i.e., a smoothed derivative. We convert step sizes to velocity via $v_i = \frac{\Delta x_i}{\Delta t}$.

Before extracting noise statistics, we must first subtract the constant active velocity from the trajectory. For translational motion, we must combine the two Cartesian components to find $v_0$ then separate them to subtract the activity. To find its magnitude $v_0$, we use^[but perhaps we should instead use $v_0 = \langle \vec v \cdot \hat n \rangle$?]
$$v_0^2 = \langle \vec v^2 \rangle$$
where $\vec v^2 = \vec v \cdot \vec v = v_x^2 + v_y^2$. We must subtract it from the velocity as a vector $\vec v_0 = v_0 \hat n$, where $\hat n = (\cos\theta, \sin\theta)$, i.e., $v_0^x = v_0 \cos\theta$ and $v_0^y = v_0 \sin\theta$. Thus we shall measure the statistics of $\delta v_i = v_i - v_0 \hat n_i$.

Rotational motion is one dimensional, so the mean can be calculated on $\omega = \Delta \theta / \Delta t$ itself: $\omega_0 = \langle \omega \rangle$. We then analyze the statistics of $\delta \omega = \omega - \omega_0$, which will be equivalent to using $\omega$ itself since $\omega_0$ is constant.

Therefore we measure the noises as $\vec \eta = \delta \vec v$ and $\xi = \delta \omega$. The means $\langle \delta v_i \rangle$ and $\langle \delta \omega \rangle$ are expected to be zero^[which raises the issue of how to independently confirm the zero mean of $\delta\omega$ in the case that $\omega_0 \neq 0$.] (by symmetry of the system), but the standard deviation should give the coefficients of diffusion as
$$D = \frac{1}{2} \sigma^2 \Delta t$$.

