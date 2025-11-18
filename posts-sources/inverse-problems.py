# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: kernelspec,jupyter,nikola
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: firedrake
#     language: python
#     name: firedrake
#   nikola:
#     category: ''
#     date: 2020-08-17 19:45:59 UTC-08:00
#     description: ''
#     link: ''
#     slug: inverse-problems
#     tags: ''
#     title: Inverse problems
#     type: text
# ---

# %% [markdown]
# In previous posts, we've seen how to solve elliptic PDE, sometimes with constraints, assuming we know everything about the coefficients and other input data.
# Some problems in geophysics and engineering involve going backwards.
# We have direct measurements of some field that we know is the solution of a PDE, and from that data we want to estimate what the coefficients were.
# This is what's called an *inverse problem*.
# For example, knowing the inflow rate of groundwater and the degree to which the soil and bedrock are porous, we can calculate what the hydraulic head will be by solving the Poisson equation; this is the forward problem.
# The inverse problem would be to estimate the porosity from measurements of the hydraulic head.
#
# We've already seen many of the techniques that we'll use to solve inverse problems and in this post I'll demonstrate them.
# Inverse problems can be expressed through PDE-constrained optimization, and the biggest challenge is calculating the gradient of the objective functional with respect to the input parameters.
# There's a systematic and practical algorithm to do this called the [adjoint method](https://journals.ametsoc.org/bams/article/78/11/2577/55799/What-Is-an-Adjoint-Model).
# The UFL language for variational forms preserves enough of the high-level semantics of what problem you're solving, and consequently it's possible to generate all of the code necessary to implement the adjoint method solely from the code for the weak form.
# The package [pyadjoint](http://www.dolfin-adjoint.org/en/latest/) does this and even won a Wilkinson Prize for numerical software.
# In the following, I'll use pyadjoint to both calculate derivatives and solve optimization problems, but it's instructive to roll your own adjoint method and solvers if you haven't done it before.

# %% [markdown]
# ### The problem
#
# Suppose that the physics we're interested in can be described by the Poisson problem.
# We want to estimate is the conductivity coefficient and we have measurements of the solution $u$.
# Rather than solve for the conductivity $K$ itself, I'll instead assume that the field $q$ that we want to infer is the logarithm of the conductivity:
#
# $$K = ke^q,$$
#
# where $k$ is some real constant.
# The reason for this change of variables is to guarantee that the conductivity is positive, a necessary condition which can be challenging to enforce through other means.
# For our problem, we'll include some internal sources $f$.
# By way of boundary conditions, we'll assume that the solution is adjusts with some exchange coefficient $h$ to an external field $g$ (these are *Robin* boundary conditions).
# The weak form of this equation is
#
# $$\begin{align}
# \langle F(u, q), v\rangle = & \int_\Omega\left(ke^q\nabla u\cdot\nabla v - fv\right)dx \\
# & \qquad\qquad + \int_{\partial\Omega}h(u - g)v\, ds
# \end{align}$$
#
# I'll assume that we know the sources, external field, and exchange coefficient accurately.
# The quantity that we want to minimize is the mean-square misfit of the solution $u$ with some observations $u^o$:
#
# $$E(u) = \frac{1}{2}\int_\Omega\left(\frac{u - u^o}{\sigma}\right)^2dx,$$
#
# where $\sigma$ is the standard deviation of the measurement errors in $u^o$.
# For realistic problems we might want to consider more robust measures of solution quality, like the 1-norm, but for demonstrative purposes the square norm is perfectly fine.
#
# To make our problem as realistic as possible, we'll create a set of synthetic observations that's been polluted from the true value with random noise.
# The presence of noise introduces an additional challenge.
# The map from the parameters $q$ to the observations $u$ involves solving an elliptic PDE and thus tends to give an output field $u$ that is smoother than the input field $q$.
# (You can actually write down an analytical form of the linearization of this map that makes the smoothing property evident.)
# For many practical problems, however, the measurement errors are spatial white noise, which have equal power at all frequencies.
# If we put white noise through the inverse of a smoothing operator, we'll end up amplifying the high-frequency modes and the estimated field $q$ will be polluted with spurious osillations.
# To remove these unphysical features, we'll also include some metric of how oscillatory the inferred field is, which in our case will be
#
# $$R(q) = \frac{1}{2}\int_\Omega|\nabla q|^2dx.$$
#
# This is called the **regularization functional**.
# Depending on the problem you may want to use a different regularization functional, and at the end of this post I'll give an example of when you might want to do that.

# %% [markdown]
# ### All together now
#
# The quantity we want to minimize is the functional
#
# $$J(u, q) = E(u) + \alpha^2 R(q),$$
#
# subject to the constraint that $u$ and $q$ are related by the PDE, which we'll write in abstract form as $F(u, q) = 0$.
# The parameter $\alpha$ is a length scale that determines how much we want to regularize the inferred field.
# Making a good choice of $\alpha$ is a bit of an art form best left for another day; in the following demonstration I'll pick a reasonable value and leave it at that.
# The adjoint method furnishes us with a way to calculate the derivative of $J$, which will be an essential ingredient in any minimization algorithm.
#
# To be more explicit about enforcing those constraints, we can introduce a Lagrange multiplier $\lambda$.
# We would then seek a critical point of the Lagrangian
#
# $$L(u, q, \lambda) = E(u) + \alpha^2 R(q) + \langle F(u, q), \lambda\rangle.$$
#
# By first solving for $u$ and then for the adjoint state $\lambda$, we can effectively calculate the derivative of our original objective with respect to the parameters $q$.
# Under the hood, this is exactly what pyadjoint and (more generally) reverse-mode automatic differentiation does.
# The interface that pyadjoint presents to us hides the existence of a Lagrange multiplier and instead gives us only a *reduced* functional $\hat J(q)$.

# %% [markdown]
# ### Generating the exact data
#
# First, we'll need to make a domain and some synthetic input data, which consist of:
#
# * the sources $f$
# * the external field $g$
# * the exchange coefficient $h$
# * the true log-conductivity field $q$
#
# We have to be careful about what kind of data we use in order to make the problem interesting and instructive.
# Ideally, the the true log-conductivity field will give a solution that's very different from some kind of blunt, spatially constant initial guess.
# To do this, we'll first make the external field $g$ a random trigonometric polynomial.

# %%
import firedrake
mesh = firedrake.UnitSquareMesh(32, 32, diagonal='crossed')
Q = firedrake.FunctionSpace(mesh, family='CG', degree=2)
V = firedrake.FunctionSpace(mesh, family='CG', degree=2)

# %%
import numpy as np
from numpy import random, pi as π
x = firedrake.SpatialCoordinate(mesh)

rng = random.default_rng(seed=1)
def random_fourier_series(std_dev, num_modes, exponent):
    from firedrake import sin, cos
    A = std_dev * rng.standard_normal((num_modes, num_modes))
    B = std_dev * rng.standard_normal((num_modes, num_modes))
    return sum([(A[k, l] * sin(π * (k * x[0] + l * x[1])) +
                 B[k, l] * cos(π * (k * x[0] + l * x[1])))
                / (1 + (k**2 + l**2)**(exponent/2))
                for k in range(num_modes)
                for l in range(int(np.sqrt(num_modes**2 - k**2)))])


# %%
g = firedrake.Function(V).interpolate(random_fourier_series(1.0, 6, 1))

# %%
import matplotlib.pyplot as plt
firedrake.trisurf(g);

# %% [markdown]
# Next, we'll make the medium much more insulating (lower conductivity) near the center of the domain.
# This part of the medium will tend to soak up any sources much more readily than the rest.

# %%
from firedrake import inner, min_value, max_value, Constant
a = -Constant(8.)
r = Constant(1/4)
ξ = Constant((0.4, 0.5))
expr = a * max_value(0, 1 - inner(x - ξ, x - ξ) / r**2)
q_true = firedrake.Function(Q).interpolate(expr)

# %%
firedrake.trisurf(q_true);

# %% [markdown]
# In order to make the effect most pronounced, we'll stick a blob of sources right next to this insulating patch.

# %%
b = Constant(6.)
R = Constant(1/4)
η = Constant((0.7, 0.5))
expr = b * max_value(0, 1 - inner(x - η, x - η) / R**2)
f = firedrake.Function(V).interpolate(expr)

# %%
firedrake.trisurf(f);

# %% [markdown]
# Once we pick a baseline value $k$ of the conductivity and the exchange coefficient $h$, we can compute the true solution.
# We'll take the exchange coefficient somewhat arbitrarily to be 10 in this unit system because it makes the results look nice enough.

# %%
from firedrake import exp, grad, dx, ds
k = Constant(1.)
h = Constant(10.)
u_true = firedrake.Function(V)
v = firedrake.TestFunction(V)
F = (
    (k * exp(q_true) * inner(grad(u_true), grad(v)) - f * v) * dx +
    h * (u_true - g) * v * ds
)

# %%
opts = {
    'solver_parameters': {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }
}
firedrake.solve(F == 0, u_true, **opts)

# %%
firedrake.trisurf(u_true);

# %% [markdown]
# The true value of $u$ has a big hot spot in the insulating region, just as we expect.

# %% [markdown]
# ### Generating the observational data
#
# For realistic problems, what we observe is the true solution plus some random noise $\xi$:
#
# $$u_\text{obs} = u_\text{true} + \xi.$$
#
# The ratio of the variance $\sigma$ of the noise to some scale of the solution, e.g. $\max_\Omega u_\text{true} - \min_\Omega u_\text{true}$, will determine the degree of accuracy that we can expect in the inferred field.
#
# To make this experiment more realistic, we'll synthesize some observations by adding random noise to the true solution.
# We'll assume that the noise is spatially white, i.e. the covariance of the measurement errors is
#
# $$\mathbb{E}[\xi(x)\xi(y)] = \sigma^2\delta(x - y)$$
#
# where $\delta$ is the Dirac delta distribution.
# A naive approach would be to add a vector of normal random variables to the finite element expansion coefficients of the true solution, but this will fail for a subtle reason.
# Suppose that, at every point, the measurement errors $\xi$ are normal with mean 0 and variance $\sigma$.
# Letting $\mathbb{E}$ denote statistical expectation, we should then have by Fubini's theorem that
#
# $$\mathbb{E}\left[\int_\Omega\xi(x)^2dx\right] = \int_\Omega\mathbb{E}[\xi(x)^2]dx = \sigma^2\cdot|\Omega|.$$
#
# The naive approach to synthesizing the noise will give us the wrong value of the area-averaged variance.

# %%
ξ = firedrake.Function(V)
n = len(ξ.dat.data_ro)
ξ.dat.data[:] = rng.standard_normal(n)

firedrake.assemble(ξ**2 * dx)

# %% [markdown]
# The "right" thing to do is:
#
# 1. Compute the finite element mass matrix $M$
# 2. Compute the Cholesky factorization $M = LL^*$
# 3. Generate a standard normal random vector $z$
# 4. The finite element expansion coefficients for the noise vector are
#
# $$\hat\xi = \sigma\sqrt{\frac{|\Omega|}{n}}L^{-*}z.$$
#
# You can show that this works out correctly by remembering that
#
# $$\int_\Omega\xi^2dx = \hat\xi^*M\hat\xi.$$
#
# We'll have to do a bit of hacking with PETSc data structures directly in order to pull out one of the Cholesky factors of the mass matrix.

# %%
from firedrake.petsc import PETSc
ϕ, ψ = firedrake.TrialFunction(V), firedrake.TestFunction(V)
m = inner(ϕ, ψ) * dx
M = firedrake.assemble(m, mat_type='aij').M.handle
ksp = PETSc.KSP().create()
ksp.setOperators(M)
ksp.setUp()
pc = ksp.pc
pc.setType(pc.Type.CHOLESKY)
pc.setFactorSolverType(PETSc.Mat.SolverType.PETSC)
pc.setFactorSetUpSolverType()
L = pc.getFactorMatrix()
pc.setUp()

# %% [markdown]
# Since our domain is the unit square, it has an area of 1, but for good measure I'll include this just to show the correct thing to do.

# %%
area = firedrake.assemble(Constant(1) * dx(mesh))

# %%
z = firedrake.Function(V)
z.dat.data[:] = rng.standard_normal(n)
with z.dat.vec_ro as Z:
    with ξ.dat.vec as Ξ:
        L.solveBackward(Z, Ξ)
        Ξ *= np.sqrt(area / n)

# %% [markdown]
# The error statistics are within spitting distance of the correct value of 1.

# %%
firedrake.assemble(ξ**2 * dx) / area

# %% [markdown]
# The answer isn't exactly equal to one, but averaged over a large number of trials or with a larger mesh it will approach it.
# Finally, we can make the "observed" data.
# We'll use a signal-to-noise ratio of 50, but it's worth tweaking this value and seeing how the inferred parameters change.

# %%
û = u_true.dat.data_ro[:]
signal = û.max() - û.min()
signal_to_noise = 50
σ = firedrake.Constant(signal / signal_to_noise)

u_obs = u_true.copy(deepcopy=True)
u_obs += σ * ξ

# %% [markdown]
# The high-frequency noise you can see in the plot below is exactly what makes regularization necessary.

# %%
firedrake.trisurf(u_obs);

# %% [markdown]
# ### Calculating derivatives
#
# Now we can import firedrake-adjoint.
# Under the hood, this will initialize the right data structures to calculate derivatives using the adjoint method, and we can even take a peek at those data structures.

# %%
import firedrake.adjoint
firedrake.adjoint.continue_annotation()

# %% [markdown]
# We'll start with a fairly neutral initial guess that the log-conductivity $q$ is identically 0.

# %%
q = firedrake.Function(Q)
u = firedrake.Function(V)
F = (
    (k * exp(q) * inner(grad(u), grad(v)) - f * v) * dx +
    h * (u - g) * v * ds
)
firedrake.solve(F == 0, u, **opts)

# %% [markdown]
# The computed solution with a constant conductivity doesn't have the gigantic spike in the insulating region, so it's very easy to tell them apart.
# When the differences are really obvious it makes it easier to benchmark a putative solution procedure.

# %%
firedrake.trisurf(u);

# %% [markdown]
# Just to give a sense of how different the initial value of the observed field is from the true value, we can calculate the relative difference in the 2-norm:

# %%
print(firedrake.norm(u - u_true) / firedrake.norm(u_true))

# %% [markdown]
# Now we can start having some fun with Firedrake's adjoint capabilities.
# A lot of what we're going to do can seem like magic and I often find it a little bewildering to have no idea what's going on under the hood.
# Much of this machinery works by overloading functionality within Firedrake and recording operations to a *tape*.
# The tape can then in effect be played backwards to perform reverse-mode automatic differentiation.
# You can access the tape explicitly from the Firedrake adjoint API, which conveniently provides functions to visualise the tape using [graphviz](https://graphviz.org/) or [NetworkX](https://networkx.org).
# The plot below shows the overall connectivity of the structure of the tape; you can query the nodes using NetworkX to get a better idea of what each one represents.
# This tape will grow and grow as we calculate more things and it's a common failure mode for an adjoint calculation to eat up all the system memory if you're not careful.

# %%
import networkx
tape = firedrake.adjoint.get_working_tape()
graph = tape.create_graph(backend='networkx')
fig, axes = plt.subplots()
networkx.draw_kamada_kawai(graph, ax=axes);

# %% [markdown]
# Hopefully this gives you some sense of how all this machinery works at a lower level.
# For more details you can see the [dolfin-adjoint](http://www.dolfin-adjoint.org/en/latest/) documentation, which has loads of commentary on both the math and the code by its author, Patrick Farrell.
#
# To start on solving the inverse problem, we're going to declare that $q$ is the *control variable*, i.e. it's the thing that want to optimize over, as opposed to the field $u$ that we can observe.

# %%
q̂ = firedrake.adjoint.Control(q)

# %% [markdown]
# Next we'll create the objective functional, which measures both the degree to which our computed solution $u$ differs from the true solution and the oscillations in our guess $q$.
# Normally, we might create a symbolic variable (a Firedrake `Form` type) that represents this functional.
# If we wanted to get an actual number out of this symbolic object, we would then call `assemble`.
# So it might stick out as unusual that we're assembling the form right away here.

# %%
α = Constant(5e-2)
J = firedrake.assemble(
    0.5 * ((u - u_obs) / σ)**2 * dx +
    0.5 * α**2 * inner(grad(q), grad(q)) * dx
)

# %% [markdown]
# In fact there's a bit of magic going under the hood; `J` isn't really a floating point number, but a more complex object defined within the pyadjoint package.
# The provenance of how this number is calculated is tracked through the adjoint tape.

# %%
print(type(J))

# %% [markdown]
# We can get an actual number out of this object by casting it to a `float`.

# %%
print(float(J))

# %% [markdown]
# The advantage of having this extra layer of indirection is that, as the control variable $q$ changes, so does $J$ and firedrake-adjoint will track the sensitivity under the hood for you.
# The next step is to somehow wire up this functional with the information that $u$ isn't really an independent variable, but rather a function of the control $q$.
# This is what the `ReducedFunctional` class does for us.

# %%
Ĵ = firedrake.adjoint.ReducedFunctional(J, q̂)

# %% [markdown]
# The reduced functional has a method to calculate its derivative with respect to the control variable.

# %%
dĴ_dq = Ĵ.derivative()

# %% [markdown]
# This method call is hiding some subtleties that are worth unpacking.
# The reduced functional $\hat J$ is a differentiable mapping of the function space $Q$ into the real numbers.
# The derivative $d\hat J/dq$ at a particular value of the control variable is an element of the dual space $Q^*$.
# As mathematicians, we grow accustomed to thinking of Hilbert spaces as being isometric to their duals.
# It's easy to forget that isometric does not mean identical.
# The mapping between the primal and dual spaces can be non-trivial.
# For example, suppose $Q$ is the Sobolev space $H^1(\Omega)$.
# The dual space $H^{-1}(\Omega)$ is isometric to the primal, *but* to evaluate the mapping between them, we have to solve an elliptic PDE.
#
# The Sobolev space $H^1(\Omega)$ is a relatively tame one in the grand scheme of things.
# Real problems might involve controls in Banach spaces with no inner product structure at all.
# For example, the conductivity coefficient has to be bounded and positive, so we're probably looking in some cone in the space $L^\infty(\Omega)$.
# In general, conductivity fields can be discontinuous, although not wildly so.
# We might then want to look in the intersection of $L^\infty$ with the space [$BV(\Omega)$](https://en.wikipedia.org/wiki/Bounded_variation) of functions whose first derivatives are finite signed measures.
#
# Nonetheless, the discretization via finite elements can obscure the distinction between the primal and dual spaces.
# The control $q$ and the derivative $d\hat J/dq$ contain within them a wad of data that happens to look the same: an array of floating point numbers, the size of which is equal to the number of vertices + the number of edges of the mesh for our P2 discretization.
# What's confusing is that these numbers don't mean the same thing.
# The array living under $q$ represents its coefficients in the finite element basis for the space $Q$, while the array for $d\hat J/dq$ represents its coefficients in the *dual* basis.
# To get the action of $d\hat J/dq$ on some perturbation field $\phi$, we take the (Euclidean) dot product of the wads of data living underneath them.
# This is in distinct contrast to getting the inner product in, say, $L^2(\Omega)$ of $\phi$ with another function $\psi$, where the inner product is instead calculated using the finite element mass matrix.
#
# So, where does that leave us?
# We need some way of mapping the dual space $Q^*$ back to the primal.
# This mapping is referred to in the literature as the **Riesz map** after the Riesz representation theorem.
# The laziest way we could possibly do so is to multiply $d\hat J/dq$ by the inverse of the finite element mass matrix.
# Maybe we should instead use a 2nd-order elliptic operator.
# After all, we assumed that the controls live in an $H^1$-conforming space.
# But for illustrative purposes the mass matrix will do fine.
#
# By default, Firedrake gives you the raw value of the derivative, which lives in the dual space.
# We can specify that we want to apply the Riesz map by adding another argument.
# We can see the difference in the return types.

# %%
print(type(dĴ_dq))
print(type(Ĵ.derivative(apply_riesz=True)))

# %% [markdown]
# The first object is not a `Function` but rather a `Cofunction`, an element of the dual space.
#
# Keeping track of which quantities live in the primal space and which live in the dual space is one of the challenging parts of solving PDE-constrained optimization problems.
# Most publications on numerical optimization assume the problem is posed over Euclidean space.
# In that setting, there's no distinction between primal and dual.
# You can see this bias reflected in software packages that purport to solve numerical optimization problems.
# Almost none of them have support for supplying a matrix other than the identity that defines the dual pairing.
# The fact that a Sobolev space isn't identical to its dual has some unsettling consequences.
# For starters, the gradient descent method doesn't make sense over Sobolev spaces.
# If you can rely on the built-in optimization routines from pyadjoint, you'll largely be insulated from this problem.
# But if you've read this far there's a good chance that you'll have to roll your own solvers at some point in your life.
# To paraphrase the warning at gate of Plato's academy, let none ignorant of duality enter there.

# %% [markdown]
# ### Solving the inverse problem
#
# Ok, screed over.
# Let's do something useful now.
# The firedrake-adjoint package contains several routines to minimize the reduced objective functional.
# Here we'll call out to scipy.
# Let's see how well we can recover the log-conductivity field.

# %%
q_opt = firedrake.adjoint.minimize(Ĵ, method="Newton-CG")

# %%
firedrake.trisurf(q_opt);

# %% [markdown]
# The optimization procedure has correctly identified the drop in the conductivity of the medium to within our smoothness constraints.
# Nonetheless, it's clear in the eyeball norm that the inferred field doesn't completely match the true one.

# %%
firedrake.norm(q_opt - q_true) / firedrake.norm(q_true)

# %% [markdown]
# What's a little shocking is the degree to which the computed state matches observations despite these departures.
# If we plot the computed $u$, it looks very similar to the true value.

# %%
q.assign(q_opt)
firedrake.solve(F == 0, u, **opts)

# %%
firedrake.trisurf(u);

# %% [markdown]
# Moreover, if we compute the model-data misfit and weight it by the standard deviation of the measurement errors, we get a value that's roughly around 1/2.

# %%
firedrake.assemble(0.5 * ((u - u_obs) / σ)**2 * dx)

# %% [markdown]
# This value is about what we would expect from statistical estimation theory.
# Assuming $u$ is an unbiased estimator for the true value of the observable state, the quantity $((u - u^o) / \sigma)^2$ is a $\chi^2$ random variable.
# When we integrate over the whole domain and divide by the area (in this case 1), we're effectively summing over independent $\chi^2$ variables and so we should get a value around 1/2.
#
# Recall that we used a measurement error $\sigma$ that was about 2\% of the true signal, which is pretty small.
# You can have an awfully good signal-to-noise ratio and yet only be able to infer the conductivity field to within a relative error of 1/4.
# These kinds of synthetic experiments are really invaluable for getting some perspective on how good of a result you can expect.
