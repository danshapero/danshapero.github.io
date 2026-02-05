# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
#     date: 2026-02-01 16:58:22 UTC-07:00
#     description: ''
#     link: ''
#     slug: plate-obstacle
#     tags: ''
#     title: Obstacle problems for plates
#     type: text
# ---

# %% [markdown]
# The equations of plate theory are to minimize the following functional for the displacement $w$:
# $$J(w) = \int_\Phi\left\{\frac{h^3}{24}\left(2\mu|\nabla ^2 w|^2 + \lambda|\Delta w|^2\right) + hfw\right\}\mathrm dx.$$
# I'll write this as
# $$J(w) = \int_\Omega\left(\frac{1}{2}\mathscr C\nabla^2w :\nabla^2 w + fw\right)h\,\mathrm dx$$
# where $\mathscr C$ is an elasticity tensor.
# We'll need the explicit form
# $$\mathscr C\kappa = \frac{h^2}{12}\left(2\,\mu\,\kappa + \lambda\,\text{tr}(\kappa)\,I\right)$$
# in a moment.
# We get the dual form by introducing the curvature $\sigma = \mathscr C\nabla^2 w$ explicitly as an unknown.
# Let $\mathscr A$ be the tensor inverse to $\mathscr C$.
# The explicit form of $\mathscr A$ is
# $$\mathscr A\sigma = \frac{6}{h^2\mu}\left(\sigma - \frac{\lambda}{2(\mu + \lambda)}\text{tr}(\sigma)I\right)$$
# in which case $\nabla^2 w = \mathscr A\sigma$.
# The dual form is to find a saddle point of
# $$L(w, \sigma) = \int_\Omega\left\{\frac{1}{2}\mathscr A\sigma : \sigma - w\left(\nabla\cdot\nabla\cdot \sigma - f\right)\right\}h\,\mathrm dx.$$
# Here the displacement $w$ acts like a multiplier enforcing the constraint that $\nabla\cdot\nabla\cdot\sigma + f = 0$.
#
# To get a conforming discretization of this problem, we need that $w$ has continuous derivatives, while $\sigma$ has no continuity requirements at all.
# We can get a working non-conforming discretization by making fewer assumptions about $w$ at the expense of more assumptions about $\sigma$.
# First, we'll use a conventional continuous finite element basis for $w$, without the assumption that its derivatives are continuous.
# For the curvatures, we'll assume that it has *normal-normal* continuity.
# We don't require $\sigma$ itself to be continuous across cell boundaries, only that $n\cdot\sigma n$ is continuous.
# In the expression for the Lagrangian above, we can push one divergence of $\sigma$ over as a derivative of $w$ with no complications.
# When we push the second divergence over, we get some extra boundary terms:
# $$\begin{align}
# L(w, \sigma) = & \sum_\omega\int_\omega\left\{\frac{1}{2}\mathscr A\sigma : \sigma - \nabla^2 w : \sigma + fw\right\}h\,\mathrm dx \\
# & \qquad + \sum_\gamma\int_\gamma (n\cdot \sigma n)\left[\!\!\left[\frac{\partial w}{\partial n}\right]\!\!\right]h\,\mathrm dS
# \end{align}$$
# where the sums are over all the cells $\omega$ and facets $\gamma$ of the finite element mesh.
#
# Here I want to see what happens when we try to solve a plate obstacle problem: we constrain the solution $w$ to lie above some obstacle $\psi$, or
# $$w \ge \psi.$$

# %% [markdown]
# ### Experiments
#
# First let's try and solve a plate problem using simple input data.
# We'll set all the physical constants equal to 1 but you can look these up for the material of your fancy.

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import firedrake
from firedrake import (
    exp, Constant, inner, tr, dot, grad, dx, ds, dS, avg, jump
)
import ufl

# %%
f = Constant(-10.0)
h = Constant(1.0)
μ = Constant(1.0)
λ = Constant(1.0)


# %% [markdown]
# The code below creates the dual form of the problem using the formula I wrote down above for the explicit form of the compliance tensor.
# Lord Jesus I hope I did all that algebra correct.

# %%
def form_hhj_lagrangian(z, f, h, μ, λ):
    mesh = ufl.domain.extract_unique_domain(z)
    w, σ = firedrake.split(z)[:2]
    I = firedrake.Identity(2)
    n = firedrake.FacetNormal(mesh)

    Aσ = 6 / (h**2 * μ) * (σ - λ / (2 * (μ + λ)) * tr(σ) * I)

    L_cells = (0.5 * inner(Aσ, σ) - inner(σ, grad(grad(w))) + f * w) * h * dx
    L_facets = avg(inner(n, dot(σ, n))) * jump(grad(w), n) * h * dS
    L_boundary = inner(n, dot(σ, n)) * inner(grad(w), n) * h * ds
    return L_cells + L_facets + L_boundary


# %% [markdown]
# The right degrees are $p + 1$ for the displacements and $p$ for the moments.

# %%
p = 0
cg = firedrake.FiniteElement("CG", "triangle", p + 1)
hhj = firedrake.FiniteElement("HHJ", "triangle", p)

# %% [markdown]
# Here we'll work on the unit square.
# An interesting feature of plate problems is that they don't become much easier to solve analytically on the unit square by separation of variables because of the mixed derivatives.

# %%
n = 64
mesh = firedrake.UnitSquareMesh(n, n, diagonal="crossed")

# %% [markdown]
# #### Experiment \#1: Active set method
#
# Here we'll try solving the obstacle problem using the VINEWTONRSLS solver from PETSc.

# %%
Q = firedrake.FunctionSpace(mesh, cg)
Σ = firedrake.FunctionSpace(mesh, hhj)
Z = Q * Σ

# %%
z = firedrake.Function(Z)
L = form_hhj_lagrangian(z, f, h, μ, λ)
F = firedrake.derivative(L, z)

# %% [markdown]
# In order to get simply-supported boundary conditions, we can set both $w$ and $\sigma$ to zero on the domain boundary.

# %%
bc_w = firedrake.DirichletBC(Z.sub(0), 0, "on_boundary")
bc_σ = firedrake.DirichletBC(Z.sub(1), 0, "on_boundary")
bcs = [bc_w, bc_σ]

# %% [markdown]
# First, we'll solve the problem with no bounds constraints.
# This will give us an idea of the range of values that it can take.

# %%
firedrake.solve(F == 0, z, bcs)

# %%
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_axis_off()
colors = firedrake.tripcolor(z.sub(0), axes=ax)
fig.colorbar(colors);

# %% [markdown]
# Now we'll create the obstacle function.
# We want to make sure that a zero initial guess for the displacement is strictly feasible.
# There are solution strategies that even work with infeasible initial guesses.
# That's a complication that I don't want to worry about yet.

# %%
w_min = abs(z.sub(0).dat.data_ro.min())

# %%
x = firedrake.SpatialCoordinate(mesh)

w_0 = Constant(w_min)
δw = Constant(7/8 * w_min)
r = Constant(1 / 8)
y = firedrake.as_vector((1/2, 1/2))

expr = -w_0 + δw * (1 - inner(x - y, x - y) / r**2)
ψ = firedrake.Function(Q).interpolate(expr)

# %% [markdown]
# The plot below shows the obstacle function, but I've set the upper and lower bounds so that we only get color where the obstacle exceeds the lower bound on the initial solution.
# This gives us an idea of where the initial coincidence set will be.

# %%
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_axis_off()
colors = firedrake.tripcolor(ψ, vmax=+w_min, vmin=-w_min, axes=ax)
fig.colorbar(colors);

# %%
z_lower = firedrake.Function(Z)
z_lower.sub(0).assign(ψ)
z_lower.sub(1).dat.data[:] = -np.inf

z_upper = firedrake.Function(Z)
z_upper.sub(0).dat.data[:] = +np.inf
z_upper.sub(1).dat.data[:] = +np.inf

# %% [markdown]
# Here we'll set up the problem.

# %%
problem = firedrake.NonlinearVariationalProblem(F, z, bcs)

# %% [markdown]
# If we try and solve it using standard options, the solver diverges.

# %%
params = {
    "solver_parameters": {
        "snes_type": "vinewtonrsls",
        "snes_linesearch_type": "secant",
        "snes_linesearch_max_it": 10,
        #"snes_vi_monitor": None,
    },
}
solver = firedrake.NonlinearVariationalSolver(problem, **params)
try:
    solver.solve(bounds=(z_lower, z_upper))
except firedrake.ConvergenceError:
    print("It broke :(")

# %% [markdown]
# Reset the solution to zero.
# We'll then change the solver options to take a single SNES step and skip the convergence test.
# That way we can call `solver.solve` to do a single Newton step and then see what's happening.

# %%
z.sub(0).assign(0)
z.sub(1).assign(0);

# %%
params = {
    "solver_parameters": {
        "snes_max_it": 1,
        "snes_convergence_test": "skip",
        "snes_type": "vinewtonrsls",
        "snes_linesearch_type": "secant",
        "snes_linesearch_max_it": 10,
        #"snes_vi_monitor": None,
    },
}
solver = firedrake.NonlinearVariationalSolver(problem, **params)

# %% [markdown]
# Try the solver 50 times and save it to a list so we can plot it.

# %%
zs = [z.copy(deepcopy=True)]

num_steps = 50
for step in range(num_steps):
    solver.solve(bounds=(z_lower, z_upper))
    zs.append(z.copy(deepcopy=True))

# %% [markdown]
# Make a movie.

# %%
zmin = np.array([z.sub(0).dat.data_ro.min() for z in zs]).min()
zmax = np.array([z.sub(0).dat.data_ro.min() for z in zs]).max()

zm = max(zmax, -zmin)

# %%
# %%capture

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_axis_off()

kw = {"vmin": -zm, "vmax": +zm, "cmap": "managua_r", "num_sample_points": 4}
colors = firedrake.tripcolor(zs[0].sub(0), axes=ax, **kw)
fn_plotter = firedrake.FunctionPlotter(mesh, num_sample_points=4)
animate = lambda z: colors.set_array(fn_plotter(z.sub(0)))
animation = FuncAnimation(fig, animate, zs, interval=1e3 / 4)

# %% [markdown]
# Looks like it's thrashing on computing the coincidence set somehow.

# %%
HTML(animation.to_html5_video())

# %% [markdown]
# #### Experiment \#2: an interior point method
#
# Now let's try another scheme where we introduce a slack variable explicitly.
# We're going to use a new approach called the *latent variable proximal point* method.
# This method is an amalgam of interior point methods and the method of mirror descent.
# Suppose that we have a convex optimization problem
# $$w = \text{argmin}_z J(z), \quad Bz \in K$$
# where $K$ is some closed convex set and $B$ is a linear operator.
# A blunt approach is to use a barrier method.
# We first pick a convex functional $R$ that is finite in the interior of $K$ and grows approaching the boundary.
# (I'll get more specific about what $R$ has to look like in a moment.)
# We then look for a sequence of minimizers of
# $$J_\alpha(w) = \alpha\,J(w) + R(Bw)$$
# as the parameter $\alpha$ goes to zero.
# Barrier methods are easy to understand, but become ill-conditioned as $\alpha \to 0$.
#
# You can think of interior point methods as a hack that helps us work around the ill-conditioning of barrier methods.
# Interior point methods don't modify the optimization problem you solve.
# Instead, they modify the optimality conditions.
# I find that distasteful but I don't have a good reason why.
#
# The idea behind LVPP is to work with the convex conjugate of $R$:
# $$R(w) = \max_z\,\langle B^*z, w\rangle - R^*(z).$$
# Of course we need that $R$ has an easily computable convex conjugate.
# But we can then replace the barrier subproblem with the saddle point problem
# $$L_\alpha(w, z) = \alpha\, J(w) + \langle Bw, z\rangle - R^*(z).$$
#
# The conventional approach is to use a penalty functional $R$ that goes to infinity as $z$ approaches the boundary of $K$.
# For example, when $B$ is the identity and $K = \{z : z \ge 0\}$, we could take
# $$R(z) = -\int_\Omega\ln z\,\mathrm dx.$$
# That works but it's blunt and we can be subtler.
# For example, again for non-negativity constraints, we can take $R$ to be the "entropy" functional:
# $$R(z) = \int_\Omega z\,(\ln z - 1)\,\mathrm dx.$$
# The entropy functional is convex, and takes negative values for $z > 0$.
# But in contrast to the logarithm function, it approaches 0 as $z$ approaches the boundary of the feasible set.
# Now granted the cost does increase as $z$ approaches the boundary of the feasible set, but by itself, that doesn't seem like strong enough growth to guarantee strict feasibility.
# How is the entropy an effective barrier then?
# The answer lies not in how $R$ behaves but its derivative:
# $$\nabla R(z), \delta z\rangle = \int_\Omega \ln z\cdot \delta z\,\mathrm dx.$$
# So the derivative of $R$ grows to infinity as $z$ approaches the boundary of the feasible set, even though $R$ itself stays finite.
# We can see why a functional like this is an effective barrier by writing down the optimality conditions:
# $$\alpha\,\mathrm dJ(z) + \mathrm dR(Bz) = 0.$$
# The asymptotic behavior of $\mathrm dR$ is enough to push a candidate solution towards strict feasibility.
#
# Now we come to duality.
# The optimality condition for the dual problem are
# $$\begin{align}
# \alpha\,\mathrm dJ(z) + B^*z & = 0 \\
# Bw - \mathrm dR^*(z) & = 0.
# \end{align}$$
# The next part is galaxy brain thinking.
# The mapping $\mathrm dR$ maps the interior of $K$ into the whole space -- $\mathbb R^d$ if we're in finite dimensions, or a whole function space otherwise.
# Now we can rely on what the convex conjugate means.
# If $\mathrm dR$ is a bijective map, then $\mathrm dR^*$ is its inverse.
# It maps all of space *back* into the interior of $K$.
# Now focus on the second optimality condition.
# If $Bw = \mathrm dR^*(z)$ for some $z$, then $Bw$ has to be in the interior of $K$.

# %%
S = Q * Σ * Q

s = firedrake.Function(S)
s_n = firedrake.Function(S)
w, σ, z = firedrake.split(s)
δw, δσ, δz = firedrake.TestFunctions(S)
w_n, σ_n, z_n = firedrake.split(s_n)

# %% [markdown]
# First, we form the Lagrangian as we normally would for an unconstrained plate problem.

# %%
L_w = -form_hhj_lagrangian(s, f, h, μ, λ)

# %% [markdown]
# I alluded above to the fact that we'll be using the entropy as our penalty functional:
# $$R(w) = \int_\Omega (w - \psi)\{\ln(w - \psi) - 1\}\,\mathrm dx.$$
# I'll leave it as an exercise to the reader to calculate the convex conjugate:
# $$R^*(z) = \int_\Omega\left(\psi\cdot z + e^z\right)\,\mathrm dx.$$

# %%
R = (z * ψ + exp(z)) * dx
L_z = w * z * dx - R

# %% [markdown]
# The combined Lagrangian is
# $$L_\alpha = \alpha L(w, \sigma) + \langle z, w\rangle - R^*(z).$$

# %%
α = Constant(10.0)
L = α * L_w + L_z

# %% [markdown]
# Commence prayer.

# %%
F = firedrake.derivative(L, s) - z_n * δw * dx

# %%
bc_w = firedrake.DirichletBC(S.sub(0), 0, "on_boundary")
bc_σ = firedrake.DirichletBC(S.sub(1), 0, "on_boundary")
bcs = [bc_w, bc_σ]

# %%
params = {
    "solver_parameters": {
        "snes_type": "newtonls",
        "snes_max_it": 200,
        "snes_linesearch_type": "bt",
        "snes_linesearch_max_it": 40,
        "snes_monitor": None,
    },
}

problem = firedrake.NonlinearVariationalProblem(F, s, bcs)
solver = firedrake.NonlinearVariationalSolver(problem, **params)
solver.solve()

# %%
s_n.assign(s)
solver.solve()

# %%
w, σ, z = s.subfunctions

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_axis_off()
colors = firedrake.tripcolor(w, axes=ax)
fig.colorbar(colors);

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.view_init(elev=60)
firedrake.trisurf(w, axes=ax);

# %%
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_axis_off()
colors = firedrake.tripcolor(z, axes=ax)
fig.colorbar(colors);

# %%
δw = firedrake.Function(Q).interpolate(w - ψ)

# %%
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_axis_off()
colors = firedrake.tripcolor(δw, vmax=1e-3, axes=ax)
fig.colorbar(colors);
