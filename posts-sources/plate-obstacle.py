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
# In which we discover what a convex conjugate truly is and where everything gets fixed by flipping a sign.
#
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

# %%
z_init = z.copy(deepcopy=True)

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

# %% [markdown]
# If we try and solve it using standard options, the solver diverges.

# %%
problem = firedrake.NonlinearVariationalProblem(F, z, bcs)
params = {
    "solver_parameters": {
        "snes_monitor": ":plate-obstacle.log",
        "snes_type": "vinewtonrsls",
        "snes_linesearch_type": "bt",
        "snes_linesearch_max_it": 40,
        #"snes_vi_monitor": None,
    },
}
solver = firedrake.NonlinearVariationalSolver(problem, **params)
try:
    solver.solve(bounds=(z_lower, z_upper))
except firedrake.ConvergenceError as error:
    print(error)

# %% [markdown]
# Every day the Lord tests me, with this bullshit.

# %%
z.assign(z_init)
problem = firedrake.NonlinearVariationalProblem(-F, z, bcs)
solver = firedrake.NonlinearVariationalSolver(problem, **params)
solver.solve(bounds=(z_lower, z_upper))

# %%
with open("plate-obstacle.log", "r") as log_file:
    lines = log_file.readlines()
    errors = np.array([float(line.split()[-1]) for line in lines])

# %%
fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlabel("iteration")
ax.set_ylabel("error norm")
ax.scatter(list(range(len(errors))), errors);

# %%
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_axis_off()
colors = firedrake.tripcolor(z.sub(0), axes=ax)
fig.colorbar(colors);
