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
#     date: 2019-10-18 18:27:42 UTC-07:00
#     description: ''
#     link: ''
#     slug: weyls-law
#     tags: ''
#     title: Weyl's law
#     type: text
# ---

# %% [markdown]
# In this post we'll look at eigenfunctions and eigenvalues of the Laplace operator $\Delta$ on a domain $\Omega$ in $\mathbb{R}^d$. 
# A function $\phi$ on $\Omega$ and a number $\lambda$ are an eigenfunction/eigenvalue pair if
#
#
# $$-\Delta\phi = \lambda^2\phi$$
#
# along with the Dirichlet boundary condition $\phi|_{\partial\Omega} = 0$.
# The operator $-\Delta$ is symmetric and positive-definite, so the eigenvalues are real and positive.
# I've chosen a slightly different way of writing things in terms of $\lambda^2$ because this makes the units of the eigenvalues an inverse length.
#
# The *Weyl asymptotic law* describes how the eigenvalues grow as a function of the domain size and shape.
# Weyl proved in 1911 that, if $N(\lambda)$ is the number of eigenvalues of the Dirichlet Laplacian less than $\lambda$, that
#
# $$N(\lambda) = (2\pi)^{-d}\omega_d\cdot\text{vol}(\Omega)\cdot\lambda^{d} + \mathscr{O}(\lambda^{d})$$
#
# as $\lambda \to \infty$, where $\omega_d$ is the volume of the unit ball in $\mathbb{R}^d$.
# As a sanity check, note that $\lambda$ has units of length${}^{-1}$, so the formula above is dimensionless.
# As another sanity check, you can look at the analytical expression for the eigenvalues on a box or a sphere.
# The proof given in volume 1 of Courant and Hilbert is pretty easy to follow.
# Weyl conjectured that the second term could be expressed in terms of the area of the boundary:
#
# $$N(\lambda) = (2\pi)^{-d}\omega_d\cdot\text{vol}(\Omega)\cdot\lambda^d - \frac{1}{4}(2\pi)^{1 - d}\omega_{d - 1}\cdot\text{area}(\partial\Omega)\cdot\lambda^{d - 1} + \mathscr{o}\left(\lambda^{d - 1}\right)$$
#
# but this wasn't proved in his lifetime.
# Here we'll come up with a simple domain and show how you might verify this law numerically.

# %% [markdown]
# ### Making a mesh
#
# First, we'll generate a mesh using the Python API for [gmsh](https://www.gmsh.info).
# The calls to add a plane surface and a physical plane surface are easy to forget but essential.

# %%
import gmsh
import numpy as np
from numpy import pi as π

gmsh.initialize()
geo = gmsh.model.geo

Lx, Ly = 2.0, 1.0
lcar = 1.0 / 16
origin = geo.add_point(0, 0, 0, lcar)
points = [
    geo.add_point(Lx, 0, 0, lcar),
    geo.add_point(0, Ly, 0, lcar),
    geo.add_point(-Lx, 0, 0, lcar),
    geo.add_point(0, -Ly, 0, lcar),
]
major = points[0]

outer_arcs = [
    geo.add_ellipse_arc(p1, origin, major, p2)
    for p1, p2 in zip(points, np.roll(points, 1))
]

geo.add_physical_group(1, outer_arcs)
outer_curve_loop = geo.add_curve_loop(outer_arcs)

centers = np.array([(0, 1/2), (1/2, 1/4), (1, -1/4)])
radii = [1/8, 3/16, 1/4]
hole_curve_loops = []
for center, radius in zip(centers, radii):
    hole_center = geo.add_point(*center, 0, lcar)
    deltas = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])
    hole_points = [
        geo.add_point(*(center + radius * delta), 0, lcar) for delta in deltas
    ]
    hole_arcs = [
        geo.add_circle_arc(p1, hole_center, p2)
        for p1, p2 in zip(hole_points, np.roll(hole_points, 1))
    ]
    geo.add_physical_group(1, hole_arcs)
    curve_loop = geo.add_curve_loop(hole_arcs)
    hole_curve_loops.append(curve_loop)

plane_surface = geo.add_plane_surface([outer_curve_loop] + hole_curve_loops)
geo.add_physical_group(2, [plane_surface])
geo.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write("ellipse.msh")

gmsh.finalize()

# %% [markdown]
# To make sure everything worked right, we'll visualize the mesh after loading it in.

# %%
import firedrake
import matplotlib.pyplot as plt
mesh = firedrake.Mesh("ellipse.msh")

fig, axes = plt.subplots()
firedrake.triplot(mesh, axes=axes)
axes.set_aspect("equal")
axes.legend(loc="upper right");

# %% [markdown]
# ### Using SLEPc
#
# To compute the eigenvalues and eigenfunctions of the Laplace operator, we'll use the [Scalable Library for Eigenvalue Problem Computations (SLEPc)](http://slepc.upv.es/).
# This demo used to include all the glue code to talk to SLEPc.
# Since then, Firedrake added an interface to it.
# We can specify the problem we want to solve by creating a `LinearEigenproblem` object.

# %%
from firedrake import inner, grad, dx

Q = firedrake.FunctionSpace(mesh, family="CG", degree=2)
ϕ = firedrake.TestFunction(Q)
ψ = firedrake.TrialFunction(Q)

A = inner(grad(ϕ), grad(ψ)) * dx
M = ϕ * ψ * dx

bc = firedrake.DirichletBC(Q, 0, "on_boundary")

problem = firedrake.LinearEigenproblem(A, M, bcs=bc, restrict=True)

# %% [markdown]
# To solve the right problem and to help SLEPc get the right answer we'll pass it several options.
# First, we're solving a generalized Hermitian eigenproblem.
# Since the eigenproblem is Hermitian, all the eignevalues are real, which is a very convenient simplifying assumption.
#
# For this problem we're going to use a *spectral transformation*.
# Rather than find the eigenvalues of a matrix $A$ directly, we'll instead find the eigenvalues of a matrix $f(A)$ where $f$ is invertible and holomorphic on a domain containing the spectrum of $A$.
# We can then compute the eigenvalues of $A$ as the function $f^{-1}$ aplied to the eigenvalues of $f(A)$.
# The advantage of spectral transformations is that, with a good choice of $f$, the eigenvalues of $f(A)$ can be easier to compute than those of $A$ itself.
# Since $A$ is positive-definite and we're looking for the smallest eigenvalues, a good choice is
#
# $$f(z) = 1/(z - \sigma),$$
#
# i.e. shifting and inverting.
# This spectral transformation is equivalent to finding the eigendecomposition of $(A - \sigma M)^{-1}$.
# Computing the inverse of a matrix is generally a bad idea, but under the hood it's enough to be able to solve linear systems.
#
# Anything in SLEPc having to do with spectral transformations is prefixed with `st`.
# In our case, we're using the shift-and-invert transformation (`sinvert`).
# To solve these linear systems, we'll a Krylov subspace method (`ksp_type`) with some preconditioner (`pc_type`).
# Since $A$ is symmetric and positive-definite, we can use the conjugate gradient method (`cg`).

# %%
num_values = 250
opts = {
    "solver_parameters": {
        "eps_gen_hermitian": None,
        "eps_target_real": None,
        "eps_smallest_real": None,
        "st_type": "sinvert",
        "st_ksp_type": "cg",
        "st_pc_type": "lu",
        "st_pc_factor_mat_solver_type": "mumps",
        "eps_tol": 1e-8,
    },
    "n_evals": num_values,
}
eigensolver = firedrake.LinearEigensolver(problem, **opts)

# %% [markdown]
# To check that everything worked right, we can see how many eigenvalues converged:

# %%
num_converged = eigensolver.solve()
print(num_converged)

# %% [markdown]
# Just for fun, we can plot one of the eigenfunctions.
# The zero contours of eigenfunctions are a fascinating subject -- the Courant nodal domain theorem tells us that the $n$-th eigenfunction can have no more than $n$ nodal domains.

# %%
mode_number = 24
λ = eigensolver.eigenvalue(mode_number)
ϕ = eigensolver.eigenfunction(mode_number)[0]

# %%
fig, axes = plt.subplots()
axes.set_aspect("equal")
axes.set_axis_off()
levels = np.linspace(-1.25, +1.25, 51)
kwargs = {"levels": levels, "cmap": "twilight", "extend": "both"}
contours = firedrake.tricontourf(ϕ, axes=axes, **kwargs)
fig.colorbar(contours, orientation="horizontal");

# %% [markdown]
# The following plot shows exact eigenvalue counting function and the order-1 and order-2 approximations from Weyl's law.

# %%
Es = np.array([eigensolver.eigenvalue(k) for k in range(num_values)]).real
λs = np.sqrt(Es)

# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots()
Ns = np.array(list(range(len(λs)))) + 1
axes.plot(λs, Ns, color='k', label='Exact $N(\lambda)$')

from firedrake import assemble, Constant, ds
vol = assemble(Constant(1) * dx(mesh))
area = assemble(Constant(1) * ds(mesh))

ω_2 = π
ω_1 = 2
order_1 = 1/(2*π)**2 * ω_2 * vol * λs**2
order_2 = order_1 - 1/(2*π) * ω_1 * area * λs / 4

axes.plot(λs, order_1, color='tab:blue', label='order 1')
axes.plot(λs, order_2, color='tab:orange', label='order 2')
axes.legend()

axes.set_xlabel('Eigenvalue $\lambda$')
axes.set_ylabel('Counting function $N(\lambda)$');

# %% [markdown]
# The accuracy difference is even more stark if we look at the relative error in the eigenvalue counting function.

# %%
fig, axes = plt.subplots()
error_1 = 1 - Ns / order_1
error_2 = 1 - Ns / order_2
axes.plot(λs[50:], error_1[50:], color="tab:blue", label="order 1")
axes.plot(λs[50:], error_2[50:], color="tab:orange", label="order 2")
axes.legend(loc="upper right");

# %% [markdown]
# The order-1 approximation is pretty good, but the order-2 approximation is startlingly accurate.
# Of course we've only looked at the first few hundred eigenvalues on a mesh with several thousand vertices.
# Once the corresponding wavelengths get close to the diameter of a triangle of our mesh, I'd expect the approximation to break down.
# The mesh is too coarse at that point to resolve the highly oscillatory eigenfunctions.

# %% [markdown]
# ### Conclusions
#
# The Weyl asymptotic law has some interesting physical implications.
# The first-order version of the law tells us that you can hear the area of a drumhead by fitting the sequence of harmonic frequencies to the right power.
# The second-order version of the law tells us that you can, in the same way, hear the perimeter of the drumhead by fitting the remainder of the first-order approximation.
#
# Victor Ivrii gave a proof in 1980 of the Weyl law up to second order, under some special conditions that are thought to hold for a wide class of domains.
# While proving the law up to first order is relatively elementary, Ivrii's proof used microlocal analysis, which is well and truly above my pay grade.
