---
layout: post
title: "Nitsche's method"
categories: math
---

To solve a partial differential equation, we have to also supply boundary conditions, which describe what happens around the edge of the spatial domain.
For illustrative purposes, we'll use steady-state heat conduction as a model problem:

$$-\nabla\cdot k\nabla \theta = q,$$

where \\(\theta\\) is the temperature of the medium, \\(k\\) is the thermal conductivity, and \\(q\\) the sum of all heat sources.
At the boundary of the domain \\(\Omega\\), we can either assume the medium is at some fixed temperature:

$$\theta|_{\partial\Omega} = \theta_\Gamma$$

or we can assume that the heat flux out of the domain is fixed at some value:

$$-k\nabla\theta\cdot n = f$$

where \\(n\\) is the unit outward normal vector.
These are called, respectively, Dirichlet and Neumann boundary conditions.

When we solve this PDE using the finite element method, Dirichlet and Neumann boundary conditions are enforced in completely different ways.
Neumann boundary conditions can be incorporated directly into the weak form of the problem.
Dirichlet boundary conditions, on the other hand, require modification of the discretized problem.
Usually this amounts to modifying entries of the matrix equation that determines the solution, but there's more than one way to do the same thing.
For this reason Neumann boundary conditions are called "natural" while Dirichlet boundary conditions are "essential".
Which boundary conditions are natural and which are essential depends on the precise formulation of the problem.
If we instead solved the mixed formulation of the Poisson equation, Dirichlet conditions are natural while Neumann boundary conditions are essential, despite the fact that the two PDEs are mathematically equivalent.

Some problems can have boundary conditions that defy easy classification.
For example, consider the Stokes equations for the velocity \\(u\\) and pressure \\(p\\) of an incompressible fluid at very low Reynolds number.
Rather than the lid-driven cavity flow problem that is so often used as a test case, suppose instead that the fluid inside is in imperfect frictional contact with some outside medium.
The fluid is confined to the interior of the domain, so at the boundary it can have no normal flow component:

$$u\cdot n = 0.$$

Along the boundary, however, the fluid experiences frictional resistance:

$$(I - n\otimes n)\tau\cdot n = -\kappa(I - n\otimes n)u$$

where \\(\kappa\\) is the friction coefficient and \\(\tau\\) is the stress tensor.
This is a Robin boundary condition, but only along some directions at the boundary.
If the boundary of the domain were aligned with the coordinate axes, it would be easy to make the right modifications to the system stiffness matrix, but of course we'd like to be able to deal with arbitrary geometries too.

Essential boundary conditions are generally kind of annoying and a frequent source of error, so it would be nice if we could find a different way to impose them.
There are two "obvious" ways to do this: the penalty method and the Lagrange multiplier method.
Both of this are unsatisfactory for reasons I'll describe.
Nitsche's method is a much less obvious way to accomplish the same thing.
Additionally, it was the inspiration for the subsequent development of discontinuous Galerkin methods.

In the following, I'll assume that you're familiar with the variational description of elliptic PDE.
Defining the objective functional

$$J(\theta) = \int_\Omega\left(\frac{1}{2}k|\nabla \theta|^2 - q\theta\right)dx,$$

the solution \\(\theta\\) of the Poisson equation is the minimizer of \\(J\\).
Moreover, with sufficient regularity assumptions on the domain and the coefficients, the minimizer of \\(J\\) is also the solution of the Poisson equation.
Provided that the conductivity is positive, the objective functional is convex.
From here on out we'll use the optimization formulation of the PDE exclusively.

### The penalty method

The idea of the penalty method is that, rather than force the solution to exactly satisfy the boundary conditions, we will merely make departures from them "expensive" by adding an extra term to the objective functional:

$$J_\gamma(\theta) = J(\theta) + \int_{\partial\Omega}\frac{\gamma k}{2h^p}(\theta - \theta_\Gamma)^2ds.$$

The factor of \\(k\\) is there to get the units right.
The other factors are the constant \\(\gamma\\), which depends only on the element family and the shape of the domain, and the local element size \\(h\\).
The exponent \\(p\\) controls how much we increase the penalty for departures from the boundary conditions as the mesh is refined.

Using the penalty method is equivalent to applying a Robin boundary condition with a very high exchange coefficient:

$$-k\nabla\theta\cdot n = h^{-p}\gamma k(\theta - \theta_\Gamma).$$

This approach has the virtue of being very easy to implement.

In other respects, however, the penalty method falls short.
The scaling of the penalty as some power of the mesh spacing is necessary for the boundary condition error to be comparable to the discretization error, but this leads to poor conditioning of the resulting matrix equation as the mesh is refined.
Moreover, the accuracy of the solution in the \\(H^1\\) norm is totally compromised for higher order elements.


### The Lagrange multiplier method

We can instead proceed by analogy with mixed finite element methods and introduce a Lagrange multiplier \\(\lambda\\) to exactly enforce the boundary conditions.
We then seek to extremize the Lagrangian

$$L(\theta, \lambda) = J(\theta) + \int_{\partial\Omega}\lambda\cdot(\theta - \theta_\Gamma)ds,$$

where \\(\lambda\\) is in the dual \\(H^{-1/2}(\partial\Omega)\\) to the trace space \\(H^{1/2}(\partial\Omega)\\).
If you're familiar with mixed finite elements, you'll be wondering which function space to use for the Lagrange multipliers in order to get a discretization that satisfies the Ladyzhenskaya-Babuška-Brezzi conditions.
Babuška himself proposed this idea in the 60s but left open the question of which finite element space for \\(\lambda\\) would give a stable discretization.
Pitkäranta studied this problem in a series of papers from the mid-80s.
As far as I can tell the discrete finite element spaces are impractical for real use.
I haven't found a software package that implements them.

We can, however, do a little bit more math to get an expression for \\(\lambda\\) that we'll use shortly.
Assuming that the solution \\(\theta\\) is in \\(H^2\\) and integrating the variational derivative of the Lagrangian along a perturbation \\(\phi\\) by parts, the remaining boundary terms are

$$\left\langle\frac{\partial L}{\partial\theta}, \phi\right\rangle = \ldots + \int_{\partial\Omega}\left(\lambda + k\nabla\theta\cdot n\right)\phi \hspace{2pt}ds = 0,$$

and since \\(\phi\\) is arbitrary we then have that

$$\lambda = -k\nabla\theta\cdot n.$$

In other words, the Lagrange multiplier is whatever value of the heat flux that would have made the boundary values of \\(\theta\\) equal to \\(\theta_\Gamma\\).


### The augmented Lagrangian method

The idea behind the augmented Lagrangian method is to smash the penalty and Lagrange multiplier methods together:

$$L_\gamma(\theta, \lambda) = J(\theta) + \int_{\partial\Omega}\lambda(\theta - \theta_\Gamma)ds + \int_{\partial\Omega}\frac{\gamma k}{2h^p}(\theta - \theta_\Gamma)^2ds.$$

This isn't as silly as it sounds.
If the field \\(\theta\\) satisfies the constraints that the Lagrange multiplier imposes, then the penalty term is 0, adding nothing to the objective functional.
The extra penalty might improve the overall stability of the system to the point where it's easier to satisfy the LBB conditions.

The real advantage, as we'll show in the following, is that we can obtain optimal-order convergence for any finite element using a penalty exponent of \\(p = 1\\), rather than the \\(p = 2\\) of the pure penalty method.
Both the accuracy of the discrete solution and the condition number of the linear system are superior to that of the penalty method.


### Nitsche's method

The augmented Lagrangian method is a general trick for constrained convex optimization problems.
Nitsche's method is what we get when we combine the augmented Lagrangian approach with the fact that the Lagrange multiplier is the boundary heat flux:

$$J_\gamma(\theta) = J(\theta) - \int_{\partial\Omega}k\frac{\partial\theta}{\partial n}\left(\theta - \theta_\Gamma\right)ds + \int_{\partial\Omega}\frac{\gamma k}{2h}\left(\theta - \theta_\Gamma\right)^2ds.$$

This is a quadratic functional in the temperature, but there's no guarantee from the outset that it should be convex.
In the following, we'll derive some estimates on the boundary flux term that enable us to pick a value of \\(\gamma\\) for which the Nitsche functional is convex.
