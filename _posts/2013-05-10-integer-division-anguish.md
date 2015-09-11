---
layout: post
title: "Integer division rounds off decimals, causes anguish"
date: 2013-05-10
categories: math
---

Last summer I wrote a program in Fortran 90 for modeling the flow of ice streams, which is what I do my doctoral research on.
It was a succession of idiotic pratfalls worthy of the Stooges Three.

In order to predict how a solid or fluid is going to flow, we need to know how it reacts under strain -- does it snap back instantaneously (elastic) or creep back slowly to equilibrium (viscous)?
This is called the constitutive relation for the material; as you might imagine, water and pure ethanol will have fairly similar constitutive relations since they're both liquids, but water and steel do not since steel is a solid.

The simplest constitutive relation you could come up with for a viscous flow is Hooke's law: stresses \\((\tau)\\) and strain rates \\((\dot\varepsilon)\\) are linearly related to each other:

\\[ \tau = 2\mu\dot\varepsilon. \\]

The proportionality constant is the viscosity, \\(\mu\\).

Ice is actually a pretty unusual substance in that the viscosity depends on the strain rate; it is a [non-Newtonian](http://en.wikipedia.org/wiki/Non-Newtonian_flud) material.
Empirically, we observe that the viscosity decreases as the substance deforms; this is called *shear-thinning*.
Blood and ketchup are both shear-thinning fluids also.
Experiments by Nye in the 50s showed that the relation

\\[ \mu = \frac{B}{2}\dot\varepsilon^{-2/3} \\]

gave a good fit to laboratory measurements of ice under uniaxial strain.

So, in my program I had a line to find the viscosity at each point `i` in the computational grid:

`mu(i) = 0.5 * B * strain(i)**(-2/3)`

But, computers are very particular: when you do any arithmetic on two integers, you get an integer as a result.
Dividing one number by another doesn't give you the remainder unless one of them is a floating-point number:

`4/2 = 2,  10/3 = 3,  10.0/3 = 3.3333333333`

Consequently, the expression I wrote above evaluates to

`mu(i) = 0.5 * B * strain(i)**0`,

or just `0.5 * B`.
The fix is to write one of the numbers as a floating-point literal:

`mu(i) = 0.5 * B * strain(i)**(-2.0/3)`.

This mistake is especiall insidious because it's easy to miss.
Instead of simulating a shear-thinning fluid, I wound up with an ordinary Newtonian fluid.
It certainly wasn't correct, but it wasn't so obviously wrong that I would notice until I checked my results against an analytical solution.