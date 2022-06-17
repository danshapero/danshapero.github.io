.. title: Home
.. slug: index
.. date: 2019-11-20 22:09:16 UTC-08:00
.. tags:
.. category:
.. link:
.. description:
.. type: text
.. hidetitle: True

This page is to host my experiments with math and programming.
Most of the posts use the finite element modeling library `Firedrake <https://www.firedrakeproject.org>`_ to look at some practical aspect of how to solve partial differential equations.
The diagram below shows all the posts and the arrows show how they relate to each other.
Every post is a Jupyter notebook which you can download and run yourself.
If you want to contact me you can find an email address on my `GitHub <https://www.github.com/danshapero>`_ page.

.. graphviz::

    digraph posts {
        node [shape="plaintext", fontcolor="#007bff", fontsize="11"];
        edge [arrowsize=0.25];

        calc [label="Variational\ncalculus", href="/posts/variational-calculus/"];
        stokes [label="Stokes\nflow", href="/posts/stokes/"];
        weyl [label="Weyl's law", href="/posts/weyls-law/"];
        yau [label="Yau's conjecture", href="/posts/yau-conjecture/"];
        kac [label="Kac's conjecture", href="/posts/kac-conjecture/"];
        nitsche [label="Nitsche's\nmethod", href="/posts/nitsches-method/"];
        nitsche_stokes [label="Nitsche's method\nfor Stokes flow", href="/posts/nitsches-method-stokes/"];
        nitsche_nonlinear [label="Nitsche's method\nfor nonlinear PDE", href="/posts/nitsches-method-nonlinear/"];
        obstacle [label="The obstacle\nproblem", href="/posts/obstacle-problem/"];
        conservation_laws [label="Conservation\nlaws", href="/posts/conservation-laws/"];
        convection_diffusion [label="Convection-\ndiffusion", href="/posts/convection-diffusion/"];
        inverse_problems [label="Inverse\nproblems", href="/posts/inverse-problems"];
        total_variation [label="Total\nvariation\nregularization", href="/posts/total-variation"];
        admm [label="Alternating\ndirection\nmethod of\nmultipliers", href="/posts/admm"];
        shallow_water [label="The shallow water\nequations", href="/posts/shallow-water/"];
        overland_flow [label="Overland\nflow", href="/posts/overland-flow/"];
        rosenbrock [label="Rosenbrock\nschemes", href="/posts/rosenbrock/"];
        langevin [label="Langevin\nMonte\nCarlo", href="/posts/langevin-mcmc/"];
        symplectic [label="Symplectic\nintegrators", href="/posts/symplectic-integrators/"];
        billiards [label="Billiards\non surfaces", href="/posts/surface-billiards/"];

        {rank=same; calc, conservation_laws};

        calc -> weyl;
        calc -> stokes;
        calc -> nitsche;
        calc -> obstacle;
        stokes -> nitsche_stokes;
        stokes -> inverse_problems;
        inverse_problems -> total_variation;
        nitsche -> admm;
        total_variation -> admm;
        inverse_problems -> langevin;
        obstacle -> total_variation;
        weyl -> yau;
        yau -> kac;
        nitsche -> nitsche_stokes;
        nitsche_stokes -> nitsche_nonlinear;
        conservation_laws -> convection_diffusion;
        conservation_laws -> shallow_water;
        shallow_water -> rosenbrock;
        shallow_water -> overland_flow;
        nitsche -> convection_diffusion;
        symplectic -> billiards;
    }
