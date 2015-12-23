---
layout: post
title: "Computational geometry is horrible"
categories: math
---

Algorithms in computational geometry can fail in dramatic ways when implemented naively in floating point arithmetic: convex hulls that aren't convex, Delaunay triangulations that don't obey the Delaunay property, and many other unspeakable horrors.
As someone who works on numerical PDE, this situation is quite aggravating; I just want a good computational mesh and fast so that I can move on to "the important stuff".
In this post, I'll give examples of how naive geometric computations can fail and what to do about it.


## Orientation

The orientation test is to determine whether a set of three points \\(x, y, z\\) in the plane are in clockwise or counterclockwise order.
This amounts to the evaluation of a determinant:

$$\textrm{orientation}(x, y, z) = \det\begin{bmatrix}1 & x_1 & x_2 \\ 1 & y_1 & y_2 \\ 1 & z_1 & z_2 \end{bmatrix}$$

The points are oriented counterclockwise if the determinant of this matrix is positive and clockwise if the determinant is negative.
When the points are colinear, i.e. \\(z = a + bx + cy\\), this row is a linear combination of the other two rows and consequently the determinant is 0.

The orientation predicate is used as a building block in more complex predicates. For example, the point \\(x\\) lies inside the convex polygon consisting of the line segments \\(\\{x_0, \ldots, x_{n-1}\\}\\) if \\(\textrm{orientation}(x_i, x_{i+1}, x)\\) has the same sign for all \\(i\\).

Evaluating the orientation predicate in floating-point arithmetic can fail to reflect the true orientation of the points due to cancellation errors.
It's easy to imagine that three points which are close to colinear can be erroneously regarded as having orientation 0 in floating-point arithmetic, and this is indeed the case.
We can probe the robustness of the orientation predicate in floating-point arithmetic by picking three colinear points and seeing how the orientation varies as we perturb one of them.
We'll use the three points

$$x = (0.5, 0.5), \qquad y = (12.0, 12.0), \qquad z = (24.0, 24.0)$$

and compute the orientation of \\(x' = x + (m\epsilon, n\epsilon), y, z\\) for \\(m, n\\) from 0 to 255 and \\(\epsilon\\) sufficiently small.

![orientation3]({{ site.url }}/assets/orientation3.png)

In this iamge, red blocks represent counterclockwise orientation, blue blocks represent clockwise orientation and black blocks represent colinear points.
The large black blocks along the diagonal show that many non-colinear points are erroneously regarded as colinear.

The pathologies get even more drastic.
In exact arithmetic, the orientation predicate is invariant to cyclic permutations:

$$\textrm{orientation}(x, y, z) = \textrm{orientation}(y, z, x) = \textrm{orientation}(z, x, y).$$

In floating-point arithmetic, this identity fails to hold.
If we do a cyclic permutation on the input points, still perturbing \\(x\\), we get this image instead:

![orientation2]({{ site.url }}/assets/orientation2.png)

which is quite different from the first; fewer non-colinear points are classified as colinear.
However, if we do another cyclic permutation on the inputs, we get this monstrosity:

![orientation1]({{ site.url }}/assets/orientation1.png)

The regions of the plane classified as clockwise, counterclockwise or colinear are no longer connected, and there are isolated points which have their orientation completely flipped.


## Can't stop here, this is bat country

Geometric predicates can give incorrect answers when implemented naively, so we have to do something smarter.
One approach is to write a library that does all floating-point arithmetic exactly, but this will be costly.
Exact floats must be allocated on the program heap rather than the stack, since each one requires an indeterminate amount of memory.
The size of the exact floats used will also tend to balloon upwards as the algorithm continues and the numbers involved take on longer and longer tails, whether or not these tails are necessary to deliver an accurate result.

An alternate approach due to Douglas Priest and Jonathan Richard Shewchuk is to use *adaptive precision*.
Rather than default to exact arithmetic, the naive predicates are always tried first.
If an a-posteriori error analysis detects that the result from a naive predicate is not trustworthy due to roundoff, the inaccurate result is used as the first approximation of a succession of more exact predicates until we do arrive at an accurate outcome.
