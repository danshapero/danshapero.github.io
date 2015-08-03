---
layout: post
title: "Nesting and passing Fortran procedures for fun and profit"
date: 2015-01-31 12:38:00
categories: fortran
---

In writing the library [SiGMA](http://github.com/danshapero/sigma), I found in a few places where I was repeating code that I thought could be abstracted out.
The particular operation was to build a graph object from some source, no matter what the underlying storage format was for the graph (list of lists, compressed, etc.).
Sometimes, the source was another graph, in which case we were merely copying; the source could also be a sparse matrix, a file, a sub-graph of another graph, and so forth.
I'll refer to such a source as an edge stream, as its basic functionality is to successively return a sequence of edges.
Every graph implements an edge iterator, so graphs naturally generate an edge stream.

Some graph formats are "dynamic" in that edges can be added efficiently, while other implementations are purely static in that, once built and filled with edges, adding a new edge requires rebuilding the entire structure.
Nonetheless, even a static graph format can be built efficiently from an edge stream by determining the size of the necessary storage in one pass through the edge stream, then filling that storage on the second pass.

The problem then becomes how to devise an appropriate abstraction for generic edge streams which come from arbitrary sources.
Clearly, the relevant procedures need to keep track of the source of the edge stream somehow.
But we're not coding in C; Fortran hates aliasing, so we can't make pointers to an object unless we're explicitly allowed to do so through either the `target` or `pointer` attributes.

I found a solution to this problem, but it was peculiar for two reasons.
First, it uses not one but two unusual features of the Fortran language, nested procedures and using functions as arguments.
Second, it creates new abstractions, but it does so in a way that isn't especially object-oriented.
In order to explain it, I'll give examples of each feature in isolation and then combine them into something interesting.


# Procedures as arguments

In Fortran, procedures can be passed as arguments to and returned from other procedures.
I'll illustrate this with some code for integrating functions of a real variable.

{% highlight fortran %}
module integration

  implicit none

  abstract interface
    function real_function(x) result(y)
      real(dp), intent(in) :: x
      real(dp) :: y
    end function real_function
  end interface

contains

  function integrate(f, a, b, n) result(q)
    procedure(real_function), intent(in) :: f
    real(dp), intent(in) :: a, b
    integer, intent(in) :: n
    real(dp) :: q
    ! local variables
    integer :: k
    real(dp) :: dx

    dx = (b - a) / (n + 1)
    q = 0.0
    do k = 0, n - 1
      q = q + dx * f(a + k * dx)
    end do

  end function integrate

end module integration

{% endhighlight %}

The abstract interface in this module allows us to specify procedures with the signature of a real-valued function of a real variable.
The integrate function uses this interface to specify that its first argument must be a real-valued function.
We can demonstrate our integration routine like so:

{% highlight fortran %}
program main

  use integration
  implicit none

  real(dp) :: q

  q = integrate(f, 0.0_dp, pi, 10000)

contains

  function f(x) result(y)
    real(dp), intent(in) :: x
    real(dp) :: y

    y = 1 / dsqrt(1 - 0.25 * sin(x)**2)

  end function

end program
{% endhighlight%}

We could get even more abstract; the function we integrate could be doing all sorts of weird monkey business like interpolating its values from a set of discrete data points, or it could be the solution of an ordinary differential equation; we can always use the same integration procedure.
Note that we didn't have to declare `f` as a `real_function` anywhere; the interface is detected automatically.

I've used the simplest and worst quadrature rule in the book for purely illustrative purposes; use Simpson's rule or something better if you want decent results.


# Nested procedures

Fortran procedures can contain other procedures in order to encapsulate auxiliary tasks.
For many uses, this same effect can be achieved by lifting the nested procedure out to the main module.
Nonetheless, the nesting relationship makes it obvious what the contained procedure's role is in the broader context and hides it from the outside environment at the same time.

In other cases, nesting allows for a very clear and concise expression of certain concepts.
In the example main program from the last section, the function `f` that we integrated sure looks a lot like the integrand in the definition of an [elliptic integral](https://en.wikipedia.org/wiki/Elliptic_integral).
We can use nested functions and the integrate method defined above to give a general procedure for evaluating elliptic integrals.

{% highlight fortran %}
  function elliptic_integral(phi, z) result(F)
    real(dp), intent(in) :: phi, z
    real(dp) :: F

    F = integrate(g, 0, phi, 10000)

  contains

    function g(x) result(y)
      real(dp), intent(in) :: x
      real(dp) :: y

        y = 1 / dsqrt(1 - z**2 * sin(x)**2)

    end function g

  end function elliptic_integral
{% endhighlight %}

The interesting thing here is the use of the variable `z` inside the nested function `g`; it is never explicitly passed to `g`, and yet it is still visible inside the scope of g, through the environment of `elliptic_integral`.
However, it is entirely hidden from the integrate function.
This is called a [lexical closure](https://en.wikipedia.org/wiki/Closure_%28computer_programming%29).


# Building graphs

Now that I've illustrated some of the basic concepts, I can talk about how this is used in SiGMA.
The graph API is extended with a method `build` which takes in procedures providing the functionality of an edge stream.

{% highlight fortran %}
  type :: edge_cursor
    integer :: first, last, current
  end type edge_cursor

  abstract interface
    function make_cursor() result(cursor)
      type(edge_cursor) :: cursor
    end function make_cursor

    function edge_stream(cursor) result(edge)
      type(edge_cursor), intent(inout) :: cursor
      integer :: edge(2)
    end function edge_stream
  end interface
{% endhighlight %}

The `edge_cursor` object is used to store our current location within the stream, and the `make_cursor` function is used to create a cursor object for the given source.
If the edge stream comes from another graph or a file, the manner in which we make the cursor can vary.
The `edge_stream` interface describes functions that return a new edge when they're called, updating their state using a cursor.
Some stub code for building a graph would look something like this:

{% highlight fortran %}
  subroutine build_mygraph(g, start_stream, next_edge)
    class(mygraph), intent(inout) :: g
    procedure(make_cursor)      :: start_stream
    procedure(edge_stream)      :: next_edge

    cursor = start_stream()

    do while( cursor%current /= cursor%last )
      edge = next_edge(cursor)

      i = edge(1)
      j = edge(2)

      <do something to g, i, j>

    enddo

  end subroutine build_mygraph
{% endhighlight %}

The exact details of how to build the graph depend on the precise storage format: for a list-of-lists graph, one pass through the edge stream will suffice, while a compressed graph requires 2 passes, one to ascertain the degree of all nodes and a second to actually store the edges.
Finally, the following functions illustrate how to make an edge stream that reads edges from a file.
On the first line, the file stores the total number of edges, and on all following lines stores the starting and ending vertices of each edge.

{% highlight fortran %}
  subroutine build_graph_from_file(g, filename)
    class(graph), intent(inout) :: g
    character(len=*), intent(in) :: filename

    call g%build(make_cursor_file, edge_stream_file)

  contains

    function make_cursor_file() result(cursor)
      type(edge_cursor) :: cursor

      close(1001)
      open(file = filename, unit = 1001)

      cursor%start = 1
      cursor%current = 0
      read(1001, *) cursor%last

    end function make_cursor_file


    function edge_stream(cursor) result(edge)
      type(edge_cursor), intent(inout) :: cursor
      integer :: edge(2)

      cursor%current = cursor%current + 1
      read(1001, *) edge(1), edge(2)

    end function edge_stream

  end subroutine build_graph_from_file
{% endhighlight %}

The state that the nested functions close over is the unit number of the open file, which I lazily set to 1001 in the hopes that nothing else is using that at the time.


# Concluding remarks

The procedures above don't need to know exactly how the underlying graph type intends to build itself, they only provide the callbacks that allow it to keep getting more edges.
Likewise, the graph doesn't need to know any details of where its edges are coming from, only that it can invoke a function to get more edges until there are none left.
Before I had this abstraction, whenever I needed to build a graph, I would always build a new graph in a dynamic format and then copy it over to the invoking graph, which is a waste of memory.

I particularly like this approach because of its extensibility.
For example, if I wanted to write a new method to build a graph from a source that lived in another process, the edge stream would close over calls to the communication library MPI instead of the file as in the last example.

This approach isn't common in Fortran or other procedural languages, which don't treat functions as first-class citizens.
By contrast, in functional languages like Lisp and ML, using lexical closures to encapsulate state that persists across function calls is commonplace.
As more languages like Python and Julia incorporate ideas from both functional and object-oriented programming, the distinction will eventually become meaningless.
