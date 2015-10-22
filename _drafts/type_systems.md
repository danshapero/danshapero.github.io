---
layout: post
title: "Structural and nominal type disciplines"
categories: types
---


One of the great things about FORTRAN when it was first introduced was that you no longer had to give names to the results of intermediate computations.
In assembly code, you always have to assign the values of every sub-expression to some register.
When programming languages started getting more abstract you could write something like

`z = x + a * y`

without having to assign `a * y` to some register.

A common definition of functional programming is the ability to use functions as values -- passing them as arguemnts to other functions, returning functions from other functions, and so on and so forth.
Of course, one can do such things in languages that I would hardly call functional; you can pass and return function pointers in C and Fortran.
However, the function you wish to manipulate as a value must ultimately be named and defined somewhere.
Languages like Lisp and ML take this a step further by allowing you to create and use functions without having to explicitly give them names.
For example, if I want to add up all the squares of the elements of a list `l` in Scheme, I don't have to define some function `square`; rather, I would pass an anonymous function to `reduce`.

{% highlight scheme %}
(reduce (lambda (r x) (+ r (* x x))) l 0)
{% endhighlight %}

I don't have to give a name to a function to be able to use it.
The rabbit hole gets even deeper when recursion comes into the picture.
Normally, when you write a recursive function, you need to have a name to recurse on; for example, in the body of the factorial function

{% highlight scheme %}
(define (fact n)
  (if (= n 0)
      1
      (* n (fact (- n 1)))))
{% endhighlight %}

you recurse on `fact` applied to `(- n 1)` in the induction step.
The Y-combinator allows you to express this notion without any pre-existing name `fact`, using only anonymous functions.

I would go further, and argue that one of the central tenets of functional programming is that **you never have to name a value to be able to use it**.
I first read about this principle when I started learning Lisp and OCaml and I wondered why it was such a big deal.
However, I think that once you accept the notion that 



# Nominal type systems

To explain what makes ML's type system so special, it helps to examine its opposite, for which I'll hold up C as an exemplar.
In order to be able to pass a value to a function, the type of that value must be compatible with the type the function expects.
You can't pass a float to a function that expects an integer, you can't pass one kind of struct where another is expected, and so on and so forth.
The chief means by which the compiler decides whether a value can be passed to a function or returned from it is by the type name; the C type system is *nominal*.

Type names can be aliased by using `typedef`.
For example, the C standard library defines certain integer types like `size_t`, `intptr_t`, `ptrdiff_t` which are aliases for either 32- or 64-bit integers.
Application code which aims to be as agnostic to the underlying platform as possible can us `intptr_t` without having to know whether it's a normal or a long integer, for which the aliasing mechanism is invaluable.

Suppose that I defined two structs:

{% highlight c %}
struct csr_matrix {
  size_t num_rows;
  size_t num_cols;
  size_t * offsets;
  size_t * nodes;
  double * values;
};

struct csc_matrix {
  size_t num_rows;
  size_t num_cols;
  size_t * offsets;
  size_t * nodes;
  double * values;
};
{% endhighlight %}

These would be used to represent row- and column-oriented storage formats for sparse matrices, respectively.
Despite the fact that the structure of a `csc_matrix` is identical to that of a `csr_matrix`, trying to pass a column matrix to a procedure expecting a row matrix would throw an error because the types are incompatible, where compatibility is determined by type name.

This is a good thing which helps us detect errors.
Even though the representation in computer memory of a row matrix is the same as that of a column matrix, they mean two different things in the mathematical sense, and therefore should not be compatible.
Moreover, the struct could be opaque, in which case we don't even know what's contained in it.


# Structural type systems

ML and related languages cleave to the principal that nothing requires a name in order to be used.
Consequently, they don't have a nominal type system like C; rather than give an explicit name like `csr_matrix`, the type of that object would be the tuple

` (num_rows : int, num_cols : int, offsets : Array int, nodes : Array int, values : Array double).`


Structural type disciplines are not only a feature of functional languages.
Interfaces in golang are determined purely through structure and argument compatibility.
Additionally, the template meta-language in C++ obeys a structural type discipline, while the underlying C++ language obeys a nominal type discipline.
A class which defines the required class-local objects and typedefs to fulfill the iterator concept can be used wherever an STL iterator is expected; you never have to formally declare that it satisfies the iterator concept.


# Not better, just different

So, are structural or nominal type disciplines better?
The answer is: yes!
There are cases where one typing discipline is more natural than the other for expressing the programmer's intention.
For example, golang's structural typing for interfaces largely obviates the need for any kind of formal inheritance mechanism in the language, which can be manually implemented through delegation if need be.
This is especially convenient when certain attributes (e.g. printable, serializable, subscription) are common to nearly every object in a library; other languages would resort to multiple inheritance.
On the other hand, it opens up the possibility of making an object which accidentally satisfies an interface that you didn't intend or want it to.
While languages with nominal systems may be less prone to this kind of error, they can be inflexible and hard to extend instead.

Ideally, a language would provide mechanisms for using either typing discipline.
While the OCaml type system is largely structural, hiding the structure of a type within a module prevents unintended type compatibility.
This is common practice in the OCaml standard library containers like Map and Set, for which you only know that the module `M` exposes some type `t` and functions for operating on it, but nothing more.
This notion is formalized through the theory of existential types.
