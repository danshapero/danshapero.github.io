---
layout: post
title: "Clojure's threading macro in Common Lisp"
date: 2015-01-22 22:12:00
categories: lisp macros
---


Macros: make your own syntax
============================

A few months back I drank the Lisp kool-aid; I've had a lot of fun learning the ins and outs of functional programming.
One of the great things about Lisp is that it's *homoiconic*: code and data are represented in the same fashion.
We can then analyze and transform Lisp source code in the same way that we would data of any other kind.

In Lisp dialects, this capacity is exposed to the user through *macros*, with which one can extend the syntax of Lisp.
A practical example of this is how Common Lisp's iteration constructs like `do` and `loop` are macros that expand to more primitive statements like `tagbody` and `go`.
For example, this is a bit of code I wrote to generate a ring graph on `n` vertices:

{% highlight common-lisp %}
(do ((i 0 (1+ i))
     (j 1 (mod (1+ j) n))
     (g (make-instance 'graph)
        (add-edge g i j)))
    ((= i 16) g)))
{% endhighlight %}

If we quote the code above and pass it to the Common Lisp function `macroexpand`, we get to see what it really is under the hood:

{% highlight common-lisp %}
 (BLOCK NIL
  (LET ((I 0) (J 1) (G (MAKE-INSTANCE 'GRAPH)))
    (TAGBODY
      (GO #:G1385)
     #:G1384
      (TAGBODY)
      (PSETQ I (1+ I)
             J (MOD (1+ J) N)
             G (ADD-EDGE G I J))
     #:G1385
      (UNLESS (= I 16) (GO #:G1384))
      (RETURN-FROM NIL (PROGN G)))))
{% endhighlight %}

This is what people mean when they say that Lisp is a programmable programming language.


Clojure's threading macro
=========================

Clojure is a modern Lisp dialect that runs on the JVM; lately it's been gaining a lot of popularity in web programming for a variety of reasons, chief among which is the grace with which it handles concurrency.
One of my favorite features about it is the threading macro.

In Lisp dialects, one can wind up writing very deep nested function calls when some data has to pass through many computations before it's used.
For example, this Clojure code reads in a text file and converts the contents into a hash-map:

{% highlight clojure %}
(read-string (slurp (io/file (io/resource filename))))
{% endhighlight %}

That's a little bit ugly.
Clojure's threading macro allows us to write this as:

{% highlight clojure %}
  (-> filename
      io/resource
      io/file
      slurp
      read-string)
{% endhighlight %}

The first form filename is inserted as the second item in a list with the second form as the head; this process is then repeated until no more forms are left.
The operator `->` is called the threading operator; the snippet above macro-expands to the awful nested calls from before.
This usage is reminiscent of a style called concatenative programming, as exemplified in Unix pipes.


Porting the macro to Common Lisp
================================

The threading operator is so useful in Clojure that I can hardly live without it.
Naturally, when I started learning Common Lisp, I wanted to be able to use the same macro in this environment too, so I looked on Google for an implementation.
The first one I found was on [Github](https://github.com/nightfly19/cl-arrows):

{% highlight common-lisp %}
(defmacro -> (initial-form &rest forms)
  (let ((output-form initial-form)
        (remaining-forms forms))
    (loop while remaining-forms do
         (let ((current-form (car remaining-forms)))
           (if (listp current-form)
        (setf output-form (cons (car current-form)
           (cons output-form (cdr current-form))))
        (setf output-form (list current-form output-form))))
         (setf remaining-forms (cdr remaining-forms)))
    output-form))
{% endhighlight %}

Maybe I'm being nitpicky, but I didn't like the use of `setf`; this struck me as something that could be done without assignment.
I found another version, available [here](https://gist.github.com/kriyative/2030582), that uses a recursive macro-expansion:

{% highlight common-lisp %}
(defmacro -> (x &rest args)
  "A Common-Lisp implementation of the Clojure `thrush` operator."
  (destructuring-bind (form &rest more)
      args
    (cond
      (more `(-> (-> ,x ,form) ,@more))
      ((and (consp form)
            (or (eq (car form) 'lambda)
                (eq (car form) 'function)))
       `(funcall ,form ,x))
      ((consp form) `(,(car form) ,x ,@(cdr form)))
      (form `(,form ,x))
      (t x))))
{% endhighlight %}

I think macros are neat, but I'm still new at this and recursive macros are scary.
I wanted something that I could have conceivably come up with myself.
The first version I came up with used a do loop:

{% highlight common-lisp %}
(defmacro -> (x &rest forms)
  (flet ((expand-form (x form)
           (if (consp form)
               (if (or (eq (car form) 'lambda)
                       (eq (car form) 'function))
                   `(funcall ,form ,x)
                   `(,(car form) ,x ,@(cdr form)))
               `(,form ,x))))
    (do ((forms forms (cdr forms)
         (x x (expand-form x (car forms))))
        ((not forms) x))))
{% endhighlight %}
        
but I had a gut feeling that I could improve on it.
Finally, it hit me: I don't need a looping construct, it's just a reduction of the list of forms using the function that expands a single form!

{% highlight common-lisp %}
(defmacro -> (x &rest forms)
  (flet ((expand-form (x form)
           (if (consp form)
               (if (or (eq (car form) 'lambda)
                       (eq (car form) 'function))
                   `(funcall ,form ,x)
                   `(,(car form) ,x ,@(cdr form)))
               `(,form ,x))))
    (reduce #'expand-form forms :initial-value x)))
{% endhighlight %}
    
My very first Lisp macro!
And hopefully not the last either.