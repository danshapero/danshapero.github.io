---
layout: post
title: "NixOS is neat, also weird"
categories: nix
---

[NixOS](http://nixos.org) is an operating system built around the `nix` package manager.
Nix takes a completely different approach to package management from what we've come to think of as normal.

Many package managers, such as `aptitude`, are built off of `dpkg`, and thus inherit its quirks and limitations.
For starters, only users with root access can install packages.
On your own computer this is fine, but on a shared machine you're often left installing from source.
You then have to manually specify header and library directories whenever something depends on this package.

Moreover, only one version of a package can be installed at a given time.
This limitation is harmful in multiple situations.
If you use a Linux distribution with a regular release schedule such as Ubuntu or Linux Mint, a library you want may depend on a more recent version of some dependency than is available through the system's package manager.
On the other hand, if you use a distribution like Arch where packages are almost always kept to the newest, bleeding-edge version, you've no doubt had an update break existing dependencies.

Using virtual environments and machines has become increasingly common as a way out of dependency hell.
What makes nix so impressive is that it achieves the same effect as creating a virtual development machine just by making clever use of symlinks and environment variables.


### The nix way

Nix aims instead at a "purely functional", declarative approach to package management.
A package's dependencies and build process are expressed through a *nix-expression*, written in nix's own domain-specific language.
From the user's perspective, nix resolves and builds dependencies just the same as apt-get or yum would.

What it actually does is far more interesting.

With this approach, users without administrative privileges can install and use libraries in their home directory, and multiple versions of the same package can safely coexist.

The [repository](http://github.com/NixOS/nixpkgs) of nix expressions for all nix packages is publicly hosted.
If a package you need isn't already included, it's a matter of writing one file with a fairly simple syntax and opening a pull request to add it.
I just did that today to add the graph partitioning library [metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) to nix.
The maintainers are very active and merge pull requests daily.

So far, I've found that NixOS solves many of the more annoying problems with package management in Linux.
While there are many scientific libraries missing from `nixpkgs`, this can be fixed very easily and I intend to add several myself.
