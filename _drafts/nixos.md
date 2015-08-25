---
layout: post
title: "NixOS is neat, also weird"
categories: linux
---

[NixOS](http://nixos.org) is an operating system built around the rather unusual `nix` package manager.
Many package managers, such as `aptitude`, are built off of `dpkg`, and thus inherit its quirks and limitations.
For example, only users with root access can install packages, whereas the rest of us are left compiling from source and manually specifying library and header locations to dependencies.
Moreover, only one version of a package can be installed at a given time.
This is especially frustrating when different packages depend on different versions of the same dependency.

On the other hand, `nix` aims at a "purely functional", declarative approach to package management.
With this approach, users without administrative privileges can install and use libraries in their home directory, and multiple versions of the same package can safely coexist.
Additionally, packages in `nix` and how to build them are specified in its own domain-specific language.
The [repository](http://github.com/NixOS/nixpkgs) of all `nix` packages is publicly hosted.
If a package you need isn't already included, it's a matter of writing one file with a fairly simple syntax and opening a pull request to add it.
I just did that today to add the graph partitioning library [metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) to nix.
The maintainers are very active and merge pull requests daily.

So far, I've found that NixOS solves many of the more annoying problems with package management in Linux.
While there are many scientific libraries missing from `nixpkgs`, this can be fixed very easily and I intend to add several myself.
That said, nix is fairly new and not without its fair share of gotchas, a list of which will be kept here and updated as I find more.

* Installing `clang-<version>` doesn't adjust your include or library search paths to find the C++ standard library and headers.
Instead, the right package is `clang-wrapper-<version>`.
* I set the software sources for the root user to those for a stable version of NixOS, while my user account uses the bleeding-edge sources.
If you install a newer version of the C/C++ compiler in a user account, the `cc` and `c++` executables will refer to the older versions installed by root.
Consequently, if you do install software from source you'll need to explicitly set the right compiler.
* Installing and using `nox` as root can create files (usually a cache of the output of `nix-env -qa`) in `/tmp/` which a future installation by a non-admin user will attempt to read and write to.
Since `root` owns these files, the user will lack the appropriate privileges and thus be unable to use `nox` at all.
Either delete these temporary files or don't use `nox` as root.
* Libraries that the user installs go in `/home/<user>/.nix-profile/lib`, and yet this directory isn't included in `LD_LIBRARY_PATH`.
When building other libraries from source, the linker won't find libraries installed by nix without adding the appropriate `-L` flag at link-time.
In my case, I was using CMake, so I had to set `CMAKE_SHARED_LINKER_FLAGS`.
* A better way of doing that is creating a nix-expression which tracks the dependencies needed to build the source in question.
You can then invoke `nix-shell`.
This command fetches any missing dependencies and sets environment variables so that those dependencies can be found easily.
Within this environment, building the package should be straightforward.
* When using the foreign function interface in Common Lisp, the usual directories that get searched (`/usr/lib`, `/usr/local/lib`) aren't in use at all.
Instead, you have to set the special variable `cffi:*foreign-library-directories*` to include your `/home/<user>/.nix-profile/lib/`.
Note that the final `/` matters.
Guess how long it took me to figure out that little tidbit.