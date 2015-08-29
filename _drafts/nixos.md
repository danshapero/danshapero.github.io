---
layout: post
title: "Things I've had to fix in Nix"
categories: linux
---

I've recently started using the new and rather interesting [NixOS]() linux distribution.
NixOS takes a different approach to package management that makes dependency hell much less hellish.
Being a fairly new distribution, NixOS definitely has its rough edges and gotchas.
Here are some that I've found so far:

* Installing `clang-<version>` doesn't adjust your include or library search paths to find the C++ standard library and headers.
Instead, the right package is `clang-wrapper-<version>`.
* On my system, the software sources for the root user point to a stable version of `nixpkgs`, while my user account points to the master branch of the git repo for the latest versions.
If you install a newer version of the C/C++ compiler in a user account, the `cc` and `c++` executables will refer to the older versions installed by root.
Consequently, if you do install software from source you'll need to explicitly set the right compiler.
* Installing and using `nox` as root can create files (usually a cache of the output of `nix-env -qa`) in `/tmp/` which a future installation by a non-admin user will attempt to read and write to.
Since `root` owns these files, the user will lack the appropriate privileges and thus be unable to use `nox` at all.
Either delete these temporary files or don't use `nox` as root.
* Libraries that the user installs go in `/home/<user>/.nix-profile/lib`, and yet this directory isn't included in `LD_LIBRARY_PATH`.
When building other libraries from source, the linker won't find libraries installed by nix without adding the appropriate `-L` flag at link-time.
In my case, I was using CMake, so I had to set `CMAKE_SHARED_LINKER_FLAGS`.
* A better way of doing that is creating a nix-expression which tracks the dependencies needed to build the source in question.
You can then invoke `nix-shell`: this command fetches any missing dependencies and sets environment variables so that those dependencies can be found easily.
Within this environment, building the package should work.
The nix shell that you enter is temporary; that way, you can switch to different build environments for different packages.
* If the root account and the user account subscribe to different nix channels, invoking `nix-shell` will usually employ the root's channels.
Circumvent this by altering the `NIX_PATH` environment variable to e.g. `/home/<you>/.nix-defexpr/channels/`.
* When using the foreign function interface in Common Lisp, the usual directories that get searched (`/usr/lib`, `/usr/local/lib`) aren't in use at all.
Instead, you have to set the special variable `cffi:*foreign-library-directories*` to include your `/home/<user>/.nix-profile/lib/`.

  Note that the final `/` matters.
Guess how long it took me to figure out that little tidbit.