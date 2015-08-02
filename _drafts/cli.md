---
layout: post
title: "Using Linux in console mode"
categories: linux yak-shaving
---


In my very first post, I detailed how I almost bricked a laptop while trying to run it in terminal-only mode.
I haven't given up on the idea of dispensing with decades of advancements in user interfaces, so in this post I'll show how to do more things from the command line.

Booting into terminal mode
==========================

In order to try out working terminal-only mode without permanently altering your system, you can type `ctrl + alt + f2` to get to `tty2`.
What this does is switch you over to a different controlling teletype terminal.
Usually, when your computer boots into X, it does so on `tty7` or `tty8`.
In order to switch back to your usual desktop, type `ctrl + alt + f7` or `f8` as the case may be.

Once you've switched to a different virtual terminal, you can log in and see how you like working only from the command line.
Depending on which text editor you use, many of the tasks of an average programmer can be done entirely from the terminal, often with less clutter than using a window manager.
If you like working this way, you can tell your machine to boot into text-only mode by default.
When you want to use your window manager or GUI applications, you can always do

`startx`

at the command line.

I'm using the Grand Unified Bootloader (GRUB), so to boot into text-only mode I have to modify `/etc/default/grub` by altering the line

`GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"`

to

`GRUB_CMDLINE_LINUX_DEFAULT="quiet text"`.


Brightnes
=========

Since I'm using a laptop, I want to modify the screen brightness while in terminal mode.
In X, there's usually a simple key combination to increase or decrease the brightness, but this will no longer work in terminal-only mode.
On my machine, the screen brightness is controlled by the file

`/sys/class/backlight/intel_backlight/brightess`.

By writing a number to this file, we can change the screen brightness like so:

`sudo su -c "echo 500 > /sys/class/backlight/intel_backlight/brightness"`

We had to use super-user privileges to run this command, since we're writing to a system file.
It's kind of annoying to have to use `sudo` and type in a password whenever I want to change the screen brightness.
Although it's usually inadvisable to change the permissions of systme files, this one is pretty harmless, so one could try

`chmod a+w /sys/class/backlight/intel_backlight/brightess`

to allow every user to write to this file.

Unfortunately, it's not that easy. Everything in `/sys/` is dynamically generated when the computer boots, depending on which devices are or are not available. Consequently, we'd have to change the permissions with `chmod` on every boot.
However, that can be done automatically by adding the last command to the file `/etc/rc.local`.
This script contains commands that are executed just after booting, with admin privileges.
Finally, to streamline things even further, you can write a shell script which will increment or decrement the screen brightness by a fixed amount, rather than type the long `echo` command into the shell every time.


Wifi
====

The next hurdle is dealing with wireless networks from the command line.