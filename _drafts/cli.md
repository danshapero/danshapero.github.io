---
layout: post
title: "Using Linux in console mode"
categories: linux yak-shaving
---


In my very first post, I detailed how I almost bricked a laptop while trying to run it in terminal-only mode.
I haven't given up on the idea of dispensing with decades of advancements in user interfaces, so in this post I'll show how to do more things from the command line.

# Booting into terminal mode

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


# Brightness

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


# Wifi

The next hurdle is dealing with wireless networks from the command line.
This can all be done with the command-line interface to network-manager, `nmcli`.

If you've already connected to a wireless network through graphical tools, the name and password are likely stored somewhere on your system.
You can see all the networks that you've already discovered with the command

`nmcli con list`.

You can then either connect or disconnect from one of these with

`nmcli con <up/down> id <NetworkName>`.

What about if you want to connect to a new wireless network from the terminal?
The commands

`nmcli dev wifi`

`nmcli dev wifi connect <NetworkName> password <NetworkPassword>`

will, respectively, scan and list all available wifi networks, and connect to a given network.

This is a big improvement over the old way of doing things with `wpa_supplicant`.
You had to know something about your network card drivers, you had to write to system files with `sudo`, you had to convert the ASCII password into the equivalent hexadecimal code; it was a mess.


# Fonts and colors

This is where the real yak-shaving began.
Entire herds of bald yaks, roaming the harsh landscape of the Mongolian steppe; and their sad little moos were most heartwrenching.

Remember before, when we were booting into the linux `tty`?
These virtual terminals are programs like any other -- they have a specific responsibility, namely to accept user input, give that input to a shell like bash and display the output.
However, they're coded directly into the Linux kernel; they don't live in "user space".
This is in contrast to graphical terminal emulators that you use in your desktop environment, like GNOME terminal.

Being coded directly into the Linux kernel, these terminals don't have a lot of the niceties of their graphical counterparts, chief among which is the ability to display truetype fonts or to use more than 16 colors.
While I'm all for minimalism, I can't live without a good font and color scheme that are easy on the eyes, especially in low light.

Instead of working directly from the Linux kernel console, one can instead add a thin layer of indirection by using a *framebuffer terminal emulator*.

#### The Linux framebuffer

The framebuffer is a device that resides in `/dev/fb<n>` which abstracts over monitors and other output devices.
The X server provides an interface to your monitor for graphical applications by taking their requests to draw objects, and writing to the framebuffer in order to render those objects.

However, other applications besides X can use the Linux framebuffer to render images.
For example, one could write a terminal emulator which, instead of using the X libraries, paints directly onto the framebuffer.
One would then still have the basic functionality of a Linux console, but be free to use more colors or fonts.
Several [framebuffer terminal emulators](http://unix.stackexchange.com/questions/20458/how-to-use-dev-fb0-as-a-console-from-userspace-or-output-text-to-it) already exist.
Many were developed by users who want to render Chinese, Japanese and Korean (CJK) characters at a terminal.
Additionally, user-space virtual terminals can write to other devices besides `/dev/fb0`, such as a [Braille terminal](http://mielke.cc/brltty/).

Of the existing FB terminal emulators, I tried three of them: [fbterm](https://code.google.com/p/fbterm/), [terminology](https://www.enlightenment.org/about-terminology) and [kmscon](http://www.freedesktop.org/wiki/Software/kmscon/), in that order.


#### `kmscon`

I ultimately settled on `kmscon`, after initially being deterred by having to always run it with super-user privileges.
However, its compatibility with the color scheme in `xterm` makes it preferable to `fbterm`, and its light weight puts it ahead of `terminology`.

A number of tweaks were necessary to ensure that `kmscon` worked properly.

First, if your OS is using `systemd`, you can tell it to start `kmscon` on `tty2` - `tty7` by default, leaving the usual Linux kernel console on `tty1`.
You could even have it start on every `tty`, but the thought of recovering from some device driver bug with no tried-and-true kernel console to fall back on is a frightening one indeed.

I found that on Arch Linux, rendering unicode glyphs failed without properly setting the `locale`.
This can be fixed with

`export LANG=en_US.UTF8`.