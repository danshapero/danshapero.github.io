---
layout: post
title: "Inaugural post: how I almost destroyed a new laptop"
date: 2013-05-09 02:59:00
categories: stupidity tty
---

I write and use open-source software under Linux in pursuit of my PhD in applied math.
In the process, I have done some staggeringly moronic things.
Hopefully, my mistakes, which are legion, can be a cautionary tale to others.

And on that note, let's start with how I almost destroyed a fancy new laptop.
I use the text editor emacs (see: [real programmers use emacs](http://xkcd.com/378/)) for writing programs and run them from a terminal, and don't often use the actual desktop environment save for playing music.

![The author, at work](http://img02.deviantart.net/7178/i/2011/029/8/3/emacs_user_at_work_by_earlcolour-d38aj2x.jpg)

Consequently, I thought it would be nice if my computer only booted into command line-only mode; if I want to use X-windows, I can do `startx` to get my window manager back.
This way, I can still get all my work done, and without a desktop environment, my laptop would hopefully be drawing less power.
Moreover, Linux users, and mathematicians in particular, consider the endurance of immense aggravation to be a mark of high moral character.

Googling for "linux boot to terminal only" revealed the following command:

```
sudo update-rc.d mdm disable
```

which would change the right system file so that the Linux Mint desktop manager was disabled on startup.
Ever cautious, I tried this on an old machine with the expected results: boots to terminal, `startx` to get the desktop environment.

Since that worked just fine, I felt empowered enough to try it on the new machine.
When I restarted, short of finding a command prompt, the screen was totally black.
A yawning abyss, staring back at me.

So, the screen failed, but maybe the computer is still responding to commands.
I thought that, if I enter my username and password and type `startx`, it would restore the desktop environment and I could undo this nightmare.
No dice.

My next option was to try and boot from a USB stick.
Usually, you do this because you want to install or re-install an operating system.
A drastic measure, but effective.
After booting from the same USB stick I used to install Linux Mint in the first place, a black screen again.

Last resort: boot from a Windows recovery disk, which I am ashamed to admit I even made.
Result? Black screen.

That's when real panic set in, followed by the revelatory comprehension of the magnitude of my own stupidity.
The towering, endless cliffs of stupidity, stretching up as far as the eye can see.

As per normal, the solution to my problem was totally mundane and relates to which `tty` is being used.
In the olden days, you interacted with a computer through a teletype machine.

![tty](http://startup.nmnaturalhistory.org/content/images/artifacts/36_l.jpg)

You put in your commands at the keyboard and got some [punched tape](https://en.wikipedia.org/wiki/Punched_tape) from the strip at the left, feed that tape into the computer, and the machine outputs the results back to the printer.
You won't find one of these outside of a museum now, but internally your computer is still emulating a teletype machine.

When later operating systems were freed from the explicit dependence on hardware, they were able to emulate not just one but multiple virtual terminals at the same time.
These are numbered `tty1`, `tty2`, and so forth.
(Some good uses for your many virtual terminals are enumerated [here](https://mostlylinux.wordpress.com/troubleshooting/ttysessions/).)
To switch between them, you can type

```ctrl + alt + f1```

or `f2`, `f3`, etc.
(Try it now!)
Usually, `tty7` is reserved for the X-server.
This is the program that manages your windowing system -- the way that you usually interact with your computer through a graphical user interface.

Sometimes, when you terminate X-windows, you get sent to `tty7` instead of `tty1`.
To fix my black screen, all I had to do was enter `ctrl + alt + f1` to switch to teletype 1 and, presto change-o, everything was back to normal.
Close call.