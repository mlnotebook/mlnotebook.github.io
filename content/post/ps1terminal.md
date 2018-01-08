+++
date = "2017-08-08T10:05:14Z"
description = "Making easier to identify which computer we're working on when we `ssh` a lot"
tags = ['linux']
title = "Modifying the Terminal Prompt for Sanity"
topics = ['linux', 'productivity']
social = true
featured_image='/img/ps1img.png'
+++

If you're working with more than one computer at a time, then you're probably using some form of remote access framework - most likely `ssh`. This is common in machine learning where our scripts are run on some other host with more capabilities. In this post we'll look at how to modify the terminal prompt layout and colours to give us information we need at a glance: the current user; whether they're `root`; what computer we're working on; what folder in and the time that the last command was given.

<!--more-->

When we `ssh` into another computer, the terminal prompt will most likely change. Often it becomes colourless (usually all-white text) and the structure may change based on the initial setup. I've often issued commands to the wrong computer because of this so it would be useful if we were able to clearly see which computer we're working on at a glance.

Many users don't know that they can edit their terminal prompt *without root privileges* to give them better indications of their user, host and location. This is done by editing the `PS1` variable in the `~/.bashrc` file. `~/.bashrc` (where `~` is the shortcut to our `/home/<username>` folder and `.` indicates a hidden file) is a set of commands that is run every time a new terminal window is opened. This has a lot to do with how the terminal window functions as well as `alias` shortcuts to longer commands. We edit this with an editor like `nano`:

~~~bash
nano ~/.bashrc
~~~

The first thing we will do is to make sure that whenever we are in a terminal window (`ssh` or otherwise) as the current user, we are seeing colours in the terminal - this is useful for certain text editors as well as the prompt. Find the line that is currently commented out, and uncomment it:

~~~bash
# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
force_color_prompt=yes
~~~
Now for the prompt. In this file, we need to find the line where the PS1 format is defined. PS1 is the name for the terminal prompt. It should be a couple of blocks after the `force_color_prompt` variable.

~~~bash
if [ "$color_prompt" = yes ]; then
    PS1='\A [\[\e[0;36m\]\u\[\e[0m\]@\[\e[1;36m\]\h\[\e[0m\]:\w\[\$ '
else
    PS1='\u@\h:\w\$ '
fi
~~~

Here you'll see the differece that the `force_color_prompt` variable makes: there is a lot more formatting code in the `true` part of this `if` block that adds color. The above formatting creates the prompt below from one of my machines:

<div style="width:100%; text-align:center;">
<div style="text-align:center; display:inline-block; width:100%; margin:auto;min-width:325px;">
<img title="Natural Image RGB" width=700px src="/img/ps1/exampleuser.png" ><br>
<b>Example terminal prompt for regular user account</b>
</div>
</div>

I'll identify the different components here, but you can find a list of all of the possible elements that can be included [here](https://ss64.com/bash/syntax-prompt.html 'PS1 Prompt Variables').

* `\A` - the current time in `hh:mm` format
* `\u` - the current user
* `\h` - the current host
* `\w` - the current working directory
* `\$` - the $ character (if it's not escaped, the shell reads this as if it's trying to find a variable as in `$PATH`)

Any characters which are not escaped (i.e. preceeded by backslash '`\`') are printed as they appear: e.g. `@` and `$`. Assigning the PS1 variable the value: '`\A \u@\h:\w\$`' we get `time user@host:/directory$' like this:

<div style="width:100%; text-align:center;">
<div style="text-align:center; display:inline-block; width:100%; margin:auto;min-width:325px;">
<img title="Natural Image RGB" width=700px src="/img/ps1/plainexample.png" ><br>
<b>Example terminal prompt with no formatting</b>
</div>
</div>

In order to get colors in the prompt, we need to surround our variables e.g. '`\A`', with some (very ugly) specific syntax. Where we want the color to start, we write '\[\e[0;XXm\]' and where we want to finish the colour and return to normal, we can write '\[\e[0m\]'. The 'XX' in the first term is a 2-digit code that refers to a color. For example, to make the username green, we change the PS1 variable to this:

~~~bash
PS1='\A \[\e[0;32m\]\u\[\e[0m\]@\h:\w\$ '
~~~

<div style="width:100%; text-align:center;">
<div style="text-align:center; display:inline-block; width:100%; margin:auto;min-width:325px;">
<img title="Natural Image RGB" width=700px src="/img/ps1/greenuser.png" ><br>
<b>Example terminal prompt with green username</b>
</div>
</div>

A list of colors and their respective numbers can be found [here](https://unix.stackexchange.com/a/124408 'PS1 Prompt Colors'). I choose green if we're logged in as a regular user (as in green for go) but I choose red if the user is `root`. This means I can always see at a glance if I should be careful with the commands that I write.

You'll also notice that we can change the *style* of the font along with the color. I find this useful for making the `host` standout by making it bold. This is done by changing the `0` before the 'XX' color to `1`.

~~~bash
if [ "$color_prompt" = yes ]; then
    PS1='\A [\[\e[0;31m\]\u\[\e[0m\]@\[\e[1;36m\]\h\[\e[0m\]:\w\[\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
~~~

<div style="width:100%; text-align:center;">
<div style="text-align:center; display:inline-block; width:100%; margin:auto;min-width:325px;">
<img title="Natural Image RGB" width=700px src="/img/ps1/exampleroot.png" ><br>
<b>Example terminal prompt for a `root` user</b>
</div>
</div>

For my full PS1 variable, I have the colours, the bold host and I also added some square brackets (not escaped!) to make it a little more visually pleasing. You can change the `~/.bashrc` file for each user on each computer. So if you have a regular user account *and* a root account on the same machine, you can create a different PS1 for both by editing their respecitve files. So feel free to change colours and formats as you wish!



