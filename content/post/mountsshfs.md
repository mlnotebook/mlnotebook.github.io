+++
date = "2018-05-21T16:57:32+01:00"
title = "Permenantly Mounting a Remote Drive with sshfs on Linux"
description = "A guide to mounting remote drives that require ssh access"
topics = []
tags = []
draft = true

+++

You're at work and are happily coding away saving and opening your scripts from the remote drive which is conveniently auto-mounted on your work computer. However, when you get home and that great idea comes to you, you have to connect to the work's VPN (because the drive is on a private network) and then `ssh` to your computer before you can load up the files... usually in some terminal-based editor like `nano`. This post shows how to permenantly mount remote drives so you can open your files locally in whatever editor you like.

<!-- more -->

<h2 id="intro"> Introduction </h2>

In my workflow, I like to use Sublime as my editor (because it's clean and colorful. So I have an issue when I'm at home and I want to load up my scripts into my local Sublime - I can't (well, actually I can because Sublime has a 'remote server' plugin, but it's not great as it requires a lot of clicking around).

So, I want to load my remote drives locally so I can just open the files as I would in the office. I found that `sshfs` was the command for this but also found that it wasn't persistent. Furthermore, other guides online kept missing out bits that I needed, so I've written this to fill those in.

<h2 id="sshfs"> sshfs </h2>

First, let's install `sshfs`. At the terminal, run the command (as super-user):

```bash
sudo apt-get install sshfs
```

Done. Nice and easy!

<h2 id="fstab"> /etc/fstab </h2>

So this is what we need to edit to mount remote drives and do it persistently. `fstab` needs to opened as super user in order to save the changes. We'll open it in `nano`:

```bash
sudo nano /etc/fstab
```

The basic syntax for `sshfs` is:

```bash
sshfs user@host:/remotefolder localfolder fuse -o option1 option2 optionN 0 0
```

There are a couple of changes to this needed for it to work in `fstab`. Remove the first space and concatenate the options:

```bash
sshfs#user@host:/remotefolder localfolder fuse option1,option2,optionN 00
```

* `user`: your username on the remote computer
* `host`: the computer you're trying to connect to (make sure to use the full address `machinename.domain.com`
* `remotefolder`:  don't miss the `:` between the `host` and `remotefolder`. This is the folder on your work machine that you want to mount on your local computer (i.e. the remote drive)
* `localfolder`: This is the place on your machine that you want the drive to be mounted to
* `fuse`: is the filesystem type we're using for `sshfs`
* `option1`: these are a list of options we want sshfs to use (important!)
* `0 0`: required for this to work

Lets make the local mount point. It needs to be created manually for the mounting to work so as an example, lets mount the drive in the `/vol` folder and call the mount `remotedrive`:

```bash
sudo mkdir /vol/remotedrive
```
Next, let's take care of the options:

* `reconnect` - this need to be active to make the drive mount
* `allow_other` - allows you to access the folder even though you're technically not the same user at the one at work (even if you use the same username on your local machine)
* `default_permissions` - sets all the permissions to default (read/write)
* `nonempty`
* `ServerAliveInterval` - checks every now and again that the drive is still mounted by keeping the connection open (so it doesn't time out)
* `IdentifyFile` - this is useful, it uses the ssh-key instead of a password to mount the drive so you won't keep getting asked to type the password
* `uid` - important if you also don't want to keep putting in your password every time you want to save something! Ensure that you have the same permission as the remote user

So this will go something like:

```bash
sshfs#user@host:/remotefolder localfolder fuse reconnect,allow_other,default_permissions,nonempty,ServerAliveInterval=15,IdentifyFile=/home/user/.ssh/id_rsa,uid=1000 00
```
I chose `15` for the `ServerAliveInterval`. For all of this to work, you need to have your ssh-key in `/home/user/.ssh/` which most of us usually do: if you're not sure about the whole `ssh` thing and 'keys' have a quick google, it's very straight-forward.

You also need to check that your `uid` is really `1000`. Simply use:

```bash
echo $UID
```
If it's not, `1000`, just change that in your `fstab`.

Finally, we need to make sure that we have `fuse` sorted out and that we're part of the `fuse` group. Replace `user` with your username.

```bash
sudo gpasswd -a user fuse
```















