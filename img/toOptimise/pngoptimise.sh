#!/bin/bash

## Overwrite images ##
## Keep file system permission and make a backup of original PNG (see options below)  ##
for i in *.png; do optipng -o5 -quiet -keep -preserve -log optipng.log "$i"; done
