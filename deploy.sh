#!/usr/bin/env bash

# This script allows you to easily and quickly generate and deploy your website
# using Hugo to your personal GitHub Pages repository. This script requires a
# certain configuration, run the `setup.sh` script to configure this. See
# https://hjdskes.github.io/blog/deploying-hugo-on-personal-github-pages/index.html
# for more information.

# Set the English locale for the `date` command.
export LC_TIME=en_US.UTF-8

# GitHub username.
USERNAME=mlnotebook
# Name of the branch containing the Hugo source files.
SOURCE=hugo
# The commit message.
MESSAGE="Site rebuild $(date)"
if [ $# -eq 1 ]
	then MESSAGE="$1"
fi

msg() {
    printf "\033[1;32m :: %s\n\033[0m" "$1"
}

msg "Pulling down the \`master\` branch into \`public\` to help avoid merge conflicts"
#git subtree pull --prefix=public \
#	git@github.com:$USERNAME/$USERNAME.github.io.git master -m "Merge master"

msg "Building the website"
hugo

msg "Pushing the updated \`public\` folder to the \`$SOURCE\` branch"
git add public
git commit -m "$MESSAGE"
git push origin "$SOURCE"

msg "Pushing the updated \`public\` folder to the \`master\` branch"
git subtree push --prefix=public \
	git@github.com:$USERNAME/$USERNAME.github.io.git master
