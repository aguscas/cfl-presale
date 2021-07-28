#!/bin/zsh

[[ $1 == "1" ]] && echo "Running 1" && socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\" &
[[ $1 == "2" ]] && echo "Running 2" && open -a Xquartz
