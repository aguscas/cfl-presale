#!/bin/bash

IPEN0=$(ifconfig en0 | grep "inet " | awk '{print $2}')
echo $IPEN0
export IPEN0
