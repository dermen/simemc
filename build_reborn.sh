#!/bin/bash
curdir=$PWD
export CCTBXROOT=$HOME/xtal_gpu3
cd $CCTBXROOT
git clone https://gitlab.com/kirianlab/reborn.git
ln -s $PWD/reborn/reborn modules/.
cd $curdir
libtbx.python -c "from reborn.misc import interpolate"

