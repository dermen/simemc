#!/bin/bash
curdir=$PWD
export CCTBXROOT=$PWD/../../
cd $CCTBXROOT
git clone https://gitlab.com/kirianlab/reborn.git
ln -sv $PWD/reborn/reborn modules/.
cd $curdir
libtbx.python -c "from reborn.misc import interpolate"

