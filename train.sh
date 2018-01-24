#!/usr/bin/env bash

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
export save=logs/sml_original_${RANDOM}${RANDOM}
mkdir -p $save
th train_sml.lua | tee $save/log.txt