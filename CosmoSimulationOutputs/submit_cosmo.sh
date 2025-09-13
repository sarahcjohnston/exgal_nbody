#!/bin/bash -l

./swift --cosmology --self-gravity --power --threads=16 parameter_cosmo.yml  > output.log 2> error.log

