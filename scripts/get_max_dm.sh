#!/bin/bash

f_mid=$1
t_int=$2

D="0.004148808" # Dispersion constant in GHz^2 pc^-1 cm^3 s
f_high=$(bc <<< "scale=4; ($f_mid+168)/1000")
f_low=$(bc <<< "scale=4; ($f_mid-168)/1000")

max_dt=$(bc <<< "scale=5; ($t_int * 4096 * 10^(-3))")
max_dm=$(bc <<< "scale=5; $max_dt / ($D * ($f_low^(-2) - $f_high^(-2)))")

echo $max_dm