# Rayleigh-Benard Convection GQL Test

Test against Curtis Saxon's extensive RBC GQL data

## Notes
* `fingerprint.py` is Curtis's original script. It uses the same equation formulation as the d2 RBC example.
* `rayleigh_benard_2.py` is my script to reproduce it with our new GQL tools.
* Need to use MCNAB2 and a `max_dt` greater that about 1 to reproduce Curtis's fast approach to the straight roll solution
