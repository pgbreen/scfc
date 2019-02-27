Self-Consistent Field Code
============================

Pure Python implementation of self-consistent code (see [Hernquist & Ostriker 1992](http://adsabs.harvard.edu/abs/1992ApJ...386..375H))

Example:

```sh
$./scf.py -i inputfile -lmax 2
# Time 	Energy 	Kinetic   Potentail Amp-m2     L_z     L_tot    time_seconds 
 0.0 -0.2505358 0.2482453 0.4987811 8.6801E-03 1.3882E+00 0.0015663 0.0024844 1.053 3.363 1.116   0.2 
 1.0 -0.2504870 0.2491536 0.4996407 1.6117E-02 -4.2685E-01 0.0015663 0.0034391 1.061 3.354 1.120   1.3  
```
 - Input file assumed to be of form m1 x1 x2 x3 v1 v2 v3 (currently assumes masses are identical but easily updatable to multi-mass). 
 - lmax maximum degree of spherical harmonic, also includes degrees up to lmax



Options:

```sh
$./scf.py -h
usage: scf.py [-h] [-i I] [-lout LOUT] [-aout] [-rb] [-nr] [-tout] [-tsnap]
              [-dt] [-tend] [-rcut] [-lmax LMAX]

optional arguments:
  -h, --help  show this help message and exit
  -i I        file name
  -lout LOUT  label output files default to name of input
  -aout       output projection coeff
  -rb         rbasis - rescale radial functions
  -nr         number of radial basis functions
  -tout       output time
  -tsnap      output time for snap shot, save to file e.g. f128ksnapt0.dat
  -dt         time step
  -tend       termination time
  -rcut       rut off value for calculating inertia tensor
  -lmax LMAX  maximum value of l includes all m < l (ignored if not set)
```

Reasonable performance up to 100K (scheme is embarrassingly parallel so you can go to much much high N if you move to C++ with OpenMP, CUDA etc)
