Self-Consistent Field Code
============================

Python implementation of self-consistent code (see [Hernquist & Ostriker 1992](http://adsabs.harvard.edu/abs/1992ApJ...386..375H))

Example:

```sh
$./scf -i inputfile -lmax 2
```
 - Input file assumed to be of form m1 x1 x2 x3 v1 v2 v3 (currently assumes masses are identical but easily updatable to multi-mass). 
 - lmax maximum degree of spherical harmonic, also includes degrees up to lmax
 - use -h flag for other options

Reasonable  performance up to 100K (scheme is embarrassingly parallel so you can go to much much high N if you move to C++ with OpenMP, CUDA etc)
