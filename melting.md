Notes on Melting
================

Things to consider
------------------

 - Manhattan distance
 - width of peaks in MSD/speed vs decay elsewhere
 - rescale by some time to collapse w.r.t. initial config
 - startup of shaker: single particle, lower density crystal
 - random config is matched differently for $\rho$ vs for $\psi$
 - MSD averaged over all times vs. shell/manhattan #
 - remove zeros from densities (or use $1 / <d^2>$ averaged over vor.
   neighbors
 - MSD on single particles

Lacie Drive
-----------

All videos:
 - 120 fps
 - 50 Hz
 - 100 mV

Organized by crystal size (9x9 not as good)
`original-data`: sarah's particles (walker, spinner, isotropic)

Particle design
---------------

SCAD files
"Michael v 1.0" - final version that michael used for the analysis,
with posts moved in and bumps for marking
stub height: 0.33 mm
post height: 1.0 mm
width: 6 mm
base thickness: 2 mm
Probably the same as `large-based-moved-in-posts`?

Analysis
--------

### `positions.py` values:

for 12x12:

 - kernel: -3
 - area: 5--50
 - ecc: 0.8

 for 5x5:

 - kernel: -3.5
 - area: 10--100
 - ecc: 0.8

### workflow

`positions.py` uses `remove_disks`
generates `POSITIONS.txt` file that saves ecc/size parameters  
`tracks` does tracking and kills duplicates  
`tracks_to_pos.py` adds an "id" column?  
`tracks` uses first frame for some info: make sure it finds all particles and no extras by counting number of particles  
then `positions` is final  

### `analysis.py` computes all the statistics

generates npz with all stats as arrays  
argument is just a prefix (file base): `nxn_configuration`  
different scripts to generate plots (`s` to save)  
`avg_histograms` makes curve from hist  


