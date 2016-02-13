Notes on Melting
================

Michael Mueller's last group meeting, Wed 6 May 2015
----------------------------------------------------

Things to consider

 - Manhattan distance is *not* shell ("valency")
 - width of peaks in MSD/speed vs decay elsewhere
 - rescale by some time to collapse w.r.t.\ initial config
 - effects from startup of shaker  
    also look in:
     - single particle
     - lower density crystal
 - random config is matches different configurations for $\rho$ vs for $\psi$
 - MSD averaged over all times vs. shell
 - remove zeros from densities or find alternative:
    - $1 / <d^2>$ averaged over vor. neighbors
    - convex hull
 - MSD on single particles

Debriefing, Thu 7 May 2015
--------------------------

### Files ###

Videos (tiffs, cines) organized by crystal size (9x9 not as good)

All videos taken at 

 - 120 fps
 - 50 Hz
 - 100 mV

Directory `original-data` has Sarah's particles:

 - walker
 - spinner
 - isotropic

### Particles ###

 - `Michael v 1.0`  
    final version that michael used for the analysis with posts moved in and
    bumps for marking

    - stub height: 0.33 mm
    - post height: 1.0 mm
    - width: 6 mm
    - base thickness: 2 mm
    - Probably the same as `large-based-moved-in-posts`?

### Analysis ###

`positions.py`

 - uses `remove_disks`
 - generates `POSITIONS.txt` file that saves ecc/size parameters
 - detection parameters used for center dots
     - 12x12: kernel -3, area 5--50, ecc 0.8
     - 5x5: kernel -3.5, area 10--100, ecc 0.8

`tracks.py`

 - does tracking
 - kills duplicates
 - uses first frame for some info: make sure it finds all particles and no extras by counting number of particles  

`tracks_to_pos.py`

 - adds an `'id'` column?
 - then `POSITIONS` is final

`analysis.py`

 - computes all the statistics
 - generates npz with all stats as arrays
 - argument is just a prefix (file base): `nxn_configuration`
 - different scripts to generate plots (`s` to save)

`avg_histograms.py`

 - makes curve from hist

