#!/usr/bin/env python

import numpy as np

def local_bond_orientional(positions, m):
    """ local_bond_orientational(positions, m)
    
        Returns the list of local m-fold bond-orientational order parameters

             _      1    Nj     (i m theta  )
        phi (r ) = ---  SUM   e           jk
           m  j     Nj  k=1
    """
    return


def local_positional(positions, m):
    """ local_positional(positions, m)
        
        Returns the list of local m-fold positional order parameters
                        _   _ 
              _       i G * r
        zeta (r ) = e    m   j
            m  j   
    """
    return

def global_particle_orientational(orientations, m=4):
    """ global_particle_orientational(orientations, m=4)
        
        Returns the global m-fold particle orientational order parameter

                1   N    i m theta
        phi  = --- SUM e          j
           m    N  j=1 
    """
    return
