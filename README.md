# Accretion disk solver
This script allows easy access to the disk Equations proposed by Lipunova+99 (which include the seminal thin-disk solution of Shakura & Sunyaev+1997).
The script solves the differential Equations for four types of disk:
Conservative and non-advective disk (the analytical Shakura & Sunyaev 73 thin disk solution)
Non-conservative and non-advective disk ("Slim" disk with Outflows, although it is solved numerically, an analytical solution exists as shown in Lipunova+99)
Conservative and advective disk ("Slim" disk without outflows, non-analytical solution exists, numerically solved)
Non-conservative and advective disk ("Slim" advective disk with outflows, non-analytical solution exists, numerically solved)

The code solves the disk Equations and can be used to retrieve quantities of interest for observers such as the height, density, radial velocity, radiative output, etc.
