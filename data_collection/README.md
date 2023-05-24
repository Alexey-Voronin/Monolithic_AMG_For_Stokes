# Monolithic Algebraic Multigrid Preconditioners for the Stokes Equations

This repository supports an academic paper focused on Monolithic Algebraic Multigrid Preconditioners for the Stokes Equations. Here, you'll find scripts used for data collection and plot generation, and definitions for key abbreviations. 

## Data Collection Scripts

Scripts for data collection are included in this repository. To collect all the relevant data, use the following command: `bash collect_data.sh`.

To collect data specific to an example, please navigate to the respective directory and run the same command there.

## Plotting Scripts

You'll find Python plotting scripts in each subdirectory of `data_collection` and `parameter_search`. These scripts generate the figures that appear in the manuscript.

## Abbreviations

We have used several abbreviations throughout our directory structure and in our files. Here's what they mean:

### Directory Abbreviations

| Abbreviation | Description |
| :-----------:|:------------|
| th           | Taylor-Hood discretization |
| sv           | Scott-Vogelius discretization |
| amg\_p2p1     | Monolithic AMG applied directly to P2/P1 |
| amg\_isop2p1  | Defect-correction based monolithic AMG |
| hlo          | Relaxation on both higher and lower order discretizations (levels 0 and 1)|
| ho           | Relaxation on higher order discretizations (level 0)|
| lo           | Relaxation on lower order discretizations (level 1)|

### File Abbreviations

| Abbreviation         | Description |
| :-----------:        |:------------|
| problem\_iterator.py | Creates an iterator object for different size Stokes problems |
| mg\_params.py        | Contains the solver parameters |
| disc.py              | Describes the problem discretization and the auxiliary operator needed for the solver |
| collection\_data.py  | Combines the information described above to perform strong-scaling data collection |
| collection\_data.sh  | Shell script that traverses subdirectories and launches Python data collection scripts |

