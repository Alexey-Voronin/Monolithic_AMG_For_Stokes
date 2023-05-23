# Data Collection Scripts

This repository contains scripts for collecting data. To collect all the data, execute the following command: `bash launch.sh`. 

For collecting data for a specific example, navigate to the respective directory and run the same command.

## Abbreviations

### Directory Abbreviations

The following table provides a list of directory abbreviations along with their descriptions:

| Abbreviation | Description                                 |
| :-----------:|:--------------------------------------------|
| th           | Taylor-Hood discretization                   |
| sv           | Scott-Vogelius discretization                |
| amg_p2p1     | Monolithic AMG applied directly to P2/P1     |
| amg_isop2p1  | Defect-correction based monolithic AMG       |
| hlo          | Relaxation on higher and lower order discretizations (levels 0 and 1)|
| ho           | Relaxation on higher order discretizations (level 0)|
| lo           | Relaxation on lower order discretizations (level 1)|

### File Abbreviations

The following table provides a list of file abbreviations along with their descriptions:

| Abbreviation         | Description                                                |
| :-----------:        |:-----------------------------------------------------------|
| problem_iterator.py  | Creates an iterator object over different size Stokes problems|
| mg_params.py         | Contains solver parameters                                  |
| disc.py              | Describes problem discretization and the auxiliary operator needed for the solver|
| collect_data.py      | Combines the above-described information to perform strong-scaling data collection|

