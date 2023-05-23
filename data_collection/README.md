# Data collection scripts

To collect all the data run `bash launch.sh`. 
To collect data for a specific example perform the same command in the
respective directory. 

## Abrivations

### Directory 
| Abbrivation                  | Description |
| :-----------:                |:------------|
| th                           | Taylor-Hood discretization    |
| sv                           | Scott-Vogelius discretizaton  |
| amg_p2p1                     | monolithic AMG applied directly to P2/P1 |
| amg_isop2p1                  | defect-correction based monolithic AMG |
| hlo                          | relaxation on (h)igher and (l)ower order discretizations (levels 0 and 1)|
| ho                           | relaxation on (h)igher order discretizations (levels 0)|
| lo                           | relaxation on (l)ower order discretizations (levels 1)|

### Files
| Abbrivation                  | Description |
| :-----------:                |:------------|
| problem_iterator.py          | creates iterator object over different size Stokes problems  |
| mg_params.py                 | contains solver paramters  |
| disc.py                      | describes problem discretization and the auxilary operator needed for the solver|
| collect_data.py              | combines the above described infromation to perform strong-scaling data-collection |
