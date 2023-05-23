echo $(basename $(dirname $PWD))"/"$(basename $PWD)
nice -20 python -O collect_data.py structured/3D 
nice -20 python -O collect_data.py unstructured/3D

nice -20 python -O collect_data.py structured/2D_bfs
nice -20 python -O collect_data.py unstructured/2D
