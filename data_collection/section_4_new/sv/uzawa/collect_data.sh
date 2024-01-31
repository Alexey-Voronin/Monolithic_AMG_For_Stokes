echo `basename "$PWD"`
nice -20 python -O collect_data.py unstructured/2D & 
nice -20 python -O collect_data.py structured/2D_bfs
