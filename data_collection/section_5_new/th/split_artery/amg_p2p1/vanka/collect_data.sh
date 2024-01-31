echo $(basename $(dirname $PWD))"/"$(basename $PWD)
nice -20 python -O collect_data.py unstructured/3D
