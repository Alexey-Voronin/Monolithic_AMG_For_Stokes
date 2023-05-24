
currpath=`pwd`

todos=($(find . -type d -links 2 | grep -v '__'))
for i in ${todos[@]}
do
	echo $i
	cd $i
	echo `pwd`
	python ../../collect.py
	cd $currpath
done
