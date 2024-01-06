echo $(basename $PWD)

cd th/  
bash collect_data.sh

cd ../sv/
bash collect_data.sh

cd ..

