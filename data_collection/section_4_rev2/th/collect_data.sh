echo $(basename $PWD)

cd uzawa

cd ../amg_isop2p1/
bash collect_data.sh

cd ../amg_p2p1/
bash collect_data.sh

cd ..
