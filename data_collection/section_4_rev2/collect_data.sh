echo `basename "$PWD"`

cd sv/
bash collect_data.sh

cd ../th/
bash collect_data.sh

cd ..
