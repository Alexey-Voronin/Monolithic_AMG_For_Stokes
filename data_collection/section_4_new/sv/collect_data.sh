echo `basename "$PWD"`

cd defect_correction/
bash collect_data.sh

cd ../uzawa/
bash collect_data.sh

cd ..
