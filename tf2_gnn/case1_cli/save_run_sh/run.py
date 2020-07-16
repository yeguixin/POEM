rm -rf trained_moled out.log embed*

for i in {0..9} ;do
	python -u train.py GGNN GraphBinaryClassification /dev/shm/zjq/case1_data/amd/new_${i}/ast >> out.log
done

