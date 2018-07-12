# Processing info in shell

FILES=./lab3/data/train/*.wav
for i in $FILES
do
	a=${i##*/}
	a=${a%.*}
	echo $a
	./lab3/opensmile-2.3.0/SMILExtract -C ./lab3/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf -I $i -O trainfeatures_${a}_gemaps.csv

	# CALL .PY MAKE SURE IT ITERATES LINES
	#python ./text_process.py


	# Delete gemaps.csv file

done
