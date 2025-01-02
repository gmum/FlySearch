python3 viscode.py $1 $2 >$1.tex
./compile_appendix.sh $1
/bin/ls build | grep -v .pdf | xargs -I {} rm build/{}
rm $1.tex
