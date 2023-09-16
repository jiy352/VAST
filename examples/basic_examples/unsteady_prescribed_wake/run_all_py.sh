echo "Test Summary" > test_summary.txt
echo `date` >> test_summary.txt;
for f in ex_*.py; do python -W "ignore" "$f" >> test_summary.txt; done


echo "Test passed" >> test_summary.txt;
grep -wc "passed!" test_summary.txt >> test_summary.txt;
echo "Total tests" >> test_summary.txt;
grep -wc "nonlinear" test_summary.txt >> test_summary.txt;