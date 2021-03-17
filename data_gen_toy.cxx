for i in {1..16};
  do
  	python -W ignore generate_data.py $i 1250 0>&1 2>&1 &
  	done;
wait %{1..16};
echo all runs complete
