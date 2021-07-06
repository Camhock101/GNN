for i in {1..16};
  do
  rat pos_noise.mac -o $i.root 0>&1 2>&1 &
  done;
wait %{1..16};
echo all runs complete
