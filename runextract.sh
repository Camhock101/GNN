  for i in {1..16};   
    do     
    python3 extract.py $i &   
    echo running
    done;   
  wait %{1..16};
  echo complete  
