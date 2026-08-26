set -x
file=`hadoop fs -ls viewfs:///home/reco_5/mpi/products/simple-mio-kai |tail -n 1 |awk '{print $NF}'`

dst=latest_bin.tgz
rm -rf $dst
hadoop fs -get $file $dst


[[ x"$1"=x"-f" ]] && tar xzvf $dst && rm -rf $dst
