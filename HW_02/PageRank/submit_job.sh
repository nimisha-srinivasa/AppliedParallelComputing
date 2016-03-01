#!/bin/bash
################################################################################
#  A simple Python based example for Spark
#  Designed to run on SDSC's Comet resource.
#  T. Yang. Modified from Mahidhar Tatineni's scripe for scala
################################################################################
#SBATCH --job-name="sparkpython-demo"
#SBATCH --output="sparkwc.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --export=ALL
#SBATCH -t 00:30:00

### Environment setup for Hadoop and Spark
module load spark
export PATH=/opt/hadoop/2.6.0/sbin:$PATH
export HADOOP_CONF_DIR=$HOME/mycluster.conf
export WORKDIR=`pwd`

myhadoop-configure.sh

### Start HDFS.  Starting YARN isn't necessary since Spark will be running in
### standalone mode on our cluster.
start-dfs.sh

### Load in the necessary Spark environment variables
source $HADOOP_CONF_DIR/spark/spark-env.sh

### Start the Spark masters and workers.  Do NOT use the start-all.sh provided
### by Spark, as they do not correctly honor $SPARK_CONF_DIR
myspark start

###copy data into hdfs
hdfs dfs -mkdir -p /user/$USER
#hdfs dfs -put $WORKDIR/facebook_combined.txt /user/$USER/
hdfs dfs -put $WORKDIR/data/simple2 /user/$USER/input
#hdfs dfs -put $WORKDIR/data/wiki /user/$USER/wiki

#run-example org.apache.spark.examples.graphx.LiveJournalPageRank facebook_combined.txt --numEPart=8

#spark-submit run_pagerank.py s data/simple1 20 > temp
#this works: pyspark run_pagerank.py s simple1 20 > temp
#it works:  spark-submit run_pagerank.py s simple1 20 > temp
#spark-submit run_pagerank.py s wiki 20 > wiki.out
rm output
spark-submit run_pagerank.py s  /user/$USER/input 1 > output
#

#copy output
rm -rf pagerank-out >/dev/null || true
mkdir -p pagerank-out
#hadoop dfs -copyToLocal output/part* $WORKDIR
hadoop dfs -copyToLocal output pagerank-out


### Shut down Spark and HDFS
myspark stop
stop-dfs.sh

### Clean up
myhadoop-cleanup.sh
