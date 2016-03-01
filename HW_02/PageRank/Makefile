# Build file for spark pagerank

# This file requires GNU make and depends on paths on instruction machines.

####
SOURCES = $(wildcard *.java)
TARGET = wc.jar
JAR = jar
HHOME= /opt/hadoop/1.2.1
#DONOT USE name HOME.
HADOOP = ${HHOME}/bin/hadoop
HADOOPLIB = ${HHOME}
CLASSPATH= ${HADOOPLIB}/hadoop-core-1.2.1.jar:${HADOOPLIB}/hadoop-client-1.2.1.jar
JAVAC = javac -cp $(CLASSPATH)


all: $(TARGET)

pagerank:
        rm -rf pagerank-out
        sbatch submit-spark-comet.sh

classes:
        mkdir -p classes
        
clean:
        rm -rf pagerank-out*
