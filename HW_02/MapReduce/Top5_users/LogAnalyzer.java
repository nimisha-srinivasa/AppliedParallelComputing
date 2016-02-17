import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class LogAnalyzer {
	 public static void main(String[] args) throws Exception {
		 Configuration conf = new Configuration();
		 if (args.length != 2) {
			 System.err.println("Usage: loganalyzer <in> <out>");
			 System.exit(2);
		 }
		 Job job = new Job(conf, "analyze log");
		 job.setJarByClass(LogAnalyzer.class);
		 job.setMapperClass(Map.class);
		 job.setReducerClass(Reduce.class);
		 job.setOutputKeyClass(NullWritable.class);
		 job.setOutputValueClass(Text.class);
		 FileInputFormat.addInputPath(job, new Path(args[0]));
		 FileOutputFormat.setOutputPath(job, new Path(args[1]));
		 System.exit(job.waitForCompletion(true) ? 0 : 1);
	 }
}
