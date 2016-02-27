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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;

public class LogAnalyzer {
	 public static void main(String[] args) throws Exception {
		 Configuration conf = new Configuration();
		 if (args.length != 2) {
			 System.err.println("Usage: loganalyzer <in> <out>");
			 System.exit(2);
		 }
		 Job job = new Job(conf, "analyze log");
		 job.setJarByClass(LogAnalyzer.class);
		 job.setMapperClass(MyMapper.class);
		 job.setReducerClass(Reduce.class);
		 job.setOutputKeyClass(Text.class);
		 job.setOutputValueClass(IntWritable.class);
		 /*get the FileSystem, you will need to initialize it properly */
		FileSystem fs= FileSystem.get(conf); 
		/*get the FileStatus list from given dir */
		FileStatus[] status_list = fs.listStatus(new Path(args[0]));
		if(status_list != null){
		    for(FileStatus status : status_list){
		        /*add each file to the list of inputs for the map-reduce job*/
		        FileInputFormat.addInputPath(job, status.getPath());
		    }
		}
		 FileOutputFormat.setOutputPath(job, new Path(args[1]));
		 System.exit(job.waitForCompletion(true) ? 0 : 1);
	 }
}
