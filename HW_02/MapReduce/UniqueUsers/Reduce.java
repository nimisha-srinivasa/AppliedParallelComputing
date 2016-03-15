import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;


public class Reduce extends Reducer<Text, Text, Text, IntWritable> {
	 private IntWritable total = new IntWritable();
	 private Set<Text> unique_user_set = new HashSet<Text>();
	 @Override
	 public void reduce(Text key, Iterable<Text> values, Context
	context) throws IOException, InterruptedException {
		 int sum = 0;
		 for (Text value : values) {
		 	unique_user_set.add(value);
		 }
		 total.set(unique_user_set.size());
		 context.write(key, total);
	 }
}