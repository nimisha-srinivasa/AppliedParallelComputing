import java.io.IOException;
import java.util.Map;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

public class Reduce extends Reducer<Text, IntWritable, Text, IntWritable> {
	 private IntWritable total = new IntWritable();
	 @Override
	 public void reduce(Text key, Iterable<IntWritable> values, Context
	context) throws IOException, InterruptedException {
		 int sum = 0;
		 for (IntWritable value : values) {
		 	sum += value.get();
		 }
		 total.set(sum);
		 context.write(key, total);
	 }
}