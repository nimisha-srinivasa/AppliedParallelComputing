import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.io.IOException;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.conf.Configuration;

public class MyMapper extends Mapper<Object, Text, Text, IntWritable> {
	 private final static IntWritable one = new IntWritable(1);
	 private Text url = new Text();
	 /*to extract the user IP and their count*/
	 private Pattern p = Pattern.compile("(\\d+.\\d+.\\d+.\\d+).*(?:GET)\\s([^\\s]+)");
	 @Override
	 public void map(Object key, Text value, Context context)
	 throws IOException, InterruptedException {
		 String[] entries = value.toString().split("\r?\n");
		 for (int i=0, len=entries.length; i<len; i+=1) {
			 Matcher matcher = p.matcher(entries[i]);
			 if (matcher.find()) {
				url.set(matcher.group(1));
				context.write(url, one);
			 }
		 }
	 }
}