import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.io.IOException;
import java.util.*;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.conf.Configuration;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

public class MyMapper extends Mapper<Object, Text, Text, Text> {

	 Pattern p = Pattern.compile("(\\d+.\\d+.\\d+.\\d+).*\\[(.*?):.*\\].*(?:GET)\\s([^\\s]+)");
	 @Override
	 public void map(Object key, Text value, Context context)
	 throws IOException, InterruptedException {
		 String[] entries = value.toString().split("\r?\n");
		 Text user_ip=new Text();
         Text date = new Text();
		 for (int i=0, len=entries.length; i<len; i+=1) {
			 Matcher matcher = p.matcher(entries[i]);
			 if (matcher.find()) {
			 	user_ip.set(matcher.group(1));
                date.set(matcher.group(2));
			 	context.write(date, user_ip);
				
			 }
		 }
	 }
}