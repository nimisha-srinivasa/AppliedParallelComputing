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

public class MyMapper extends Mapper<Object, Text, Text, IntWritable> {
	private Map<Text, IntWritable> countMap = new TreeMap<Text, IntWritable>();

	 private Pattern p = Pattern.compile("(\\d+.\\d+.\\d+.\\d+).*(?:GET)\\s([^\\s]+)");
	 @Override
	 public void map(Object key, Text value, Context context)
	 throws IOException, InterruptedException {
		 String[] entries = value.toString().split("\r?\n");
		 Text url=new Text();
		 for (int i=0, len=entries.length; i<len; i+=1) {
			 Matcher matcher = p.matcher(entries[i]);
			 if (matcher.find()) {
			 	url.set(matcher.group(1));
			 	if(countMap.containsKey(url)){
			 		countMap.put(url, new IntWritable(countMap.get(url).get()+1));
			 	}else{
			 		countMap.put(new Text(url),new IntWritable(1));
			 	}
				
			 }
		 }
	 }

	 @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        Map<Text, IntWritable> sortedMap = sortByValues(countMap);
        int counter = 0;
        for (Text key: sortedMap.keySet()) {
            if (counter ++ == 5) {
                break;
            }
            context.write(new Text(key), new IntWritable(sortedMap.get(key).get()));
        }
    }

    @SuppressWarnings("unchecked")
    private static <K extends Comparable, V extends Comparable> Map<K, V> sortByValues(Map<K, V> map) {
        List<Map.Entry<K, V>> entries = new LinkedList<Map.Entry<K, V>>(map.entrySet());

        Collections.sort(entries, new Comparator<Map.Entry<K, V>>() {

            @Override
            public int compare(Map.Entry<K, V> o1, Map.Entry<K, V> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });
        Map<K, V> sortedMap = new LinkedHashMap<K, V>();
        for (Map.Entry<K, V> entry : entries) {
            sortedMap.put(entry.getKey(), entry.getValue());
        }
        return sortedMap;
    }
}