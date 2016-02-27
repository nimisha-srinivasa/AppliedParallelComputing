cd ~/log/trafficdata
for i in {4..18}
do
	output_dir="apache$i.splunk.com"
	cp -r apache1.splunk.com $output_dir
done

for i in {19..33}
do
	output_dir="apache$i.splunk.com"
	cp -r apache2.splunk.com $output_dir
done

for i in {34..48}
do
	output_dir="apache$i.splunk.com"
	cp -r apache3.splunk.com $output_dir
done


