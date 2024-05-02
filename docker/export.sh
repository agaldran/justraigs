
docker build --tag justraigs_data_centric .

docker save justraigs_data_centric | gzip -c > justraigs_data_centric.tar.gz