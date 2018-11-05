
sshfs v4:/data/krishna-data/first_third_project/share $HOME/mnt
SRC=$HOME/mnt/

mkdir -p images

# Copy mat files containing bounding boxes
for vidid in `seq 1 8`; do
	mkdir -p images/$vidid
	cp ${SRC}/${vidid}/frame/Mac*.mat images/$vidid/Mac.mat
done

# Copy jpg files
while read -r line; do
	vidid=${line::1}
	fname=$(basename $line)
	cp ${SRC}/${vidid}/frame/Mac*/${fname} images/${vidid}/${fname}
done < inventory.tsv

fusermount -u $HOME/mnt
