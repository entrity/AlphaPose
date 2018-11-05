
sshfs v4:/data/krishna-data/first_third_project/share $HOME/mnt
SRC=$HOME/mnt/

mkdir -p images
while read -r line; do
	vidid=${line::1}
	fname=$(basename $line)
	mkdir -p $vidid
	ln -s ${SRC}/${vidid}/frame/Mac*/${fname} ${vidid}/${fname}
done < inventory.tsv

fusermount -u $HOME/mnt
