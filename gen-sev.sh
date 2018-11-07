
function gen()
{
cat <<HEREDOC
<!DOCTYPE html>
<html>
<head>
	<title></title>
	<style type="text/css">
		table { border-spacing: 16px; }
		td {
			min-width: 200px;
			max-height: 200px;
			border: 1px solid grey;
			padding: 0;
			margin: 0;
		}
		img.vi {
			object-fit: scale-down;
			width: 100%;
			visibility: visible;
		}
		td img.hi {
			display: none;
			visibility: hidden;
		}
		td:hover img.hi {
			visibility: visible;
			display: inline;
			max-height: 800px;
			max-width: 800px;
			position: fixed;
			bottom: 0;
			right: 0;
		}
		span {
			font-size: 8pt;
		}
		span.orig {
			font-style: italic;
		}
	</style>
</head>
<body>
<table>
HEREDOC

	imid=$1
	while read -r origjpg; do
		b=$(basename $origjpg)
		d=$(dirname $origjpg)
		tmp=${d#*orig/}
		i=${tmp%/vis*}

		echo "<tr id=\"$i-$b\">"
		printf "<td>"
		printf "<span class='orig'>orig</span><br>"
		printf "<img class=\"vi\" src=\"$origjpg\">"
		printf "<img class=\"hi\" src=\"$origjpg\">\n"
		printf "</td>\n"
		echo "</tr>"
		echo "<tr>"
		# add td for each fine-tuned model
		while read -r ftjpg; do
			printf "<td>"
			if [[ -e "$ftjpg" ]]; then
				printf "<span>${ftjpg}</span><br>"
				printf "<img class='vi' src='$ftjpg'>"
				printf "<img class='hi' src='$ftjpg'>\n"
			fi
			printf "</td>"
		done < <(find examples/frz -path "*/$i/vis/*" -name "$b")
		echo "</tr>"
	done < <(find examples/orig -path \*${imid}/vis/\* -name \*.jpg)
cat <<-HEREDOC
</table></body></html>
HEREDOC
}

for imid in `seq 8`; do
	gen $imid > $imid.html
done

