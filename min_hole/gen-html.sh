indir=${1:-spread12}

function header()
{
cat <<HEREDOC

	<!DOCTYPE html>
	<html>
		<head>
			<title></title>
			<style type="text/css">
				table { border-spacing: 16px; }
				img {
					height: 200px;
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
}
function footer()
{
cat <<HEREDOC
		</table>
	</body>
	</html>

HEREDOC
}

function table()
{
	find $indir -type f -name \*.jpg | sed 's/.*\///g' | grep -oP '\d+' | sort -h | uniq | while read f; do
		echo "<tr>"
		echo -e "\t<td><span class=tiny>MAX ($indir) (${f})</span><br><img src=\"${indir}/${f}.jpg\"></td>"
		echo -e "\t<td><span class=tiny>AVG ($indir) (${f})</span><br><img src=\"${indir}/avg-${f}.jpg\"></td>"
		echo -e "\t<td><span class=tiny>MIN ($indir) (${f})</span><br><img src=\"${indir}/min-${f}.jpg\"></td>"
		echo "<tr>"
	done
}



header
table
footer