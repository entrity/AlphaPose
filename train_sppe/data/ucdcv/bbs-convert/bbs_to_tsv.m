% Make 3 outputs files

% Make a TSV with the following columns
% - vid id
% - relative image path
% - bounding box (4 cols)
% - key points (17 cols for COCO style)

function bbs_to_tsv ()
	fout = fopen( 'bbs.tsv', 'w' );
	fprintf( fout, 'vidid\tframe\tx1\ty1\tx2\ty2\n' );
	fout_annotator_input = fopen( 'annotator-input.tsv', 'w' );
	fout_sftp_batch      = fopen( 'sftp-batch.bat', 'w' );
	fprintf( fout_sftp_batch, 'cd /data/krishna-data/first_third_project/share\n' );
	
	count = 0;

	for vidid = 1:8
		% Load bounding boxes: 'positions'
		load( sprintf('../images/%d/Mac.mat', vidid) );
		% Dump bounding boxes
		for framei = 1:length(positions)
			bbs = positions{framei};
			if length(bbs) > 0
				for bbi = 1:length(bbs)
					bb_to_file( fout, sprintf('%d\t%d', vidid, framei), bbs{bbi}, false );
					if mod(count, 43) == 0
						bb_to_file( fout_annotator_input, sprintf('inputs/%d/%05d.jpg', vidid, framei), bbs{bbi}, true );
						fprintf( fout_sftp_batch, sprintf('get %d/frame/Mac*/%05d.jpg ../images/%d/\n', vidid, framei, vidid) );
					end
				end
			end
			count = count + 1;
		end
	end

	fclose(fout_sftp_batch);
	fclose(fout_annotator_input);
	fclose(fout);
end

function bb_to_file(fid, prefix, bb, do_round)
	fprintf( fid, prefix );
	for ci = 1:4
		if do_round
			fprintf( fid, sprintf( '\t%d', round(bb(ci)) ) );
		else
			fprintf( fid, sprintf( '\t%f', bb(ci) ) );
		end
	end
	fprintf( fid, '\n' );
end