% Make a TSV with the following columns
% - vid id
% - relative image path
% - bounding box (4 cols)
% - key points (17 cols for COCO style)

function generate_dataset ()
	dbstop on error

	% Load bounding boxes
	bbs = {};
	for vidid = 1:8
		load( sprintf('images/%d/Mac.mat', vidid) );
		bbs{vidid} = positions;
	end

	% Load inventory.tsv
	fid = fopen( 'inventory.tsv', 'r' );
	inventory = textscan( fid, '%s' );
	inventory = inventory{1};
	fclose(fid);

	% Load pose annotations
	kypts = load_keypoints();

	% Output TSV
	fid = fopen( 'meta.tsv', 'w' );
	for row = 1:length(inventory) % iterate for images mentioned in inventory
		fpath = inventory{row};
		[dpath, name, ext] = fileparts(fpath);
		vidid = str2num( dpath(1) );
		frame_offset = str2num( name );

		bbs_in_frame   = bbs{vidid}{frame_offset};
		kypts_in_frame = kypts{vidid}{frame_offset};
		% Find keypoints annotations that match the given bbs in this frame
		bbs_and_kypts = match_keypoints_to_bbs( bbs_in_frame, kypts_in_frame )
		% Iterate humans (bbs and kypts)
		for I = 1:length(bbs_and_kypts{1}) % iterate bounding boxes (for humans) in given image
			bb = bbs_and_kypts{1}{I}
			kp = bbs_and_kypts{2}{I}
			% Print vidid and image path
			fprintf( fid, '%d\t%s', vidid, sprintf('images/%d/%s.%s', vidid, name, ext) );
			% Print bounding box
			for coord_i = 1:length(bb)
				fprintf( fid, '\t%d', bb(coord_i) );
			end
			% Print pose keypoints
			for pt_i = 1:size(kp,2)
				fprintf( fid, '\t%d\t%d', kp(1,pt_i), kp(2,pt_i) );
			end
			fprintf( fid, '\n' );
		end
	end
end

% Return cell {bbs} [kypts]
function match = match_keypoints_to_bbs ( bbs_in_frame, keypoints_in_frame )
	% Compute distances between all keypoint sets and all bounding boxes
	distances = zeros( length(bbs_in_frame), size(keypoints_in_frame,3) );
	assert( size(distances,1) == size(distances,2) );
	for b = 1:length(bbs_in_frame)
		for k = 1:size(keypoints_in_frame, 3)
			bb = bbs_in_frame{b};
			kps= keypoints_in_frame(:,:,k);
			distances( b, k ) = distance_btwn_pose_and_bb( bb, kps );
		end
	end
	% Find best for each
	[Y,I] = min(distances);
	assert(length(I) == length(unique(I)))
	% Return
	match = {};
	match{1} = bbs_in_frame;
	match{2} = keypoints_in_frame(:,:,I)
end

% kypts is cell array: {vidid}{frame} => [ xy, 21points, personinframe ]
function kypts = load_keypoints ()
	kypts = {};
	for l = 1:8
		kypts{l} = {};
	end
	fid = fopen( 'keypoints.tsv', 'r' );
	data = textscan( fid, '%d\t%s\t%s' ); % {vidid} {fname} {kypts}
	fclose(fid);
	for l = 1:length(data{1})
		vidid = data{1}(l);
		fname = data{2}{l};
		[pth, frame, ext] = fileparts(fname);
		frame = str2num(frame);
		% Handle the keypoints
		csv = data{3}{l};
		vector = cellfun( @(x) str2num(x), split(csv,',') );
		n = prod(size(vector));
		n_bodies = n / 21 / 2;
		kypts{vidid}{frame} = reshape( vector, 2, 21, n_bodies );
	end
end


function l1 = distance_btwn_pose_and_bb ( bb, kypts )
	l1 = 0;
	for k = 1:size(kypts,2)
		kypt = kypts(:,k);
		x = kypt(1);
		y = kypt(2);
		if x < bb(1)
			l1 = l1 + abs(x - bb(1));
		elseif x > bb(3)
			l1 = l1 + abs(x - bb(3));
		end
		if y < bb(2)
			l1 = l1 + abs(y - bb(2));
		elseif y > bb(4)
			l1 = l1 + abs(y - bb(4));
		end
	end
end
