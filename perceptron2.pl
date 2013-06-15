# perceptron.pl V2
# clean program in V1, no novelties


########## MAIN PROGRAM STARTS HERE ############

  	#### TRAINING #######

&readTrain;		# reads training data
&numerizeTable;		# prepares table for numerical work
&fillEmptyAges;		# Fills blank ages with the age average

# Now implements the perceptron algorithm (pocket PA, since saves best solution)
$maxIts=2*$#data;	# number of iterations; usually 5 x num of train cases
$best=0;			# initialize best case
for (0..$nf) {$W[$_]=rand()-0.5;}  # init perceptron weights random
for $nits (1..$maxIts) { 
	&runPercepAll;		# classifies training set with current W
	$score=&calcScore;	# percentage of correct classifications
	if ($score>$best) {	# if found a best weight vector W...
		print "at $nits the best score is: ", &calcScore,"\n";
		@bestW=@W;
		$best=$score;
	}
	$m=&getOneMisclass;		# find at random one misclassified case
	$c=$varTable[$m][0];		# its label coef is +1 or -1
	# now apply the weight update of the perceptron algorithm
	for (1..$nf) {$W[$_] += $c*$varTable[$m][$_];}
}

print  "\n\nRESULTS:\n\n";
print join "\n",@bestW,"\n"; 
print "score: ", $best,"\n\n\n";
&trainSaveCSV;		# save CSV table of training set (for using external libsvm) 

		######  TESTING  ######

&readTest;		# Now reads test data
#print  "\n\nsize of data is ",$#data,"\n";
# restores size of lines, adding 1 fake label, in order to use same subs of training
for (0..$#data) {$data[$_] = "1,".$data[$_] ;} 

$SumAges = 0; $nSumAges = 0; @varTable=(); # resets vars and table
&numerizeTable;		# prepares table for numerical work
&fillEmptyAges;		# Fills blank ages with the age average
&testSaveCSV;		# save CSV table of test set (for external libsvm) 
&testSaveCSVnoLabel;  # save CSV table of test set (for R random forests) 

# now classifies the test set using the optimal (or best) W found in training 
@predLabs=();	# array to save predictions
&runPercepAll;	# run perceptron to classify test data		
@predLabs = map {$_<0 ? 0 : 1;} @predLabs;  # converts -1's to 0's: kaggle format
# print @predLabs,"\n";	
print "\n $#predLabs\n";	# to confirm number of test cases
open(OFP,"> test.res") or die("File not found\n");
for (@predLabs) {print OFP "$_\n";}	# print file with prediction for test data


####### PERL SUBROUTINES ###########

# converts data to sensible numerical values, using normalization
# or zeroing variables that are discarded (ticket number and cabin)
sub transform() {
	my $ref =shift;
	@row=@$ref;
	if ($row[0] eq "0") {	# label is -1 or 1 for classif. with perceptron
		$row[0]=-1;	 
	} else {
		$row[0]=1;	 
	}	
	$row[2]=length($row[2])/20.0;	# normalized length of name
	if ($row[3] eq "male") {
		$row[3]=1;	 
	} else {
		$row[3]=0;	 
	}
	if (ord $row[4]) {
		$row[4] = $row[4]/25.0; # normalized age
		$SumAges += $row[4];	   # to get age average and fill blanks
		$nSumAges++;
	} else {
		$row[4] = 0;	# normalized age
	}
	if ($row[10] =~ /C/) { # embarked
		$row[10] = 0;	 
	} elsif ($row[10] =~ /S/) {
		$row[10] = 1;	 
	} else  {
		$row[10] = 2;	 
	}
	$row[7]=0;		# ticket number nulled 
	$row[8]=$row[8]/20.0;  # normalized ticket cost
	$row[9]  =~ "" ?  $row[9]=0:	$row[9]=1	;	# is worst
	$row[9]=0;		# cabin zeroed; price already reflects cabin
	return \@row;
}

# sign(x) needed by perceptron classifier
sub sign(){
	my $x = shift;
	return -1 if $x<0;
	1;
}

# reads train data
sub readTrain() {
	open(IFP,"< train.csv") or die("File not found\n");
	@data=<IFP>;
	@header = split ",", shift @data;
	print $#data+1," persons for training\nFields in line: ";
	$nf=$#header;	# number of features (is size of data line minus 1)
	print $nf, " classification vars\n";
	print "0		1	2	3	4	5	6	7	8	9	10\n";
	print join " ", @header, "\n";
}

# reads test data; recal test data has no 1st column with labels
sub readTest() {
	open(TFP,"< test.csv") or die("File not found\n");
	@data=<TFP>;
	@header = split ",", shift @data;
	print $#data+1," persons for testing\nFields in line: ";
	$nf=$#header;
	print $nf+1, " classification vars\n";
	print "1	2	3	4	5	6	7	8	9	10\n";
	print join " ", @header, "\n";
}

# massages the original data: fills missing cases, 
# quantifies data (e.g. departure point # is converted to 1,2,3,...
# this work is done with the sub transform
sub numerizeTable(){
		foreach (@data) {
		@line = split ",";
		# name fills two fields because have a ',' in between. Here is re-melted
		$line[2]=$line[2] . $line[3];	$line[3]="nnn";
		@line = grep {! /nnn/} @line;	# removes extra field name
		@line = @{&transform(\@line)};
		push @varTable, [@line];	# puts in table refs for rows
	}
}

# classifies data with current weight vector, W
sub runPercepAll(){	# runs current W to classify all training cases
	for (my $i=0; $i <= $#varTable; $i++) {
		@line = @{$varTable[$i]};
		$p=0; 
		for (1..$nf) {$p+=$W[$_]*$line[$_];}
		$p=&sign($p);
		$predLabs[$i]=$p*$line[0];
	}	
}

# gets one misclassified
sub getOneMisclass(){		# chooses one misclassified case randomly
	my $i=int(rand($#varTable));	# start at random index
	# modulus % is used to wrap around end of vector
	while ($predLabs[$i]>0) 	# $predLabs[$i]<0 means $i is misclassified 
		{$i++; $i=$i % ($#varTable+1) ;}
	return $i;
}

# returns percentage of correct classifications
sub calcScore(){		
	$s=0;
	for (0..$#predLabs) { $s += $predLabs[$_];}
	$n=$#predLabs+1;
	return ($s+$n)/2/$n;
}

# save CSV table of train set (for using external libsvm) 
sub trainSaveCSV() {
	open(CSV,"> trainClean.csv") or die("File not found\n");
	for (my $i=0; $i <= $#varTable; $i++) {
		@line = @{$varTable[$i]};
		$line = join ",", @line;
		print CSV $line,"\n";
	}
}

# save CSV table of test set (for using external libsvm) 
sub testSaveCSV() {
	open(CSVT,"> testClean.csv") or die("File not found\n");
	for (my $i=0; $i <= $#varTable; $i++) {
		@line = @{$varTable[$i]};
		#print @line,"\n";
		$line = join ",", @line;
		print CSVT $line,"\n";
	}
}

# save CSV table of test set (for R random forests) 
sub testSaveCSVnoLabel() {
	open(CSVTNL,"> testCleanNoLabels.csv") or die("File not found\n");
	for (my $i=0; $i <= $#varTable; $i++) {
		@line = @{$varTable[$i]};
		shift @line;		# removes 1st entry, the label
		$line = join ",", @line;
		print CSVTNL $line,"\n";
	}
}

# fill empty ages with average
sub fillEmptyAges(){
	print "Normalized age Info: ",$SumAges," ",$nSumAges,"\n\n";
	$avAge = $SumAges/$nSumAges . " ";
	$avAge = substr $avAge, 0, 6;
	for (my $i=0; $i <= $#varTable; $i++) {
		@line = @{$varTable[$i]};
		$line[4]=$avAge if $line[4] eq "0";
		$varTable[$i] =  [@line];
	}
}


