#!/usr/bin/perl

# Helper scripts that takes the pairs created with explode_collapse.pl and the metric scores
# for each pair, computes MBR and by highest score. Restores original sample number (not order, due to sorting).
# Grepping for "^BEST:" will result in a file with as many hypotheses as orignal input sentences in the right order.

my $N = $ARGV[0];
my $R = $ARGV[1];
open(IDS,    "cat < $ARGV[2] |");
open(SCORES, "cat < $ARGV[3] |");

$| = 1; 

sub score {
  my $samples = shift;
  my $scores  = shift;

  my %cnd;
  foreach(@$samples) {
    $cnd{$_} = scalar keys %cnd if not exists($cnd{$_});
  }

  my @scored;
  foreach my $t (@$samples) {
    my $sum = 0;
    my $tid = $cnd{$t};
    my $c = 0;
    foreach my $r (@$samples) {
      my $rid = $cnd{$r};
      if(exists($scores->{$tid}->{$rid}) and $c < $R) {
        $sum += $scores->{$tid}->{$rid};
        $c++;
      }
    }
    push(@scored, [$sum / $c, $t]);
  }
  my ($best, @rest) = sort { $b->[0] <=> $a->[0] } @scored;
  printf("BEST\t%.4f\t%s\n", @$best);
  printf("REST\t%.4f\t%s\n", @$_) foreach(@rest);
}

my $samples = [];
my $scores = {};
my $id1 = 0;
while(<STDIN>) {
  chomp;
  push(@$samples, $_);
  if(@$samples == $N) {
    my ($ids, $score);
    while(($ids = <IDS>) and ($score = <SCORES>)) {
      chomp($ids, $score);
      my($id2, $r, $t) = split(/\t/, $ids);
      if($id1 == $id2) {
        $scores->{$t}->{$r} = $score;
      } else {
        score($samples, $scores);
        $samples = [];
        $scores = {};
        $scores->{$t}->{$r} = $score;
        last;
      }
    }
    $id1++;
  }
}
score($samples, $scores);

close(SCORES)
