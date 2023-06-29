#!/usr/bin/perl

# Helper script that takes the sample file with N samples and M references (first M among N samples)
# and creates deduped(!) N' x M' pairs (N' is N after deduplication, same for M') for scoring. 
# Creating the pairs is "exploding", deduping is "collapsing", hence the name. 
# Includes ids so that the original order from before deduplication can be restored.

my $N = $ARGV[0];
my $R = $ARGV[1];
$R = $N if not defined($R);

sub explodeCollapse {
  my $id = shift;
  my @samples = @_;

  my %cnd;
  foreach(@samples) {
    $cnd{$_} = scalar keys %cnd if not exists($cnd{$_});
  }

  my @uniq = sort { $cnd{$a} <=> $cnd{$b} } keys %cnd;
  foreach my $t (@uniq) {
    my $c = 0;
    foreach my $r (@uniq) {
      last if($c >= $R);
      # this outputs the pseudo-reference first!
      printf("%d\t%d\t%d\t%s\t%s\n", $id, $cnd{$r}, $cnd{$t}, $r, $t);
      $c++;
    }
  }
}

my @samples;
my $id = 0;
while(<STDIN>) {
  chomp;
  push(@samples, $_);
  if(@samples == $N) {
    explodeCollapse($id, @samples);
    @samples = ();
    $id++;
  }
}
