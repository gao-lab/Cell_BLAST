#!/usr/bin/env perl

use strict;
use Bio::EnsEMBL::Registry;

my $output_path = "../Ensembl/family";
my $reg = "Bio::EnsEMBL::Registry";

print("Connecting database...\n");
$reg->set_reconnect_when_lost(1);
$reg->load_registry_from_db(
    -host => "mysqldb1.cbi.pku.edu.cn",
    -user => "Anonymous"
);
my $family_adaptor = $reg->get_adaptor("Multi", "Compara", "Family");

print("Fetching families...\n");
my $families = $family_adaptor->fetch_all();

print("Saving homologies...\n");
open(FILE, ">$output_path/$ARGV[0].csv") or die;
foreach my $family (@$families) {
    my $family_id = $family->stable_id();
    print("Processing family: $family_id\n");
    my $family_members = $family->get_all_GeneMembers();
    foreach my $family_member (@$family_members) {
        my $taxon_id = $family_member->taxon_id();
        if ($taxon_id == $ARGV[0]) {
            my $stable_id = $family_member->stable_id();
            my $gene_name = $family_member->get_Gene()->external_name();
            FILE->print("$stable_id,$gene_name,$family_id\n");
        }
    }
}
FILE->close() or die;
print("Done!\n");
