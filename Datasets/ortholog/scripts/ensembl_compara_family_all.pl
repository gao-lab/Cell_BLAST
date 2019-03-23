#!/usr/bin/env perl

use strict;
use Bio::EnsEMBL::Registry;

my $output_path = "../Ensembl/family";
my $reg = "Bio::EnsEMBL::Registry";

print("Connecting database...\n");
$reg->set_reconnect_when_lost(1);
$reg->load_registry_from_db(
    # -host => "asiadb.ensembl.org",
    # -host => "ensembldb.ensembl.org",
    -host => "mysqldb1.cbi.pku.edu.cn",
    -user => "Anonymous"
);
# my $gene_adaptor   = $reg->get_adaptor("Human", "Core", "Gene");
my $family_adaptor = $reg->get_adaptor("Multi", "Compara", "Family");

print("Fetching families...\n");
# my $gene = $gene_adaptor->fetch_by_stable_id("ENSG00000206172");
# my $gene = $gene_adaptor->fetch_by_stable_id("ENSG00000012048");
# my $families = $family_adaptor->fetch_all_by_Gene($gene);
my $families = $family_adaptor->fetch_all();

my %species_files;
open(PROGRESS_FILE, "+>>$output_path/.progress") or die;
seek(PROGRESS_FILE, 0, 0);
chomp(my @progress = <PROGRESS_FILE>);
my %progress = map {$_ => undef} @progress;

foreach my $family (@$families) {
    my $family_id = $family->stable_id();
    if (exists($progress{$family_id})) {
        print("Ignoring family: $family_id\n");
        next;
    }
    print("Processing family: $family_id\n");
    my $family_members = $family->get_all_GeneMembers();
    foreach my $family_member (@$family_members) {
        my $stable_id = $family_member->stable_id();
        my $taxon_id = $family_member->taxon_id();
        if (not exists($species_files{$taxon_id})) {
            open($species_files{$taxon_id},
                ">>$output_path/$taxon_id.csv") or die;
        }
        $species_files{$taxon_id}->print("$stable_id,$family_id\n");
    }
    PROGRESS_FILE->print("$family_id\n");
}

foreach my $species_file (values(%species_files)) {
    $species_file->close() or die;
}
print("Done!\n");
