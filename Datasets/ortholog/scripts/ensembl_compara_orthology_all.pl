#!/usr/bin/env perl

use strict;
use Bio::EnsEMBL::Registry;

my $output_path = "../ensembl/orthology_new";
my $reg = "Bio::EnsEMBL::Registry";

print("Connecting database...\n");
$reg->set_reconnect_when_lost(1);
$reg->load_registry_from_db(
    -host => "mysqldb1.cbi.pku.edu.cn",
    -user => "Anonymous"
);
# my $gene_member_adaptor = $reg->get_adaptor("Multi", "Compara", "GeneMember");
my $homology_adaptor = $reg->get_adaptor("Multi", "Compara", "Homology");

print("Fetching homologies...\n");
# my $gene_member = $gene_member_adaptor->fetch_by_stable_id("ENSG00000206172");
# my $homologies = $homology_adaptor->fetch_all_by_Member($gene_member);
my $homologies = $homology_adaptor->fetch_all();

foreach my $homology (@$homologies) {
    my $homology_id = $homology->stable_id();
    my $homology_description = $homology->description();
    if (not $homology_description eq "ortholog_one2one" and
        not $homology_description eq "ortholog_one2many" and
        not $homology_description eq "ortholog_many2many") {
        next;
    }
    my @members = @{$homology->get_all_Members()};
    scalar @members == 2 or die;
    my @genes = (
        $members[0]->gene_member()->stable_id(),
        $members[1]->gene_member()->stable_id()
    );
    my @gene_names = (
        $members[0]->gene_member()->display_label(),
        $members[1]->gene_member()->display_label()
    );
    my @taxons = (
        $members[0]->gene_member()->taxon_id(),
        $members[1]->gene_member()->taxon_id()
    );
    my @files = (
        "$taxons[0]_$taxons[1]",
        "$taxons[1]_$taxons[0]"
    );
    open(FILE, ">>$output_path/$files[0].csv") or die;
    FILE->print("$genes[0],$gene_names[0],$genes[1],$gene_names[1],$homology_description\n");
    FILE->close() or die;
    open(FILE, ">>$output_path/$files[1].csv") or die;
    FILE->print("$genes[1],$gene_names[1],$genes[0],$gene_names[0],$homology_description\n");
    FILE->close() or die;
}
print("Done!\n");
