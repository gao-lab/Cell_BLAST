#!/usr/bin/env perl

use strict;
use Bio::EnsEMBL::Registry;

my $output_path = "../Ensembl/orthology";
my $reg = "Bio::EnsEMBL::Registry";

print("Connecting database...\n");
$reg->set_reconnect_when_lost(1);
$reg->load_registry_from_db(
    -host => "mysqldb1.cbi.pku.edu.cn",
    -user => "Anonymous"
);
my $genome_db_adaptor = $reg->get_adaptor("Multi", "Compara", "GenomeDB");
my $method_link_species_set_adaptor =
    $reg->get_adaptor("Multi", "Compara", "MethodLinkSpeciesSet");
my $homology_adaptor = $reg->get_adaptor("Multi", "Compara", "Homology");

print("Fetching homologies...\n");
my $genome_dbs = [
    $genome_db_adaptor->fetch_by_name_assembly($ARGV[0]),
    $genome_db_adaptor->fetch_by_name_assembly($ARGV[1])
];
my $method_link_species_set =
    $method_link_species_set_adaptor->fetch_by_method_link_type_GenomeDBs(
        "ENSEMBL_ORTHOLOGUES", $genome_dbs
    );
my $homologies = $homology_adaptor->fetch_all_by_MethodLinkSpeciesSet($method_link_species_set);

print("Saving homologies...\n");
my @taxons = (
    @{$genome_dbs}[0]->taxon_id(),
    @{$genome_dbs}[1]->taxon_id()
);
open(FILE, ">$output_path/$taxons[0]_$taxons[1].csv") or die;
foreach my $homology (@$homologies) {
    my $homology_id = $homology->stable_id();
    my $homology_description = $homology->description();
    if (not $homology_description eq "ortholog_one2one" and
        not $homology_description eq "ortholog_one2many" and
        not $homology_description eq "ortholog_many2many") {
        next;
    }
    my @members = @{$homology->get_all_Members()};
    my @genes = (
        $members[0]->gene_member()->stable_id(),
        $members[1]->gene_member()->stable_id()
    );
    my @gene_names = (
        $members[0]->gene_member()->get_Gene()->external_name(),
        $members[1]->gene_member()->get_Gene()->external_name()
    );
    my @gene_taxons = (
        $members[0]->gene_member()->taxon_id(),
        $members[1]->gene_member()->taxon_id()
    );  # Need to align with @taxons
    if ($gene_taxons[0] == $taxons[0] and $gene_taxons[1] == $taxons[1]) {
        FILE->print("$genes[0],$gene_names[0],$genes[1],$gene_names[1],$homology_description\n");
    } elsif ($gene_taxons[0] == $taxons[1] and $gene_taxons[1] == $taxons[0]) {
        FILE->print("$genes[1],$gene_names[1],$genes[0],$gene_names[0],$homology_description\n");
    } else {
        die;
    }
}
FILE->close() or die;
print("Done!\n");
