#!/usr/bin/env julia

# Adapted from bin/cellfishing in the CellFishing.jl repo

using CellFishing
using HDF5
using DocOpt

using Dates
using Random
using SparseArrays
using StatsBase
using Statistics

args = docopt("""
Usage:
    cellfishing [--n-bits=<n-bits> --n-lshashes=<n-lshashes> --infer-stats=<infer-stats> --genes=<genes> --n-neighbors=<n-neighbors> --cutoff=<cutoff> --min-hits=<min-hits> --majority=<majority> --annotation=<annotation> --seed=<seed> --clean=<clean> --subsample-ref=<subsample-ref>] <reference> <query> <output>

CellFishing ― Cell search engine for scRNA-seq.

Options:
    -h --help                        Show this help.
    --version                        Show version.
    --n-bits=<n-bits>                The number of bits for hashing [default: 128].
    --n-lshashes=<n-lshashes>        The number of locality-sensitive hashes [default: 4].
    --infer-stats=<infer-stats>      Infer statistics from the query, the database, or both [default: both].
    --genes=<genes>                  Specify genes to use instead of selecting by CellFishing [default: cf_genes].
    --n-neighbors=<n-neighbors>      The number of nearest neighbors to search for [default: 10].
    --cutoff=<cutoff>...             Hamming distance cutoff [default: 150].
    --min-hits=<min-hits>            Minimal number of hits below which query will be rejected [default: 2].
    --majority=<majority>            Minimal fraction required to pass majority vote [default: 0.5].
    --annotation=<annotation>        Choose annotation column in meta table [default: cell_ontology_class].
    --clean=<clean>                  Clean data based on certain meta column [default: None].
    --subsample-ref=<subsample-ref>  Subsample reference data [default: None].
    --seed=<seed>                    Set random seed [default: None].
""")

function loadh5(datafile, clean=nothing, subsample=nothing, annotation=nothing, genes=nothing)
    HDF5.h5open(datafile) do file
        counts = read(file, "/exprs")
        if isa(counts, Dict)
            counts = SparseMatrixCSC(
                counts["shape"][2], counts["shape"][1],
                counts["indptr"] .+ 1, counts["indices"] .+ 1,
                counts["data"]
            )
        else
            counts = Matrix(counts)
        end
        featurenames = read(file, "/var_names")
        cellnames = read(file, "/obs_names")
        if annotation != nothing
            annotations = read(file, "/obs/$(annotation)")
        else
            annotations = nothing
        end
        if clean != nothing
            clean = read(file, "/obs/$(clean)")
            mask = clean .∈ [["" "na" "NA" "nan" "NaN"]]
            if count(mask) > 0
                print("Cleaning removed $(count(mask)) cells. ")
                counts = counts[:, .!mask]
                cellnames = cellnames[.!mask]
                if annotations != nothing
                    annotations = annotations[.!mask]
                end
            end
        end
        if subsample != nothing
            subsample = sample(1:size(counts, 2), subsample, replace=false)
            counts = counts[:, subsample]
            cellnames = cellnames[subsample]
            if annotations != nothing
                annotations = annotations[subsample]
            end
        end
        if genes != nothing
            genes = read(file, "/uns/$(genes)")
        else
            genes = nothing
        end
        return (counts=counts, featurenames=featurenames, cellnames=cellnames, annotations=annotations, genes=genes)
    end
end

macro message(msg, block)
    quote
        print(stderr, "  ", rpad(string($(esc(msg)), " "), 25, '\u2015'), " ")
        starttime = now()
        $(esc(block))
        println(stderr, Dates.canonicalize(Dates.CompoundPeriod(now() - starttime)))
    end
end

function writeresult(file, predictions, neighbors, time_per_cell)
    HDF5.h5open(file, "w") do file
        g = g_create(file, "prediction")
        for item in predictions
            k, v = item
            g[String(k)] = v
        end
        file["indexes"] = neighbors.indexes
        file["hammingdistances"] = neighbors.hammingdistances
        file["time"] = time_per_cell
    end
end

using Random: seed!
if args["--seed"] != "None"
    seed!(parse(Int, args["--seed"]))
end

if args["--clean"] == "None"
    args["--clean"] = nothing
end
if args["--subsample-ref"] == "None"
    args["--subsample-ref"] = nothing
else
    args["--subsample-ref"] = parse(Int, args["--subsample-ref"])
end

if args["--genes"] == "cf_genes"
    args["--genes"] = nothing
end

@message "Loading data" begin
    ref = loadh5(args["<reference>"], args["--clean"], args["--subsample-ref"], args["--annotation"], args["--genes"])
end
@message "Selecting features" begin
    if ref.genes == nothing
        features = CellFishing.selectfeatures(ref.counts, ref.featurenames)
    else
        nzv_features = ref.featurenames[dropdims(std(ref.counts, dims=2) .> 0, dims=2)]
        println("Removing $(size(setdiff(ref.genes, nzv_features), 1)) genes with zero variance.")
        ref_genes = intersect(nzv_features, ref.genes)
        features = CellFishing.Features(ref.featurenames, ref.featurenames .∈ [ref_genes])
    end
end
@message "Creating a database" begin
    n_bits = parse(Int, args["--n-bits"])
    n_lshashes = parse(Int, args["--n-lshashes"])
    database = CellFishing.CellIndex(ref.counts, features, metadata=ref.annotations, n_bits=n_bits, n_lshashes=n_lshashes)
end

k = parse(Int, args["--n-neighbors"])
@message "Loading query data" begin
    query = loadh5(args["<query>"], args["--clean"])
end
@message "Searching the database" begin
    inferstats = args["--infer-stats"]
    inferstats = inferstats == "query"    ? :query :
                    inferstats == "database" ? :database :
                    inferstats == "both"     ? :both :
                    error("invalid --infer-stats option")
    start_time = now()
    neighbors = CellFishing.findneighbors(k, query.counts, query.featurenames, database, inferstats=inferstats)
end

time_per_cell = nothing
@message "Doing annotation" begin
    predictions = Dict()
    majority = parse(Float64, args["--majority"])
    min_hits = parse(Int, args["--min-hits"])
    cutoffs = split(args["--cutoff"], ",")
    for cutoff in cutoffs
        global time_per_cell
        predictions[cutoff] = Vector{String}()
        _cutoff = parse(Int, cutoff)
        for qid in 1:size(neighbors.indexes, 2)
            mask = neighbors.hammingdistances[:, qid] .≤ _cutoff
            annotations = database.metadata[neighbors.indexes[mask, qid]]
            if size(annotations, 1) ≤ min_hits
                push!(predictions[cutoff], "rejected")
                continue
            end
            max_k, max_v = nothing, 0
            for item in countmap(annotations)
                k, v = item
                if v > max_v
                    max_k, max_v = k, v
                end
            end
            if max_v ≤ majority * size(annotations, 1)
                push!(predictions[cutoff], "ambiguous")
            else
                push!(predictions[cutoff], string(max_k))
            end
        end
        if time_per_cell == nothing
            time_per_cell = (now() - start_time).value / size(query.counts, 2)
        end
    end
end

@message "Saving results" begin
    writeresult(args["<output>"], predictions, neighbors, time_per_cell)
end
