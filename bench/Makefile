.PHONY: all clean
.DELETE_ON_ERROR:
.SECONDARY:

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

all: ability_expvar_accuracy.jsonl irf_lh_accuracy.jsonl

clean:
	rm -f ability_expvar_accuracy.jsonl irf_lh_accuracy.jsonl ability_expvar_accuracy.parquet irf_lh_accuracy.parquet

ability_expvar_accuracy.jsonl:
	julia --project=.. ability_expvar_accuracy.jl $@

irf_lh_accuracy.jsonl:
	julia --project=.. irf_lh_accuracy.jl $@

ability_expvar_accuracy.csv: ability_expvar_accuracy.jsonl
	./proc_ability_expvar_accuracy.sh $< $@

irf_lh_accuracy.parquet: irf_lh_accuracy.jsonl
	./proc_irf_lh_accuracy.sh $< $@ irf_lh_item_params.parquet
