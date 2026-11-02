#!/usr/bin/env python3
"""Public best-k CLI and Python API; never runs detection or downstream training.

Python consumers import get_best_k, k_candidates and select_k from this module.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT))
from config.datasets import DATASETS
from config.methods import GRAM_GRID_BETAS
from config.path import PATHS
from src.utils.gram_selection import (
    NOISE_TYPES, RHOS, budget_averages, get_best_k, k_candidates, load_runs,
    select_k, variant_for,
)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset','--datasets',dest='datasets',nargs='+',choices=tuple(DATASETS),default=tuple(DATASETS))
    parser.add_argument('--noise-type','--noise-types',dest='noise_types',nargs='+',choices=NOISE_TYPES,default=NOISE_TYPES)
    parser.add_argument('--rho','--rhos',dest='rhos',nargs='+',type=float,choices=RHOS,default=RHOS)
    parser.add_argument('--beta','--betas',dest='betas',nargs='+',type=float,choices=GRAM_GRID_BETAS,default=GRAM_GRID_BETAS)
    parser.add_argument('--results-root',type=Path,default=PATHS.results_root)
    parser.add_argument('--format',choices=('json','tsv','k','variant'),default='json')
    parser.add_argument('--required-seeds',nargs='+',type=int,help='fail if a selected variant lacks a requested evaluation seed')
    parser.add_argument('--list-betas',action='store_true')
    args=parser.parse_args(argv)
    if args.list_betas:
        print(' '.join(f'{b:g}' for b in GRAM_GRID_BETAS));return 0
    try:
        rows=select_k(k_candidates(load_runs(args.results_root,datasets=args.datasets,noise_types=args.noise_types,rhos=args.rhos,betas=args.betas)))
        for r in rows:
            available={int(s) for s in r['seeds'].split(',')}
            if args.required_seeds and not set(args.required_seeds)<=available:
                raise ValueError(f"selected {r['dataset']}/{r['noise_type']}/rho{r['rho']:g}/beta{r['beta']:g}: k={r['graph_neighbors']} ({r['gram_variant']}) only has seeds {sorted(available)}; requested {args.required_seeds}")
        if args.format in ('k','variant'):
            if len(rows)!=1:raise ValueError('--format k/variant requires exactly one scenario and beta')
            print(rows[0]['graph_neighbors' if args.format=='k' else 'gram_variant'])
        elif args.format=='tsv':
            for r in rows:print(f"{r['dataset']}\t{r['noise_type']}\t{r['rho']:g}\t{r['beta']:g}\t{r['graph_neighbors']}\t{r['gram_variant']}\t{r['seeds']}")
        else:
            fields=('dataset','noise_type','rho','beta','graph_neighbors','gram_variant','auprc','precision','selection_score','seeds')
            result=[{key:r[key] for key in fields} for r in rows]
            print(json.dumps(result[0] if len(result)==1 else result,indent=2))
    except (OSError,ValueError,KeyError) as exc:
        parser.exit(1,f'best-k: {exc}\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
