# Self-Spec Speedup Calculator

A small SvelteKit app for interactively estimating self-speculation speedups with sliders.

The calculation mirrors `skip_search_spec/analysis/estimate_self_spec_speedup.py`:

- normal greedy decoding costs `1.0` per generated token
- a self-spec block emits `1 + block_size * acceptance_rate` expected tokens
- each block costs one fixed verifier pass plus `block_size` draft passes
- draft body work is a percentage of the full transformer body
- FlashHead changes the draft head cost and can only retain or reduce acceptance

## Run

```sh
npm install
npm run dev
```

Then open the local URL printed by Vite.
