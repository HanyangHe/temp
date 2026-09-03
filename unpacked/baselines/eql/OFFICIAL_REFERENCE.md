# Official EQL-Div baseline integration

This baseline executes the official Python-3/Theano EQL-Div core from the
user-provided `martius-lab/EQL` repository archive at commit
`cd93824ccd330b814ddd334dd25e8b0fd5eb3f77`.

## Unchanged upstream core

The following files are copied unchanged from `EQL-DIV-ICML-Python3/src`:

- `official_eql/src/mlfg_final.py`
- `official_eql/src/utils.py`
- `official_eql/src/model_selection_val_sparsity.py`

The adapter verifies their SHA-256 hashes before each sweep. These files own:

- EQL functional and multiplication layers;
- output-layer regularized division;
- upstream initialization, including denominator biases initialized to one;
- optimizer updates and three regularization phases;
- extrapolation-penalty sampling and loss;
- hard-L0 refit;
- active-functional-unit count;
- Vint-S model selection.

The PhDN-side adapter does not reimplement those operations. It only translates
data/configuration, launches independent upstream scans, calls the upstream
selector, reloads the selected state, and exports predictions/results.

## Interface scaling

The official experiments use inputs in approximately `[-1,1]` and outputs of
order one. For arbitrary Feynman domains, the adapter can apply an external
affine map of the training input box to `[-1,1]` plus training mean/std output
scaling. These maps are inverted before reporting physical errors. The official
network and training source remains unchanged. The exact Eq. (11) acceptance
demo disables scaling and loads the original upstream data files directly.

## Licensing and environment

The bundled upstream code is GPL-3.0; see `official_eql/LICENSE`. Its original
Python-3 Readme is retained as `official_eql/Readme.md`.

Use the separate environment definition:

```text
environment_official.yml
```

The legacy official stack requires a compatible Python/Theano/Lasagne
environment and should not be mixed into the PySR/PyTorch environment.
