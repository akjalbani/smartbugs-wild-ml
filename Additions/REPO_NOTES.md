# Notes on the code repository (please act on these before submission)

1. The paper's numbers were produced by running the repository code on ALL usable contracts
   (config.MAX_CONTRACTS = None). The default in config.py and run_all.sh is 6,000 contracts.
   Tell readers how to reproduce the full run, e.g. add to README:
       python -c "from src import config, pipeline; config.MAX_CONTRACTS=None; pipeline.run()"
   On machines with ~4 GB RAM the full run runs out of memory. The included staged.py runs the
   same computation step by step with results saved to disk (recommended to add to the repo).
2. The McNemar table in the paper covers all 10 model pairs (pipeline.py only compares each
   model with the ensemble). staged.py computes all pairs with src.evaluate.mcnemar_test.
   dupcheck.py produces the duplicate-contract check.
3. PyTorch is not seeded, so CNN/GNN scores vary slightly between runs. Suggest adding
   torch.manual_seed(config.RANDOM_SEED) and np.random.seed(config.RANDOM_SEED) in pipeline.run().
   If you add this, re-run and update the numbers.
4. Placeholders still in the repo: CITATION.cff ("[Your first name]", "[your-username]"),
   LICENSE ("[YOUR NAME]"), notebook Colab link ("[your-username]"). Fill these before making it public.
5. data_download.py clones the whole smartbugs-results repository (about 4.8 GB). Only
   metadata/results_wild.json (32 MB) is needed; consider downloading just that file.
6. Commit artifacts/metrics_full.json (included here) so the paper's numbers are visible in the repo.
