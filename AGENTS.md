# Hướng dẫn cho agent

Đọc README.md, docs/RESEARCH_STATE.md, docs/NEXT_EXPERIMENTS.md và prompts/ASTRA_NEXT_RUN.md trước. User instruction mới rõ ràng vẫn có quyền thay hướng, nhưng ghi decision và không viết lại evidence cũ.

Current direction = decision value of uncertainty-aware explanation disclosure. Predictor/old endpoints frozen. Old architecture/acquisition/conformal plans under archive/ are historical, not active TODOs.

Separate REPORTED (repo evidence), REPRODUCED (actual artifact audit), PROPOSED and NOT_RUN. No fabricated human judgments, datasets, financial returns, guarantees or novelty. Keep source/version/citation and clear target definitions. Do not call stable probability correct probability or changed explanation a business error without separate evidence.

No model fishing on Taiwan/Polish inspected cohorts. Keep data splits, mask clusters, metrics and readme historical evidence unchanged. New experiments go to a new branch and unique ignored run directory. Do not force-push main or overwrite local work. Logs/worker transcripts remain local; one final report per stage, not repeated README appendices.

Model-fitting source/tests/configs were not changed by consolidation. Some historical scripts write docs: replay them in a detached worktree at the historical commit, or explicitly redirect output after testing. Do not recreate the old docs forest in current main.

Use MPS only if needed and verified; CPU multiprocessing is not independent subagent review. Log actual device, elapsed time, progress/ETA and warnings. Keep negative results and stop criteria.
