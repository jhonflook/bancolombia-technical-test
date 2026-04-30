up-deps:
	docker compose up --build -d

run-model:
	@echo "Available coins: bitcoin ethereum cardano"; \
	echo "Available models: ridge elasticnet random_forest gradient_boosting sarimax prophet"; \
	echo "Available targets: pct_change, price_usd"; \
	echo ""; \
	read -p "Coins (space-separated) [bitcoin ethereum cardano]: " coins; \
	read -p "Models (space-separated) [prophet]: " models; \
	read -p "Target [pct_change]: " target; \
	read -p "Number of trials [10]: " n_trials; \
	coins=$${coins:-bitcoin ethereum cardano}; \
	models=$${models:-prophet}; \
	target=$${target:-pct_change}; \
	n_trials=$${n_trials:-10}; \
	uv run python deploy/train_forecast_models_pct.py --coins $$coins --models $$models --target $$target --n-trials $$n_trials

# Export notebook to PDF with custom template (cover page)
# Usage: make export-pdf NOTEBOOK=notebooks/your_notebook.ipynb
export-pdf:
	uv run python scripts/export_pdf.py notebooks/statistical_analysis_of_time_series.ipynb