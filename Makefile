PYTHON ?= python3
S3_BUCKET_NAME ?= sales-deal-agent-docs
AWS_DEFAULT_REGION ?= us-east-1

.PHONY: setup setup-s3 run demo test eval eval-local embed compare docker-build deploy-infra destroy-infra clean

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	@echo "✓ Setup complete."

setup-s3:
	aws s3 mb s3://$(S3_BUCKET_NAME) --region $(AWS_DEFAULT_REGION) 2>/dev/null || true
	aws s3 sync data/documents/ s3://$(S3_BUCKET_NAME)/documents/
	@echo "✓ S3 bucket ready: s3://$(S3_BUCKET_NAME)"

run:
	$(PYTHON) -m streamlit run ui/app.py

demo:
	$(PYTHON) scripts/demo.py

test:
	$(PYTHON) -m pytest tests/ -v

eval:
	$(PYTHON) eval/run_eval.py

eval-local:
	$(PYTHON) eval/run_eval.py --local

embed:
	$(PYTHON) -c "from tools.document_search import build_vectorstore; build_vectorstore()"
	@echo "✓ FAISS index rebuilt"

compare:
	$(PYTHON) scripts/provider_comparison.py

docker-build:
	docker build -t sales-deal-agent .

deploy-infra:
	cd infra && pip install -r requirements.txt && cdk synth && cdk deploy --require-approval never

destroy-infra:
	cd infra && cdk destroy --force

clean:
	rm -rf data/faiss_index/ __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
