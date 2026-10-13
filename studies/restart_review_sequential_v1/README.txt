Sequential incident-review study preparation. Unrun, AI-assisted development cases; no human validation or registration.

Read protocol.txt, analysis_plan.txt, case_authoring.txt, and human_extension.txt.
Run offline checks: python3 -m unittest discover -s studies/restart_review_sequential_v1 -p 'test_*.py' -v
Freeze development only: python3 studies/restart_review_sequential_v1/freeze.py --output /path/to/new/release
Run on the existing pinned Gemma GPU environment: python /path/to/release/run_model.py --release /path/to/release --output /path/to/new/attempt
Analyze: python /path/to/release/analyze.py --release /path/to/release --output /path/to/attempt
Use --resume only after documenting an infrastructure interruption. Never retry an answer based on its content.
Main inference is disabled until independent cases, qualified checks, sample/margin choices and a separate reviewed freeze are supplied.
