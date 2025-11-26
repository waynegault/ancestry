# Outstanding Actions

1. **Live AI regression run** – Clear or set `SKIP_LIVE_API_TESTS=false`, run `python -m ai_interface`, and confirm Test 10 exercises the restored DNA/verification prompts end-to-end (capture logs that tie back to the README guardrail section).
2. **Telemetry baseline refresh** – After the live run passes, execute `python prompt_telemetry.py --baseline --provider <primary>` so the regression gate tracks the updated provider/prompt mix.
