# Code for the NAV re-test

- `p1.py`: Codex's P1 simulator interface, re-implemented (host, prompt pieces, tool definition, outcome accounting).
- `nav_runner.py`: runs base, NAV4 and NAV9 episodes on Tinker and the passive benchmarks. Needs `TINKER_API_KEY` and `TINKER_PROJECT_ID` in the environment.
- `analyze_nav.py`: the tables and paired tests in [../NAV_RETEST.md](../NAV_RETEST.md).
- `validate_p1.py`: the check against Codex's saved prompts and tool results. Those fixtures came from Codex's private archive and are not published, so this script documents the check rather than running for you.
- `bars.py`: fetches and caches IBKR daily bars through Clayton's local IB Gateway framework (not published); any equivalent bar source works.

Requires `tinker==0.33.1` and `tinker-cookbook==0.5.7` (its `GptOssRenderer` is what renders the prompts).
