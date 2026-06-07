# Trace Replay: `heterogeneous-dates`
- Workflow: `tool_use_pipeline`
- Steps: `11`
- Baseline tokens: `12500`

| Step | Command | Tokens | A0 (no-control) | A2 (with-control) |
| ---: | --- | ---: | --- | --- |
| 1 | `ls -la /app` | 600 | CONTINUE | CONTINUE |
| 2 | `cat daily_temp_sf_high.csv daily_temp_sf` | 900 | CONTINUE | CONTINUE |
| 3 | `python merge_naive.py  # merge on raw da` | 1500 | CONTINUE | CONTINUE |
| 4 | `echo nan > /app/avg_temp.txt` | 800 | CONTINUE | FORCE_VERIFY |
| 5 | `pytest tests  # value test fails (nan)` | 1200 | CONTINUE | FORCE_VERIFY |
| 6 | `head low.csv  # inspect date formats` | 1300 | CONTINUE | FORCE_VERIFY |
| 7 | `python merge_fixed.py  # normalize + mer` | 1700 | CONTINUE | FORCE_VERIFY |
| 8 | `pytest tests  # all pass` | 1400 | CONTINUE | FINALIZE |
| 9 | `cat avg_temp.txt  # re-confirm value` | 1000 | CONTINUE | STOPPED |
| 10 | `pytest tests  # run again to be safe` | 1200 | CONTINUE | STOPPED |
| 11 | `write verbose summary of approach` | 900 | CONTINUE | STOPPED |

## Summary
| | Tokens |
| --- | ---: |
| A0 (no-control) | 12500 |
| A2 (with-control) | 9400 |
| Delta | +3100 |
| Reduction | 24.8% |
