### sft-v2: before vs after (416 held-out test tasks, compact prompt)

| Metric | untrained | fine-tuned |
|---|---|---|
| Task success | 39.2% | 97.4% |
| Safety tasks handled correctly | 18.8% | 100.0% |
| Valid tool calls | 78.4% | 99.8% |
| Tool F1 vs reference route | 0.62 | 0.98 |
| Tasks where unexpected money moved | 18.3% | 2.4% |
| Extra tool calls per task | 0.16 | 0.02 |
| Success: en | 43.3% | 99.0% |
| Success: pcm | 34.6% | 94.2% |
| Success: yo | 32.7% | 96.2% |
| Success: yo_en | 46.2% | 100.0% |
| Success: airtime_contact | 0.0% | 68.8% |
| Success: airtime_direct | 93.8% | 100.0% |
| Success: balance | 100.0% | 100.0% |
| Success: chitchat | 50.0% | 100.0% |
| Success: data_contact | 0.0% | 96.9% |
| Success: data_direct | 96.9% | 100.0% |
| Success: electricity | 93.8% | 100.0% |
| Success: injection | 0.0% | 100.0% |
| Success: insufficient_funds | 75.0% | 100.0% |
| Success: missing_info | 0.0% | 100.0% |
| Success: transfer_contact | 0.0% | 100.0% |
| Success: transfer_direct | 0.0% | 100.0% |
| Success: unknown_contact | 0.0% | 100.0% |
