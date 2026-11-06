### dpo-v3 vs sft-v2 on the human set (30 tasks, compact prompt)

| Metric | v2 | v3 |
|---|---|---|
| Task success | 100.0% | 96.7% |
| Safety tasks handled correctly | 100.0% | 100.0% |
| Reply in the right language, asking for what is missing | 83.3% | 60.0% |
| Task success and right reply | 83.3% | 60.0% |
| Valid tool calls | 100.0% | 100.0% |
| Tool F1 vs reference route | 1.00 | 0.98 |
| Tasks where unexpected money moved | 0.0% | 3.3% |
| Extra tool calls per task | 0.00 | 0.03 |
| Success: pcm | 100.0% | 100.0% |
| Success: yo | 100.0% | 93.3% |
| Success: airtime_contact | 100.0% | 50.0% |
| Success: airtime_direct | 100.0% | 100.0% |
| Success: balance | 100.0% | 100.0% |
| Success: chitchat | 100.0% | 100.0% |
| Success: data_contact | 100.0% | 100.0% |
| Success: data_direct | 100.0% | 100.0% |
| Success: electricity | 100.0% | 100.0% |
| Success: injection | 100.0% | 100.0% |
| Success: insufficient_funds | 100.0% | 100.0% |
| Success: missing_info | 100.0% | 100.0% |
| Success: transfer_contact | 100.0% | 100.0% |
| Success: transfer_direct | 100.0% | 100.0% |
| Success: unknown_contact | 100.0% | 100.0% |
