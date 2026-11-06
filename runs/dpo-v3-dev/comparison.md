### dpo-v3 vs sft-v2 on the dev set (416 tasks, compact prompt)

| Metric | v2 | v3 |
|---|---|---|
| Task success | 97.4% | 94.7% |
| Safety tasks handled correctly | 100.0% | 96.9% |
| Reply in the right language, asking for what is missing | 93.8% | 52.6% |
| Task success and right reply | 91.1% | 50.2% |
| Valid tool calls | 99.8% | 100.0% |
| Tool F1 vs reference route | 0.98 | 0.97 |
| Tasks where unexpected money moved | 2.4% | 4.3% |
| Extra tool calls per task | 0.02 | 0.04 |
| Success: en | 99.0% | 99.0% |
| Success: pcm | 94.2% | 92.3% |
| Success: yo | 96.2% | 88.5% |
| Success: yo_en | 100.0% | 99.0% |
| Success: airtime_contact | 68.8% | 46.9% |
| Success: airtime_direct | 100.0% | 100.0% |
| Success: balance | 100.0% | 100.0% |
| Success: chitchat | 100.0% | 100.0% |
| Success: data_contact | 96.9% | 96.9% |
| Success: data_direct | 100.0% | 100.0% |
| Success: electricity | 100.0% | 100.0% |
| Success: injection | 100.0% | 96.9% |
| Success: insufficient_funds | 100.0% | 100.0% |
| Success: missing_info | 100.0% | 90.6% |
| Success: transfer_contact | 100.0% | 100.0% |
| Success: transfer_direct | 100.0% | 100.0% |
| Success: unknown_contact | 100.0% | 100.0% |
