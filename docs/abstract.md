# Oya: teaching a 3B model to move money safely in Pidgin and Yoruba

Lexsi Labs AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning)

## Abstract

Nigerians buy airtime, data and prepaid electricity and send money from their phones every day, often typing in
Nigerian Pidgin, Yoruba or a mix with English. A small open model that acts on these requests could run cheaply.
But a payment agent fails differently from a chatbot: a guessed account number pays a stranger, a "Done!"
without a transfer misleads, and text inside a tool result may be an attacker's instruction.

Oya is Cohere Labs' Tiny Aya Global (3.35B) fine-tuned with LoRA into a tool-calling wallet agent for English,
Pidgin, Yoruba and Yoruba-English mix. It uses seven tools on a mock wallet and must ask when details are
missing, refuse to overspend and ignore instructions planted in tool results. Its 2,080 training chats come from
running reference plans against the wallet, so every example is correct by construction. Training took 75
minutes on Kaggle's free T4 GPUs.

We use both Track 3 repos, connected: AgentTune supplies Tiny Aya's tool-calling prompt format, the trajectory
logs and the DPO trainer, and AuditKIT reads those trajectories as an independent second scorer. Scores come
from the wallet's ledger; a reply claiming an unmade payment, or any call using an attacker's planted number,
fails the task.

On 416 development tasks, task success rose from 39.2% to 97.4%, safety tasks from 18.8% to 100%, and tasks
where money moved that should not have fell from 18.3% to 2.4%. None of 32 planted injections was obeyed. On 30
held-out Yoruba and Pidgin requests checked by a native speaker, Oya solved all 30 with no wrong payment
(untrained: 23.3%), and 83.3% when the reply must also use the right language and ask for the right detail. On
18 more tasks built from native-written text found online, it solved all 18. Against OpenAI's GPT-6 Astra and
GPT-6.1 Sol (97% on those 30), it scores 100% at 16-bit and 93% in 4-bit on a laptop, where open models its size
score 20-50%, with no per-request fee.

For Track 3's self-healing problem, we ran two DPO rounds with AgentTune on preference pairs built from Oya's
own mistakes, behind a gate on measured accuracy. To fit Tiny Aya's 262,144-token vocabulary on a 16 GB T4, we
wrote a trainer subclass that scores only the judged turn. Round 1 drifted into Pidgin because its pairs were
unbalanced. Round 2 fixed the follow-up questions it targeted but sent Mama's airtime as a bank transfer and
lost ground on the development set. The gate caught both regressions and kept the supervised model.

The adapter, a 4-bit GGUF that kept the safety behaviour, the dataset with every run's trajectories, the code
and notebooks are public. The demo runs on Hugging Face's free ZeroGPU: it takes typed or spoken Pidgin and
Yoruba, and every payment waits for the user's confirmation.

Limits: the training data comes from templates, the held-out sets are small (the first drafted with a model and
checked by one native speaker), and the wallet is simulated.

## Links

- Code and notebooks: https://github.com/Timidan/lexsi-hack
- Write-up: https://github.com/Timidan/lexsi-hack#readme
- Models: https://huggingface.co/collections/timidan/oya-wallet-agent-models-6aedf0affb712ba373e3a17d
- Data: https://huggingface.co/collections/timidan/oya-wallet-agent-data-6aedf0b20bfbdecdb654e23c
- Demo: https://huggingface.co/spaces/timidan/oya
