# Submission pack and checklist

Lexsi Labs AI Safety & Alignment Hackathon, Track 3: Agentic Fine-tuning. Checked 2026-11-08; submitted 2026-11-10.

| | |
|---|---|
| Form | https://hackathon.lexsi.ai/submission (a Tally form; the link is also emailed before the deadline) |
| Deadline | Nov 15, 2026, 11:59 PM AoE = 12:59 on Mon Nov 16 in Lagos (moved from Nov 14; the site and the form say Nov 15) |
| Resubmitting | allowed: "only the final submission prior to the submission deadline will be considered" |
| Rules read | hackathon.lexsi.ai (rules and FAQ), the kickoff slides, the form itself |

## Form answers, in order

1. **Email:** the one you registered with.
2. **Abstract:** the text between "Abstract" and "Links" in [abstract.md](abstract.md) (492 words; the limit is 500).
3. **Discord username:** yours.
4. **Hugging Face collections:**
   - models: https://huggingface.co/collections/timidan/oya-wallet-agent-models-6aedf0affb712ba373e3a17d
   - data: https://huggingface.co/collections/timidan/oya-wallet-agent-data-6aedf0b20bfbdecdb654e23c
5. **Article:** https://github.com/Timidan/lexsi-hack#readme. The box suggests a huggingface.co/blog link, but free
   accounts cannot post Hub articles, and the FAQ says: "You can submit a README file in a GitHub repo instead."
6. **GitHub:** https://github.com/Timidan/lexsi-hack
7. **Repos used:** AgentTune, AuditKIT.
8. **Model:** Tiny Aya Global.
9. **Track:** Track 3: Agentic Fine-tuning.
10. **Problem statements:** Reliable tool calling; Self-healing agents.
11. **Consents:** sharing by Hugging Face and Cohere, and being tagged by the organisers, are your choice. The last
    box ("All repos and collections are public and final as of Nov 15, 11:59 PM AoE") must be ticked, and it means no
    pushes to GitHub and no uploads to the Hugging Face repos or the Space after the deadline.

## Requirements

| Weight | Requirement | Status | Where a judge sees it |
|---|---|---|---|
| Mandatory | Core model is one of the 7 Cohere Labs models | done: Tiny Aya Global | README section 3 |
| Mandatory | Track repos chained into one pipeline ("at least 2 ... except Track 3") | done: AgentTune trains (DPO) and sets the trajectory format; AuditKIT scores those trajectories | README section 2 |
| Mandatory | Models, adapters, logs, datasets and benchmarks in public, un-gated HF collections | done: both collections and all 6 repos load logged out, none gated | README sections 5 and 6 |
| Mandatory | Write-up with the 9 required sections (HF article, or GitHub README if articles are not available) | done: the README | README |
| Mandatory | Abstract under 500 words | done: 492 | [abstract.md](abstract.md) |
| Mandatory | Repo README covers reproducibility | done | README appendix |
| Mandatory | Original work; one final submission | done | |
| Weighted | Write-up clarity, practical usefulness, code and notebook quality (each /10), overall /100 | | README, notebooks, demo |
| Preferred | ROI comparison against a closed model | done | README section 7, "ROI" |
| If relevant | A Hugging Face Space | done: ZeroGPU demo | https://huggingface.co/spaces/timidan/oya |

Not required for round 1: a demo video (the form has no field for one), screenshots, a deck. Round 2: up to 20
shortlisted teams, announced by Nov 16, get until Nov 22, 11:59 PM AoE to "prove their thesis and submit a video
presentation or slides for final judging" (hackathon.lexsi.ai, read 2026-11-10).

## Final gate

- [x] Every mandatory item above met, checked against the live pages on 2026-11-08
- [x] Links in the form answers and in the abstract load logged out (all return 200)
- [x] Demo runs: logged in, a request works; logged out, Hugging Face allows only a few GPU runs a day, and the
      demo then says how to carry on (log in, or the Before / after replay, which needs no GPU)
- [x] Repo public, with the write-up, setup and licences in the README
- [x] AgentTune and AuditKIT visible in the abstract and README section 2, not only in code
- [x] Form submitted by the user on 2026-11-10 (Tally showed "Submission received!"; a confirmation goes to the registered email)
- [ ] Nothing pushed or uploaded after Nov 15, 11:59 PM AoE

The Space sleeps after two days without visitors and wakes on the next visit in about a minute.

**Verdict:** submitted on 2026-11-10 with every mandatory item met. Resubmitting before the deadline replaces it.
