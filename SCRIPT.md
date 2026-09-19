# The 180 seconds — word for word

Rehearse out loud twice against a stopwatch. If you overrun, **cut a sentence, never speak faster.**
Stage directions in brackets. Numbers in `[[ ]]` are yours to fill.

---

### 0:00 — 0:20 · Six clips

*[Six clips already looping silently as you start. Do not introduce yourself.]*

> "One of these six is a real robot. The other five, our system dreamed. Shout out which one is real."

*[Let them shout. Two or three seconds. Do not help.]*

> "It's number `[[n]]`. Nobody gets that right."

*[Beat.]*

---

### 0:20 — 0:32 · The problem, and who got here first

> "Comparing two robot policies costs a person a week standing next to an arm. The study we validate against took fifteen hundred real trials to produce six numbers.
>
> Three groups have already automated this with world models. **All three reported accuracy. None reported precision** — whether the thing gives you the same answer twice."

---

### 0:32 — 0:52 · The called shot

> "One policy, one task — close a drawer. The industry-standard physics simulator scores it four percent. It says the policy is broken.
>
> Before we look at the real robot, ours says `[[__]]` percent."

*[Click. Reveal.]*

> "Ninety-two. Eighty-eight points of error. Trust the simulator and you throw away a policy that works nine times in ten."

---

### 0:52 — 1:10 · The four numbers

> "So — these are your own three criteria for an evaluator. Low variance: split-half reliability, `[[__]]`. Strong separation: we resolve `[[__]]` points and below that we say we cannot tell. Agrees with the expert: kappa `[[__]]` against a measured human ceiling.
>
> Plus two only this platform makes measurable: where dreaming stops preserving the ranking, and what a thousand rollouts costs."

---

### 1:10 — 1:35 · The economics, and what is wrong with it

> "A week of a person and a dedicated rig **should** cost about a minute and eleven dollars. Hold that thought.
>
> Because the shape matters: evaluation is a recurring inference workload with no marginal human cost. Every fine-tune triggers a run. Latency-tolerant, embarrassingly parallel, and it grows as the customer succeeds.
>
> What we cut: adapting the world model — eight H100s in every published recipe, so we didn't pretend. What came out worse than we hoped: `[[the disappointing result]]`. And the bias we have **not** bounded — the policies and the world model share a backbone lineage, which is a plausible reason our numbers flatter us.
>
> Everything so far is a claim. Here it is running."

*[Switch to the app. Do not come back to the deck.]*

---

### 1:35 — 3:00 · LIVE. Eighty-five seconds, four beats.

**Beat one, the wall — 12 seconds.** *[Twelve tiles already dreaming; you started the batch at 0:00.]*

> "Twelve robot policies, running right now. None of these robots exist — each tile is a policy deciding what to do next, and a video model inventing what happens as a result. Watch one: it grows every two seconds as the next chunk finishes."

**Beat two, arrow keys — 20 seconds.** *[Blow one tile up to full screen. YOU hold the keyboard.]*

> "You might reasonably wonder whether those are pre-recorded. So — I'm pressing the right arrow."

*[Press. The wait is about two seconds. Fill it, do not apologise for it:]*

> "It's inventing the next second and a half of video right now, from that one command."

*[The arm moves right. Press up, then down. Three or four presses total.]*

> "No policy, no language model in that loop — my keypress goes straight in as an action. Nothing pre-recorded answers a keypress."

*[Do NOT offer them the keyboard here. Save it for the booth.]*

---

**Beat three, the burst — 40 seconds.****Beat four, the dial — 13 seconds.**

> "This slider is generation settings. Cost per rollout, a dollar down to four cents — ranking holds. Holds. **There.** That's where it breaks, and now we can tell you not to go past it."

*[Land on the scoreboard with its error bars showing. Then, over it, the closing line:]*

> "We measured the ruler before we trusted it. That's the whole project."

*[Stop. Say nothing else.]*

---

## If asked anything in Q&A, lead with this

> "We also ran the reverse-validation experiment — whether training loss can validate an LLM judge. The answer is no, and we can show why: scramble a judge's per-item scores until it's barely better than a coin flip and its loss correlation doesn't move at all. It stays at minus nought point nine two."

That sentence is aimed at one judge specifically. It answers a question he published as open future work, and the answer is a negative with a mechanism.

## If the live app dies

Present the demo slide and keep talking through what it would have shown. Do not debug on stage. You have sixty seconds and no more.

## Timing discipline

| Beat | Sec | Cumulative |
|---|---|---|
| Six clips | 20 | 0:20 |
| Problem + prior work | 12 | 0:32 |
| Called shot | 20 | 0:52 |
| Four numbers | 18 | 1:10 |
| Economics + limits | 25 | 1:35 |
| **LIVE — wall 12, arrow keys 20, burst 40, dial 13** | **85** | **3:00** |

**The live app is the finale and gets 85 of 180 seconds.** Every slide was trimmed to pay for it.

The structural point: the eleven dollars is now a **promise** on the economics slide that the burst then **keeps**. Say "should cost" and make them wait ninety seconds for the receipt. That is much stronger than describing it afterwards.

If you are running long at 1:10, drop the fourth card on the numbers slide. Never rush the live block and never rush the ten seconds of silence after the burst button.
