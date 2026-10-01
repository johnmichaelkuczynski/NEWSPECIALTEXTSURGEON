---
name: Fresh Tree partial output
description: Why long book runs need unmistakable activity feedback and usable partial results
---

For long Fresh Tree book runs, make active status visible throughout model waits and emit validated output as each thesis completes. STOP must retain and allow downloading every emitted node without implying that an unfinished chapter or tier is complete.

Show actual provider text from the initial response onward, not just a running timer or a count of validated nodes. Provisional streamed text must stay visibly separate from retained, exportable nodes.

**Why:** The user explicitly confirmed that a run showing zero nodes was working and had produced content; they stopped it because that counter made the app appear dead. Preserve the working generation logic rather than treating this visibility failure as proof of generation failure.

**How to apply:** Stream the model response during generation and review, identify the provider and stage, and distinguish received text from validated nodes. Do not substitute heartbeat messages or elapsed time for actual streamed content.

**Why:** The user pressed STOP mid-book specifically because they could not tell whether generation was still working, not because the result was unwanted. A run may take many minutes, and withholding a whole tier until every thesis passes review makes valid partial work appear lost. The user confirmed that Fresh Tree works well; preserve its incremental validated output when changing other analysis features.

**How to apply:** When changing generation or review, preserve incremental validated thesis events and distinguish a partial chapter from a completed one. Do not stream unreviewed model text as final output, and do not rely on server-request completion as proof that every chapter finished.

Resume must distinguish a thesis whose children were deliberately dropped by review from a thesis that has not yet been processed. Do not retry rejected work merely to fill a requested depth.

**Why:** The legitimacy rules permit only bounded regeneration. Retrying dropped branches on every resume would bypass that limit and confuse deliberate rejection with interrupted generation.

**How to apply:** Retain explicit completion for reviewed theses even when no child survives. Progress and visible depth may differ; missing descendants alone are not evidence of unfinished work.