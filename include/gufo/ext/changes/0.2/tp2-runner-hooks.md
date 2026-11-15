# TP2 runner hooks

Additive. Text runners may join a two-host pair, which needs both ranks to make
the same calls with the same plans.

Added to `TextModelRunner`, all with defaults a single-host runner can keep:
`EncodePromptContext`, `DecodePromptContext`, `PlanDecodeStep`,
`DecodeStepPlanned`, `PlanDecodeBatch`, `DecodeBatchPlanned`, `BeginTpOperation`
and `EndTpOperation`.

Added `TextRunnerTpOptions` and `TextRunnerOptions::tp`, from which a provider
builds its collective transport; a batched decode member's `drafts`; and the
request's `response_format_json` and `response_format_responses`, from which a
peer rebuilds a constraint.
