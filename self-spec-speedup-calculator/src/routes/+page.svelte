<script lang="ts">
  import Slider from '$lib/Slider.svelte';
  import {
    calculateSelfSpecSpeedup,
    defaultInputs,
    type SpeedupInputs
  } from '$lib/speedup';

  type PresetName = 'balanced' | 'optimistic' | 'conservative' | 'deep-skip';

  const presets: Record<PresetName, SpeedupInputs> = {
    balanced: defaultInputs,
    optimistic: {
      blockSize: 6,
      acceptanceRate: 0.78,
      headPortion: 0.34,
      flashheadHeadSpeedup: 7.5,
      flashheadAcceptanceMultiplier: 1.0,
      bodyShareOfNonHead: 0.97,
      draftBodyMultiplier: 0.34
    },
    conservative: {
      blockSize: 3,
      acceptanceRate: 0.45,
      headPortion: 0.22,
      flashheadHeadSpeedup: 2.6,
      flashheadAcceptanceMultiplier: 0.88,
      bodyShareOfNonHead: 0.95,
      draftBodyMultiplier: 0.68
    },
    'deep-skip': {
      blockSize: 5,
      acceptanceRate: 0.58,
      headPortion: 0.3,
      flashheadHeadSpeedup: 5.5,
      flashheadAcceptanceMultiplier: 0.93,
      bodyShareOfNonHead: 0.98,
      draftBodyMultiplier: 0.22
    }
  };

  let selectedPreset = $state<PresetName>('balanced');
  let inputs = $state<SpeedupInputs>({ ...defaultInputs });

  let estimate = $derived(calculateSelfSpecSpeedup(inputs));
  let bestSpeedup = $derived(
    Math.max(estimate.denseSpeedup, estimate.flashheadSpeedup)
  );
  let speedupScale = $derived(Math.max(1, bestSpeedup));

  function applyPreset(name: PresetName) {
    selectedPreset = name;
    Object.assign(inputs, presets[name]);
  }

  function resetInputs() {
    applyPreset('balanced');
  }

  function percent(value: number) {
    return `${(value * 100).toFixed(0)}%`;
  }

  function precisePercent(value: number) {
    return `${(value * 100).toFixed(1)}%`;
  }

  function multiplier(value: number) {
    return `${value.toFixed(2)}x`;
  }

  function number(value: number) {
    return value.toFixed(2);
  }
</script>

<svelte:head>
  <title>Self-Spec Speedup Calculator</title>
  <meta
    name="description"
    content="Interactive SvelteKit calculator for self-speculation speedup estimates."
  />
</svelte:head>

<main class="shell">
  <section class="summary">
    <div class="summary__copy">
      <p class="eyebrow">Self-speculation estimate</p>
      <h1>{multiplier(bestSpeedup)} total estimated speedup</h1>
      <p>
        Dense self-spec is {multiplier(estimate.denseSpeedup)}. FlashHead draft
        heads bring the live estimate to {multiplier(estimate.flashheadSpeedup)}.
      </p>
    </div>

    <div class="summary__metrics" aria-label="Current speedup metrics">
      <article class:accent={estimate.denseSpeedup >= estimate.flashheadSpeedup}>
        <span>Dense</span>
        <strong>{multiplier(estimate.denseSpeedup)}</strong>
        <small>{number(estimate.denseExpectedTokensPerBlock)} tokens/block</small>
      </article>
      <article class:accent={estimate.flashheadSpeedup > estimate.denseSpeedup}>
        <span>FlashHead</span>
        <strong>{multiplier(estimate.flashheadSpeedup)}</strong>
        <small>{number(estimate.flashheadExpectedTokensPerBlock)} tokens/block</small>
      </article>
      <article>
        <span>Best cost</span>
        <strong>
          {number(
            Math.min(
              estimate.denseCostPerGeneratedToken,
              estimate.flashheadCostPerGeneratedToken
            )
          )}
        </strong>
        <small>normal-token units</small>
      </article>
    </div>
  </section>

  <section class="workspace">
    <div class="controls">
      <div class="controls__bar">
        <div>
          <h2>Parameters</h2>
          <p>Uses the same model as the Python estimator.</p>
        </div>
        <div class="preset-row">
          <select
            bind:value={selectedPreset}
            aria-label="Preset"
            onchange={(event) =>
              applyPreset(event.currentTarget.value as PresetName)}
          >
            <option value="balanced">Balanced</option>
            <option value="optimistic">Optimistic</option>
            <option value="conservative">Conservative</option>
            <option value="deep-skip">Deep skip</option>
          </select>
          <button type="button" onclick={resetInputs}>Reset</button>
        </div>
      </div>

      <div class="slider-grid">
        <Slider
          label="Block size"
          bind:value={inputs.blockSize}
          min={1}
          max={16}
          step={1}
          displayValue={String(inputs.blockSize)}
          help="Number of drafted tokens attempted per self-spec block."
        />
        <Slider
          label="Acceptance rate"
          bind:value={inputs.acceptanceRate}
          min={0}
          max={1}
          step={0.01}
          displayValue={precisePercent(inputs.acceptanceRate)}
          help="Expected fraction of drafted tokens accepted by the verifier."
        />
        <Slider
          label="LM head portion"
          bind:value={inputs.headPortion}
          min={0.01}
          max={0.8}
          step={0.01}
          displayValue={precisePercent(inputs.headPortion)}
          help="Fraction of normal token cost spent in the language-model head."
        />
        <Slider
          label="Body share of non-head"
          bind:value={inputs.bodyShareOfNonHead}
          min={0}
          max={1}
          step={0.01}
          displayValue={precisePercent(inputs.bodyShareOfNonHead)}
          help="Share of the non-head remainder attributed to transformer body cost."
        />
        <Slider
          label="Draft body work"
          bind:value={inputs.draftBodyMultiplier}
          min={0}
          max={1}
          step={0.01}
          displayValue={precisePercent(inputs.draftBodyMultiplier)}
          help="How much work the draft transformer body does compared with the full body."
        />
        <Slider
          label="FlashHead head speedup"
          bind:value={inputs.flashheadHeadSpeedup}
          min={0.2}
          max={20}
          step={0.1}
          displayValue={multiplier(inputs.flashheadHeadSpeedup)}
          help="Speedup applied only to the draft language-model head cost."
        />
        <Slider
          label="FlashHead acceptance multiplier"
          bind:value={inputs.flashheadAcceptanceMultiplier}
          min={0}
          max={1}
          step={0.01}
          displayValue={precisePercent(inputs.flashheadAcceptanceMultiplier)}
          help="Percentage of the base acceptance rate retained by FlashHead."
        />
      </div>
    </div>

    <aside class="inspector">
      <section class="panel">
        <h2>Speedup</h2>
        <div class="bar-row">
          <span>Dense self-spec</span>
          <div class="track">
            <div
              class="fill dense"
              style={`width: ${(estimate.denseSpeedup / speedupScale) * 100}%`}
            ></div>
          </div>
          <strong>{multiplier(estimate.denseSpeedup)}</strong>
        </div>
        <div class="bar-row">
          <span>FlashHead self-spec</span>
          <div class="track">
            <div
              class="fill flash"
              style={`width: ${
                (estimate.flashheadSpeedup / speedupScale) * 100
              }%`}
            ></div>
          </div>
          <strong>{multiplier(estimate.flashheadSpeedup)}</strong>
        </div>
      </section>

      <section class="panel">
        <h2>Cost Portions</h2>
        <div class="stack" aria-label="Normal token cost portions">
          <span
            class="stack__head"
            style={`width: ${estimate.headFraction * 100}%`}
            title={`Head ${percent(estimate.headFraction)}`}
          ></span>
          <span
            class="stack__body"
            style={`width: ${estimate.bodyFraction * 100}%`}
            title={`Body ${percent(estimate.bodyFraction)}`}
          ></span>
          <span
            class="stack__other"
            style={`width: ${estimate.otherFraction * 100}%`}
            title={`Other ${percent(estimate.otherFraction)}`}
          ></span>
        </div>
        <dl class="facts">
          <div>
            <dt>Head</dt>
            <dd>{precisePercent(estimate.headFraction)}</dd>
          </div>
          <div>
            <dt>Body</dt>
            <dd>{precisePercent(estimate.bodyFraction)}</dd>
          </div>
          <div>
            <dt>Other</dt>
            <dd>{precisePercent(estimate.otherFraction)}</dd>
          </div>
        </dl>
      </section>

      <section class="panel">
        <h2>Per Block</h2>
        <dl class="facts two-column">
          <div>
            <dt>Dense accepted</dt>
            <dd>{precisePercent(estimate.acceptanceRate)}</dd>
          </div>
          <div>
            <dt>FlashHead accepted</dt>
            <dd>{precisePercent(estimate.flashheadAcceptanceRate)}</dd>
          </div>
          <div>
            <dt>Dense block cost</dt>
            <dd>{number(estimate.denseSelfSpecCostPerBlock)}</dd>
          </div>
          <div>
            <dt>FlashHead block cost</dt>
            <dd>{number(estimate.flashheadSelfSpecCostPerBlock)}</dd>
          </div>
          <div>
            <dt>Dense draft cost</dt>
            <dd>{number(estimate.denseDraftCostPerToken)}</dd>
          </div>
          <div>
            <dt>FlashHead draft cost</dt>
            <dd>{number(estimate.flashheadDraftCostPerToken)}</dd>
          </div>
        </dl>
      </section>
    </aside>
  </section>
</main>

<style>
  .shell {
    min-height: 100vh;
    padding: clamp(1rem, 2vw, 2rem);
    background:
      linear-gradient(140deg, rgba(15, 139, 141, 0.12), transparent 32rem),
      linear-gradient(320deg, rgba(225, 94, 71, 0.11), transparent 34rem),
      #f6f7f8;
  }

  .summary {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(22rem, 0.82fr);
    gap: 1rem;
    align-items: stretch;
    max-width: 1280px;
    margin: 0 auto 1rem;
  }

  .summary__copy,
  .summary__metrics,
  .controls,
  .panel {
    border: 1px solid #d7dde0;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.92);
    box-shadow: 0 18px 52px rgba(24, 32, 34, 0.08);
  }

  .summary__copy {
    padding: clamp(1.2rem, 4vw, 2.3rem);
  }

  .eyebrow {
    margin: 0 0 0.7rem;
    color: #0f6570;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
  }

  h1,
  h2,
  p {
    margin-top: 0;
  }

  h1 {
    max-width: 12ch;
    margin-bottom: 1rem;
    color: #152124;
    font-size: clamp(2.3rem, 5.4vw, 4.7rem);
    line-height: 0.98;
    letter-spacing: 0;
  }

  h2 {
    margin-bottom: 0.2rem;
    color: #1e2a2e;
    font-size: 1rem;
  }

  p {
    max-width: 62ch;
    margin-bottom: 0;
    color: #536168;
    line-height: 1.6;
  }

  .summary__metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1px;
    overflow: hidden;
    padding: 1px;
  }

  .summary__metrics article {
    display: grid;
    align-content: center;
    min-width: 0;
    min-height: 11rem;
    padding: 1.1rem;
    background: #ffffff;
  }

  .summary__metrics article.accent {
    background: #e9f6f4;
  }

  .summary__metrics span,
  .summary__metrics small {
    color: #607178;
  }

  .summary__metrics strong {
    color: #142426;
    font-size: clamp(1.7rem, 4vw, 3.25rem);
    line-height: 1;
    font-variant-numeric: tabular-nums;
  }

  .workspace {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(21rem, 0.65fr);
    gap: 1rem;
    max-width: 1280px;
    margin: 0 auto;
  }

  .controls {
    padding: 1rem;
  }

  .controls__bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .preset-row {
    display: flex;
    gap: 0.55rem;
    align-items: center;
  }

  select,
  button {
    min-height: 2.35rem;
    border: 1px solid #c8d2d6;
    border-radius: 8px;
    background: #ffffff;
    color: #1e2a2e;
  }

  select {
    padding: 0 2rem 0 0.8rem;
  }

  button {
    padding: 0 0.85rem;
    cursor: pointer;
  }

  button:hover,
  select:hover {
    border-color: #0f8b8d;
  }

  .slider-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
  }

  .inspector {
    display: grid;
    gap: 1rem;
    align-content: start;
  }

  .panel {
    padding: 1rem;
  }

  .bar-row {
    display: grid;
    grid-template-columns: 9.5rem minmax(5rem, 1fr) 4rem;
    gap: 0.75rem;
    align-items: center;
    margin-top: 0.85rem;
    color: #536168;
    font-size: 0.88rem;
  }

  .bar-row strong {
    color: #1a272a;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }

  .track {
    height: 0.8rem;
    overflow: hidden;
    border-radius: 999px;
    background: #e4e9ec;
  }

  .fill {
    height: 100%;
    min-width: 2px;
    border-radius: inherit;
  }

  .dense {
    background: #496ddb;
  }

  .flash {
    background: #0f8b8d;
  }

  .stack {
    display: flex;
    width: 100%;
    height: 1.15rem;
    overflow: hidden;
    margin: 0.9rem 0 1rem;
    border-radius: 999px;
    background: #e7ecef;
  }

  .stack span {
    min-width: 1px;
  }

  .stack__head {
    background: #496ddb;
  }

  .stack__body {
    background: #0f8b8d;
  }

  .stack__other {
    background: #e15e47;
  }

  .facts {
    display: grid;
    gap: 0.6rem;
    margin: 0;
  }

  .facts.two-column {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .facts div {
    display: grid;
    gap: 0.18rem;
    min-width: 0;
  }

  dt {
    color: #607178;
    font-size: 0.78rem;
  }

  dd {
    margin: 0;
    color: #1e2a2e;
    font-size: 1.05rem;
    font-variant-numeric: tabular-nums;
    font-weight: 800;
  }

  @media (max-width: 1320px) {
    .summary {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 980px) {
    .workspace {
      grid-template-columns: 1fr;
    }

    .summary__metrics {
      min-height: auto;
    }
  }

  @media (max-width: 720px) {
    .summary__metrics,
    .slider-grid,
    .facts.two-column {
      grid-template-columns: 1fr;
    }

    .controls__bar {
      align-items: stretch;
      flex-direction: column;
    }

    .preset-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
    }

    .bar-row {
      grid-template-columns: 1fr;
      gap: 0.35rem;
    }

    .bar-row strong {
      text-align: left;
    }
  }
</style>
