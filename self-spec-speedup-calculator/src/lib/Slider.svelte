<script lang="ts">
  type Props = {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    displayValue: string;
    help: string;
  };

  let {
    label,
    value = $bindable(),
    min,
    max,
    step,
    displayValue,
    help
  }: Props = $props();
</script>

<label class="slider" title={help}>
  <span class="slider__top">
    <span>{label}</span>
    <output>{displayValue}</output>
  </span>
  <input bind:value type="range" {min} {max} {step} aria-label={label} />
  <span class="slider__bounds">
    <span>{min}</span>
    <span>{max}</span>
  </span>
</label>

<style>
  .slider {
    display: grid;
    gap: 0.65rem;
    padding: 1rem;
    border: 1px solid #d6dde2;
    border-radius: 8px;
    background: #ffffff;
  }

  .slider__top,
  .slider__bounds {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
  }

  .slider__top span {
    color: #263236;
    font-size: 0.94rem;
    font-weight: 700;
  }

  output {
    color: #0f6570;
    font-variant-numeric: tabular-nums;
    font-weight: 800;
    white-space: nowrap;
  }

  input[type='range'] {
    width: 100%;
    height: 1.4rem;
    accent-color: #0f8b8d;
    cursor: pointer;
  }

  .slider__bounds {
    color: #65747a;
    font-size: 0.76rem;
    font-variant-numeric: tabular-nums;
  }
</style>
