export type SpeedupInputs = {
  blockSize: number;
  acceptanceRate: number;
  headPortion: number;
  flashheadHeadSpeedup: number;
  flashheadAcceptanceMultiplier: number;
  bodyShareOfNonHead: number;
  draftBodyMultiplier: number;
};

export type SpeedupEstimate = {
  blockSize: number;
  acceptanceRate: number;
  flashheadAcceptanceRate: number;
  headFraction: number;
  bodyFraction: number;
  otherFraction: number;
  bodyShareOfNonHead: number;
  draftBodyMultiplier: number;
  verifierBlockCostMultiplier: number;
  denseExpectedTokensPerBlock: number;
  flashheadExpectedTokensPerBlock: number;
  normalCostForDenseExpectedTokens: number;
  normalCostForFlashheadExpectedTokens: number;
  denseSelfSpecCostPerBlock: number;
  flashheadSelfSpecCostPerBlock: number;
  denseSpeedup: number;
  flashheadSpeedup: number;
  denseCostPerGeneratedToken: number;
  flashheadCostPerGeneratedToken: number;
  normalCostPerGeneratedToken: number;
  denseDraftCostPerToken: number;
  flashheadDraftCostPerToken: number;
};

export const defaultInputs: SpeedupInputs = {
  blockSize: 4,
  acceptanceRate: 0.62,
  headPortion: 0.28,
  flashheadHeadSpeedup: 4.0,
  flashheadAcceptanceMultiplier: 0.95,
  bodyShareOfNonHead: 0.97,
  draftBodyMultiplier: 0.42
};

export function calculateSelfSpecSpeedup(inputs: SpeedupInputs): SpeedupEstimate {
  const headFraction = normalizePortionFraction(inputs.headPortion, 'headPortion');
  const bodyShare = normalizePortionFraction(
    inputs.bodyShareOfNonHead,
    'bodyShareOfNonHead'
  );
  const nonHeadFraction = 1.0 - headFraction;
  const bodyFraction = nonHeadFraction * bodyShare;
  const otherFraction = nonHeadFraction - bodyFraction;

  validateInputs(inputs, headFraction, bodyFraction, otherFraction, bodyShare);

  const normalCostPerToken = 1.0;
  const verifierBlockCostMultiplier = 1.0;
  const verifierCostPerBlock = normalCostPerToken * verifierBlockCostMultiplier;
  const denseDraftCostPerToken =
    headFraction + bodyFraction * inputs.draftBodyMultiplier + otherFraction;
  const flashheadDraftCostPerToken =
    headFraction / inputs.flashheadHeadSpeedup +
    bodyFraction * inputs.draftBodyMultiplier +
    otherFraction;

  const denseAcceptanceRate = clampProbability(inputs.acceptanceRate);
  const flashheadAcceptanceRate = clampProbability(
    inputs.acceptanceRate * inputs.flashheadAcceptanceMultiplier
  );

  const denseExpectedTokens =
    1.0 + inputs.blockSize * denseAcceptanceRate;
  const flashheadExpectedTokens =
    1.0 + inputs.blockSize * flashheadAcceptanceRate;

  const normalCostForDenseExpectedTokens =
    denseExpectedTokens * normalCostPerToken;
  const normalCostForFlashheadExpectedTokens =
    flashheadExpectedTokens * normalCostPerToken;

  const denseSelfSpecCost =
    verifierCostPerBlock + inputs.blockSize * denseDraftCostPerToken;
  const flashheadSelfSpecCost =
    verifierCostPerBlock + inputs.blockSize * flashheadDraftCostPerToken;

  return {
    blockSize: inputs.blockSize,
    acceptanceRate: denseAcceptanceRate,
    flashheadAcceptanceRate,
    headFraction,
    bodyFraction,
    otherFraction,
    bodyShareOfNonHead: bodyShare,
    draftBodyMultiplier: inputs.draftBodyMultiplier,
    verifierBlockCostMultiplier,
    denseExpectedTokensPerBlock: denseExpectedTokens,
    flashheadExpectedTokensPerBlock: flashheadExpectedTokens,
    normalCostForDenseExpectedTokens,
    normalCostForFlashheadExpectedTokens,
    denseSelfSpecCostPerBlock: denseSelfSpecCost,
    flashheadSelfSpecCostPerBlock: flashheadSelfSpecCost,
    denseSpeedup: normalCostForDenseExpectedTokens / denseSelfSpecCost,
    flashheadSpeedup:
      normalCostForFlashheadExpectedTokens / flashheadSelfSpecCost,
    denseCostPerGeneratedToken: denseSelfSpecCost / denseExpectedTokens,
    flashheadCostPerGeneratedToken:
      flashheadSelfSpecCost / flashheadExpectedTokens,
    normalCostPerGeneratedToken: normalCostPerToken,
    denseDraftCostPerToken,
    flashheadDraftCostPerToken
  };
}

function validateInputs(
  inputs: SpeedupInputs,
  headFraction: number,
  bodyFraction: number,
  otherFraction: number,
  bodyShare: number
) {
  if (inputs.blockSize <= 0) {
    throw new Error(`blockSize must be positive, got ${inputs.blockSize}.`);
  }

  if (inputs.acceptanceRate < 0.0 || inputs.acceptanceRate > 1.0) {
    throw new Error(
      `acceptanceRate must be between 0 and 1, got ${inputs.acceptanceRate}.`
    );
  }

  const portionSum = headFraction + bodyFraction + otherFraction;
  if (
    headFraction < 0.0 ||
    bodyFraction < 0.0 ||
    otherFraction < 0.0 ||
    Math.abs(portionSum - 1.0) > 1e-9
  ) {
    throw new Error(`cost portions must be non-negative and sum to 1.0.`);
  }

  if (bodyShare < 0.0 || bodyShare > 1.0) {
    throw new Error(
      `bodyShareOfNonHead must be between 0 and 1, got ${bodyShare}.`
    );
  }

  if (inputs.flashheadHeadSpeedup <= 0.0) {
    throw new Error(
      `flashheadHeadSpeedup must be positive, got ${inputs.flashheadHeadSpeedup}.`
    );
  }

  if (
    inputs.flashheadAcceptanceMultiplier < 0.0 ||
    inputs.flashheadAcceptanceMultiplier > 1.0
  ) {
    throw new Error(
      `flashheadAcceptanceMultiplier must be between 0 and 1, got ${inputs.flashheadAcceptanceMultiplier}.`
    );
  }

  if (inputs.draftBodyMultiplier < 0.0) {
    throw new Error(
      `draftBodyMultiplier must be non-negative, got ${inputs.draftBodyMultiplier}.`
    );
  }

}

function clampProbability(value: number) {
  return Math.min(Math.max(value, 0.0), 1.0);
}

function normalizePortionFraction(value: number, name: string) {
  if (value >= 0.0 && value <= 1.0) {
    return value;
  }

  if (value > 1.0 && value <= 100.0) {
    return value / 100.0;
  }

  throw new Error(
    `${name} must be a fraction between 0 and 1 or a percentage between 0 and 100.`
  );
}
