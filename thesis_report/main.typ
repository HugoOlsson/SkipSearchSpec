#import "chalmers-cover2.typ": cover-pages

#set heading(numbering: "1.1.1")  // enable numbering first
#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#show raw: set text(font: "New Computer Modern Mono")
#import "functions.typ": *

#show link: set text(fill: blue)
#show link: underline
#show raw.where(block: false): set text(size: 1.1em)

#show raw.where(block: true): it => {
  block(
    fill: luma(250),         // Light grey background
    inset: 12pt,             // Padding around the code
    radius: 6pt,             // Rounded corners
    width: 100%,             // Stretch to page width
    stroke: 0.5pt + luma(200), // Subtle border
    it
  )
}
#show raw.where(block: true, lang: "python"): set text(size: 9pt)

#show heading.where(level: 1): it => {
  pagebreak(weak: true, to: "odd")           // openright: chapters on right pages
  v(2cm)
  align(center)[
    #if it.numbering != none {
      let n = counter(heading).at(it.location()).first()
      text(size: 50pt, weight: "regular")[#n]
      v(-0.4em)                              // matches \vspace{-4.2ex}
    }
    #text(size: 24.88pt, weight: "bold")[#it.body]
  ]
  v(1.5cm)
}

#show heading.where(level: 2): set text(size: 17.28pt, weight: "bold")
#show heading.where(level: 3): set text(size: 14.4pt,  weight: "bold")

#cover-pages(
  title: "Skip, Search, Speculate",
  subtitle: "Approximating the Head and Body to Raise Amdahl's Ceiling for LLM Inference",
  author: "Hugo Olsson",
  supervisor: "Matti Karpa, Data Science och AI",
  examiner: "Devdatt Dubhashi, Data Science och AI",
  cover-figure: image("my-figures/cover-image.jpg", width: 90%),
  // year defaults to datetime.today().year()
)

// ---------- ABSTRACT ----------
#page(header: none)[
  A Cryptographic Approach to Media Provenance \
  Transparent Append-Only Logs for Authentic Capture Verification \
  Hugo Example \
  Department of Computer Science and Engineering \
  Chalmers University of Technology 

  #v(0.5cm)

  #heading(level: 2, numbering: none, outlined: false)[Abstract]

  Placeholder:
  This thesis develops a system for verifying the authenticity of digital
  media at the point of capture. We present a hash-chained append-only
  log architecture with daily SQLite snapshots and external anchoring,
  and argue that it addresses several attacks that C2PA cannot.

  #v(1fr)

  Keywords: media authenticity, provenance, C2PA, transparency logs.
]

#page(header: none, footer: none, numbering: none)[]  // blank verso


// ---------- ACKNOWLEDGEMENTS ----------
#page(header: none)[
  #heading(level: 2, numbering: none, outlined: false)[Acknowledgements]

  Placeholder: I would like to thank my supervisors for their patient guidance through
  this project, my advisor at the industry partner for access to real
  capture devices, and my examiner for sharp feedback on the threat model.

  #v(1.5cm)

  #align(right)[Hugo Example, Gothenburg, April 2026]
]

#page(header: none, footer: none, numbering: none)[]  // blank verso


#page(header: none)[
  #v(1cm)
  #align(center)[
    #text(size: 24.88pt, weight: "bold")[Contents]
  ]
  #v(1.5cm)
  #outline(title: none, depth: 5, indent: 1.5em)
]


#set page(numbering: "1")
#counter(page).update(1)


= Introduction
Large language models (LLMs) have become popular since the release of ChatGPT @openai2022chatgpt and are used daily by hundreds of millions of people for personal and professional tasks [SOURCE]. LLMs are expensive to run. They require a lot of memory and compute @kwon2023pagedattention @dettmers2023qlora. It is of interest to make them efficient so that the best possible model can run on available hardware and energy resources. LLMs use an architecture called the _transformer_ which can be understood as having a body and a head. A recent paper called FlashHead @flashhead2026 proposed a solution to speed up the head using approximate nearest neighbors (ANN). FlashHeads seems promising but a key insight is that the head is usually 1-20% of the total compute to produce the next token. This means that even if the head becomes _zero_ in computational cost, the total speedup cannot exceed 1.01x-1.25x since the compute of the body is still there and becomes the bottleneck. This thesis uses this Amdahl's law framing to investigate how _both_ the head and the body can be approximated to increase the ceiling of possible speedup. The investigated techniques are FlashHead to speedup the head, and _skipping layers_ to speedup the body. This is done with the intention of a drop-in solution, meaning that the LLM itself shouldn't need retraining to apply the upgrades. While this idea could be used for any LLM, this thesis focuses on smaller LLMs with 0.5B-10B parameters due to computational constraints.

== Background and theory

The architecture of a transformer model can split into a body and a head. The body is composed of multiple layers, where each layer is one Attention block and one MLP block. The body produces an internal state#footnote[The internal state can be thought of as the transformer's "understanding" at that token position per layer. So there are internal states for each token position and after each layer.] per token position, that is updated layer by layer. The internal state per token position is represented as a high-dimensional vector called the _hidden vector_.  After all layers the hidden vector at the last token position is transformed into the next token the transformer will generate. The transformation from hidden vector to next token is performed in the head. Tokens are the atoms the transformer operates with, it doesn't generate character by character and it also doesn't necessarily generate entire words at a time. Tokens are learned#footnote[Finding good tokens is usually an optimization problem performed separately before training of the LLM parameters.] good sub-strings to represent the domain of text the LLM is trained for. In practice, those sub-strings are often something like "hello", "Chal", "mers", "apple", "opti", "miz", "ing" etc. All these tokens together make up what is called the transformers _vocabulary_.

The head has a large matrix called the _unembedding matrix_. This matrix is like a stack of vectors, one for each token in the vocabulary. The head projects the produced hidden vector for the last position to this unembedding matrix to get a dot product score with each token-unembedding-vector inside it. Tokens that get a high match with the hidden vector are likely to be sampled as the next token to generate. The selection can also be so simple that we just generate the token that has the highest match with the hidden vector – this is called greedy sampling.

As mentioned, the head is usually 1-20% of the total compute to produce a next token, depending on how big the head is compared to the body. Smaller models are closer to the upper bound whereas larger models is closer to the lower bound due to how the head/body ratio typically grows with parameter count.

The paper FlashHead proposes a technique to approximate the head projection with approximate nearest neighbors (ANN). This allows the transformer to find the next token to generate without calculating a score for all tokens in the vocabulary. However, the key framing in this thesis is that even though you make the head _zero_ in computational cost, the total speedup is bounded by the fraction of total compute the head represents. This is the Amdahl's law perspective that motivates this thesis. So if the head is 20% of the total compute to generate next token, the total speedup by approximating the head cannot exceed $1/0.8 = 1.25$.

The idea is then to improve the speed of both the head and the body to increase the upper bound of possible speedup. The technique investigated to speedup the body is _skipping layers_#footnote[The common way to skip layers is by exiting early, then you pull the hidden vector early and skip the last layers. Here _skipping layers_ is used more generally since it can mean to skip the first layers, a gap in the middle, every other layer, the last ones etc.]. The hypothesis is that, during generation, there are easy and hard tokens, and for easy tokens, maybe not all layers of processing will be needed. Some of the layers can then be skipped without hurting the quality of generation significantly. This would inherently reduce the computation needed in the body since fewer layers processed means less matrix-multiplications and calculations to perform.

A key constraint in this project is that these techniques should be applied as drop-in or close to drop-in. This means that skipping layers is evaluated from the perspective of not retraining the model to handle this. In practice, allowing retraining of the model often allows for better behavior, but it also raises the bar for needed training-compute and it loses the guarantee that the model will behave the same as original. Improving the efficiency of both the head and the body becomes even more important when the model grows in size since the head shrinks in portion of total compute, making the Amdahl ceiling even lower.

When skipping layers, there are primarily three different cases: *early-exit*, *gap-jump* or *late-start*. Early-exit refers to taking the hidden vector after a certain number of layers and then using it directly in the head, essentially exiting the body early. Gap-jump refers to skipping a number of layers in the middle so the hidden vector is taken after a layer and is given back to a later layer, not directly to the head. Late-start is the opposite of early-exit, it is when the token embedding vector skips a number of initial layers and then is processed by a number of layers before the head. There are theoretically also other variations where you skip every other layer or preserve a custom set of the total layers. These will be focused less on but are still included in some measurements. The report uses the notation (N, M) to represent a section of skipped layers. With this notation, (1,1) would mean that all layers except the first and the last are skipped. (2,2) would means that all but the first two and last two are skipped. The notation for an early-exit is then (N, 0) and (0, N) for late-start. 

*Hidden vector casting*

When skipping layers, it's not obvious that the hidden vector can be efficiently interpreted where it lands. For the easy tokens, even if the hidden vector has semantically converged early, it's not necessarily the case that the same semantics is represented geometrically equal in later layers. This means that the hidden vector could semantically converge quickly for easy tokens, aligning with the hypothesis, while at the same time not producing correct tokens when moved directly to the head or later layers. Essentially that semantic convergence does not mean geometric convergence for the hidden vector. This will likely break the ability to skip layers by just picking the hidden vector and putting it back in a more downstream position. To address this challenge, this thesis uses a technique it calls hidden vector casting (HVC). The idea is that a small learned transformation can translate geometry from where it is taken to where it is placed, essentially casting it to the correct geometric representation corresponding to its semantic meaning. In practice, this HVC can be implemented in multiple ways but a first good method to evaluate is a linear transformation.

*Speculative decoding*

The proposed methods to speedup the head and body use approximations of the full calculations. This can save compute but will also make generation quality worse, it's just a question about how much worse. When selecting an inference setup, it is usually not satisfying to apply inference speedups without clear information how the generation quality is affected. If the speedup is 30% but the generation quality has dropped with 30% then it's not necessarily a good deal. To ensure that the generation quality is the same as the original model, this project uses speculative decoding. The idea behind speculative decoding is that it's much faster to verify tokens than it is to produce them. The setup is that you have a _drafter_ and a _verifier_ model. The drafter generates a block of tokens and the verifier verifies if the tokens are the same as the verifier would have generated. If yes, then the tokens are accepted, and if not, then they are rejected from the point where the the verifier disagrees. If the drafter is fast and fairly accurate, this setup can be more performant than running the model normally while also not compromising the quality of generation. For this thesis, since the model parameters are unchanged, the verifier and drafter can be the same model, just running with different inference logic during drafting and verification. This is called _self-speculation_. The advantage with this is that only one model needs to be loaded into GPU memory.

The expected speedup from self-speculative decoding can be estimated from the draft block size $gamma$, the measured draft-token acceptance rate $a$, the verifier cost, and the drafter cost.

The acceptance rate in this thesis is defined as

$
a = frac("total_accepted_draft_tokens", "total_drafted_tokens").
$

Let $T_"normal"$ be the time for one normal autoregressive generation step. Let $T_"verifier"$ be the time for one verifier call in self-speculative decoding, and let $T_"drafter"$ be the time for one drafter step.

Each speculative round drafts $gamma$ tokens. The expected number of accepted draft tokens per round is then

$
gamma a.
$

From the way a transformer works, each verifier call produces an extra token after the last token the verifier processed. This is called a bonus token. Therefore, the expected number of output tokens per speculative round is

$
EE["tokens per round"] = 1 + gamma a.
$

The time cost of one speculative round is one verifier call plus $gamma$ drafter steps:

$
EE["time per round"] = T_"verifier" + gamma T_"drafter".
$

Normal autoregressive decoding would need one normal model call per output token. Therefore, producing the same expected number of tokens normally would take

$
EE["normal time for same tokens"] = (1 + gamma a) T_"normal".
$

The expected speedup is normal time divided by self-speculative time:

$
S =
frac((1 + gamma a) T_"normal", T_"verifier" + gamma T_"drafter").
$

Now define the measured relative costs

$
v = frac(T_"verifier", T_"normal")
$

and

$
d = frac(T_"drafter", T_"normal").
$

Substituting $T_"verifier" = v T_"normal"$ and $T_"drafter" = d T_"normal"$ gives

$
S =
frac((1 + gamma a) T_"normal", v T_"normal" + gamma d T_"normal").
$

Factoring out $T_"normal"$ from the denominator gives

$
S =
frac((1 + gamma a) T_"normal", (v + gamma d) T_"normal").
$

The $T_"normal"$ terms cancel, leaving

#[
  #set math.equation(numbering: _ => "(speedup equation)")
  $
  S = frac(1 + gamma a, v + gamma d).
  $ <selfs-speedup>
]

Using this form, the speedup can be estimated from how much compute the drafter and verifier use relative to normal inference. Since the verifier is the full model processing $gamma$ draft tokens in a single parallel forward pass, $v$ should be greater or equal to 1. Furthermore, since verifying $gamma$ tokens in parallel should be cheaper than generating them sequentially, $v$ is expected to be significantly less than the block size $gamma$. This is the core of why speculative decoding can be faster than normal generation.

The acceptance rate definition used in this thesis is not necessarily consistent with all other papers in the space, as the convention varies. Some papers use a per-position conditional rate, here called $a_c$, defined as the probability that a drafted token is accepted given that all prior tokens in the draft block were accepted. This means that acceptance rates cannot necessarily be compared without having this in mind. 

*Key-value cache*

#figure(
  image("my-figures/kv-cache-figure.jpg", width: 90%),
  caption: [Illustration of how KV-cache helps to not recalculate the entire prefix when generating next token.],
) <kv-cache-img>

During generation, each new token attends to all previously generated tokens. Without caching, this would require recomputing the same attention values for every prior token at every new step, making generation quadratically more expensive as the sequence grows. The key-value cache (KV-cache) avoids this by storing the intermediate attention computations for already processed tokens, so each new step only needs to compute attention for the newest token. This makes generation significantly faster and is standard in all practical LLM inference systems.

For speculative decoding, if the drafter and verifier are two different models, then they need a KV-cache each. This increases memory usage compared to normal generation which is a drawback. When doing self-speculation, the drafter and verifier can share KV-cache since they are fundamentally the same model just with different inference logic. 

// == Problem Statement
// The research question for this project is:

// #pad(left: 1em, right: 0em)[
// _To what extent can inference for small scale LLMs be sped up by using a setup where the draft model is made computationally cheaper in both body and head by using ablations of the techniques: skipping layers + HVC + ANNH + self-speculative decoding?_
// ]

// Following the Amdahl's law reasoning presented in the background, these enhancements could significantly improve the speed of the draft model. They will make its produced quality strictly worse, but since this setup uses a verifier, the output will still have a lower bound for the quality. It is not obvious that this will produce a solution that is better than simply running the model normally, or just with an ANN head. The quality of the drafter could become so weak that there is more harm than good to use this inference setup. In that case, the setup would add complexity without any performance gain. This provides natural baselines for evaluation: standard decoding and ANN-head-only decoding. 

== Research Questions

// The aim of this thesis is to investigate whether combining layer skipping, HVC, and ANNH in a self-speculative decoding setup can produce meaningful inference speedups for small-scale LLMs while keeping original generation quality or not increasing memory usage.

The following research questions are addressed:

+ Which layer-skipping strategy minimizes damage to generation quality per layer skipped, and does this pattern hold across model families?

+ Can a lightweight HVC bridge recover the generation quality lost from skipping layers well enough to produce an effective drafter?

+ To what extent can inference for LLMs be sped up by using a setup where the draft model is made computationally cheaper in both body and head by using the techniques: skipping layers + HVC + ANNH + self-speculative decoding?

Following the Amdahl's law reasoning presented in the background, these enhancements could significantly improve the speed of the draft model. They will make its produced quality strictly worse, but since this setup uses a verifier, the output will be lossless compared to the original model. It is not obvious that this will produce a solution that is better than simply running the model normally, or just with an ANN head. The quality of the drafter could become so weak that there is more harm than good to use this inference setup. In that case, the setup would add complexity without any performance gain. This provides natural baselines for evaluation: standard decoding and ANN-head-only decoding. 

== Scope and Limitations

=== Scope
+ To produce an inference implementation for layer skipping with HVC, allowing a model to skip a contiguous block of internal layers.
+ To produce a training setup for the HVC  to match inference objectives.
+ To produce an ANNH implementation that uses the method described in the FlashHead paper.
+ To produce a self-speculation setup that uses the original model as verifier and drafter but where skipping layers and ANNH is used in drafter-mode.
+ To produce a KV-cache implementation so that the verifier and drafter share the same KV-cache.
+ To build a testing setup that investigates the best layer-skipping choices.
+ To produce a benchmark that compares real speedup from self-speculation compared to normal generation with detailed diagnostics and profiling.


=== Limitations
This project has several limitations to keep the workload feasible:

+ Model scale: Only small-scale models are used (about 0.5B–10B parameters). This limitation makes building and testing quicker since the models need less resources and time to run. It also make sense because it is where FlashHead gives the biggest speedup and thus differentiation for this research.

+ Hardware testing: The project doesn't do comprehensive testing of performance on different chips and possible hardware. The benchmark GPUs are NVIDIA L4 and L40s. 
+ Model selection: This project limits itself to a small number of open models from Llama 3.1, Llama 3.2, Qwen 2.5, Qwen 3 and Mistral.


= Methodology

The overall methodology for the project is to implement the inference setups in PyTorch. Different configurations with different hyper parameters are measured how well they perform for their targeted task. Metrics like cross-entropy, KL divergence, Top1 match and acceptance rate is used to measure performance. The project is availible as an open source repository at #link("https://github.com/HugoOlsson/SkipSearchSpec")[SkipSearchSpec]. All measurements presented in this thesis uses the same measurement-pipeline so that they always follow a shared structure, includes their git commit/tag and stores the raw data in the repository. By including the exact commit for each result, the reader can always visit the exact state of the project for when a measurement was performed.

The methods to increase the speed of the body and head are largely separable. The job of the body is to deliver well converged hidden vectors and the job of the head is to find matching tokens given a hidden vector. The project therefore has several experiments that isolates the task of skipping layers in the body, and then other experiments that isolates the task of speeding up the head with ANNH. The best found solutions for each part are then used to produce a drafter in a self-speculative setup where the acceptance rate, correctness and speedup is measured.

== Metrics

The end goal is to make the drafter performant and close in generation to the verifier. There are different metrics to measure this behavior, the two most important ones are KL divergence and top1 match. Top1 match means the fraction of generations where the drafter and verifier produces the same highest probability token. Which one is more important depends on the acceptance policy the speculative decoding will use. If the speculative decoding uses greedy sampling, then mostly top1 matching matters, if it uses sampling with temperature then both KL and and top1 will be of importance.

== Environment
The project is written with Python, PyTorch and Hugging Face. It uses open models from Llama 3.2, Llama 3.1, Mistral-7B-v0.3 and SmolLM2-1.7B. The models are selected to work with standard completion-style prompting instead of chat-style. The models are run using the provided official forward function for the model. To skip layers, targeted layers are overwritten with a hook that makes them into a no-operation, to just pass the hidden vector forward like it is. 

#```python
class NoOpDecoderLayer(nn.Module):
    """
    Cheap replacement for a HF decoder layer.

    Returns a tuple, not a Tensor, because Qwen/LLaMA-style
    decoder loops expect layer_outputs[0].
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:

        return (hidden_states,)```

== Datasets
The dataset used to train on is 18% "HuggingFaceTB/cosmopedia-100k", 18% "codelion/fineweb-edu-1B", 41%  "MBZUAI/LaMini-instruction", 18% "flytech/python-codes-25k", and 5% "roneneldan/TinyStories".  All training are made with a single epoch. This is to maximize the amount of examples seen given allocated compute, but also to get KL/CE/top1 training graphs that don't include progress where the module has seen the data before. 

== Body Approximation

// Multiple setups are used to find the solution that gives the best result:

// - Naive early-exits (without HVC)
// - Early-exits with HVC
// - How different ablations of skipping layers (early-exit, gap-jump, late-start, etc.) affect KL and top-1 per removed layer
// - How the best skipping-ablations perform with HVC

#figure(
  image("my-figures/gap-skip-setup.jpg", width: 100%),
  caption: [Structure of skipping layers.],
) <skipping-layers-structure-img>

The @skipping-layers-structure-img illustrates the architecture when skipping layers. The figure illustrates the case of a gap-jump. If the variant is early-exit then the hidden vector goes directly into the final norm. If the case is late-start, then the hidden vector goes from the embedding to the first layer and then progresses from there. Naive early exit is achieved by turning off the HVC. 



=== Finding best layer-skipping ablations

To get best possible drafting performance per layer skipped, it is likely important to skip the right layers for the model. This can be late-start, gap-jump or early-exit and different placements of those. To test what ablations of skipped layers that do the least amount of damage to the generation quality, a setup is used to test the KL and Top1 deviation from the full model for different skip-ablations. 

The setup exists in `evaluate_layer_skip_ablations.py`. It begins with running next-token prediction with the full model over a set of windows from the dataset, it records the logits produced at every position. Different layer-skip-ablations of the model are then run over the same windows and the next-token predictions are compared to the full model. This produces an average KL divergence compared to the full model and an average top1 score for each ablation. This is then presented in a plot that can show the results of KL, KL per removed layer, or top1 for all ablations. 

This setup does not use the HVC when skipping layers. Even though the HVC should be relatively lightweight to train, training one for all possible ablations would require a lot of compute. The skip-ablations are therefore measured with the hidden vector going directly from the last layer before the gap to entry layer. The idea is that this will still show what skip-ablations that are promising starting points to then improve further with the HVC.

A delimitation for this project is that only a single HVC will be used. This then requires there to be a single contiguous gap, not multiple holes of skipped layers. The ablations tested are mostly such with a contiguous gap, but some non-contiguous ablations are also included to see how they perform in this test where the HVC doesn't need to be added. Here is a specification of the ablations that are used:

#block(
  fill: luma(250),
  stroke: 0.5pt + luma(150),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  #set text(size: 10pt)

  #strong[Layer ablation masks tested.] \
  Let the model have $L$ decoder layers indexed $0, ..., L - 1$. \
  Each ablation is represented by a kept-layer set $A subset.eq {0, ..., L - 1}$; \
  all layers not in $A$ are skipped.

  #line(length: 100%, stroke: 0.5pt + luma(200))

  #grid(
    columns: (8em, 1fr),
    column-gutter: 1em,
    row-gutter: 8pt,

    [#strong[Keep all]],
    [Keep every layer: $A = {0, ..., L - 1}$. This is the no-ablation baseline.],

    [#strong[Early exit]],
    [Keep only a prefix of the network: $A = {0, ..., k - 1}$. \
    This tests how well the model performs when computation stops after the first $k$ layers.],

    [#strong[Late start]],
    [Keep only a suffix of the network: $A = {L - k, ..., L - 1}$. \
    This tests whether the later layers can operate without the earlier layers.],

    [#strong[Internal gap]],
    [Remove one contiguous block of internal layers while keeping layers before and after it: \
    $A = {0, ..., s - 1} union {s + g, ..., L - 1}$. \
    The skipped gap has start $s$ and length $g$, and does not touch the first or last layer.],

    [#strong[Periodic drop]],
    [Drop one phase modulo a step size $p$: \
    $A = {i : i mod p != phi}$. \
    This removes every $p$-th layer for several steps and phases.],

    [#strong[Periodic keep]],
    [Keep only one phase modulo a step size $p$: \
    $A = {i : i mod p = phi}$. \
    This keeps every $p$-th layer and skips the rest.],
  )
]


=== HVC setup
Internally in the code, the HVC is sometimes called _bridge_ due to the inherent mechanism of connecting two distant points. The text might refer to it as HVC or bridge.
#figure(
  image("my-figures/finalvsreentry.jpg", width: 100%),
  caption: [The input to the HVC bridge. It gets a stacked vector of the final hidden vector from token position t-1 and the hidden vector from the last layer before the gap at position t.],
) <finalvsreentry-img>

The bridge is implemented as a linear transformation in PyTorch with residual update and layer normalizations for the two input vectors. It takes a the hidden vector from the last layer before the gap and a hidden vector from the previous position t-1. As the @finalvsreentry-img shows, the HVC bridge gets the hidden vector from the last layer before the gap and the final hidden vector from position t-1. In the code `prev_reference_hidden` is a tensor with previous position hidden vectors. The code to forward the bridge is this:


#```python
    def forward(
        self,
        gap_hidden: torch.Tensor,
        prev_reference_hidden: torch.Tensor,
    ) -> torch.Tensor:

        bridge_dtype = self.proj.weight.dtype
        x = gap_hidden.to(dtype=bridge_dtype)
        p = prev_reference_hidden.to(dtype=bridge_dtype)

        x_n = self.gap_norm(x)
        p_n = self.prev_norm(p)

        delta = self.proj(torch.cat([x_n, p_n], dim=-1))
        return x + delta
        ```


=== Training skipping layers
The file `train_skipping_layers.py` exposes the function `def train_skipping_layers` which is used to train the HVC-bridge for an ablation of layer skipping. This function uses infrastructure to load datasets, build windows, load bridge module class, setup model and bridge as frozen and non-frozen respectively. It uses the full model in its standard inference mode to act teacher and the inference setup to skip layers as the student. The optimization objectives is to, for a window, minimize the KL divergence and CE compared to the teacher output for the tokens. 


#```python
def train_skipping_layers(
    *,
    model_name: str,
    dataset_mix: list[tuple[DatasetSpec, float, int]],
    context_len: int = 256,
    num_windows_to_use: int,
    batch_size: int = 10,
    active_start_layers: int,
    active_end_layers: int,
    num_epochs: int = 1, 
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = None,
    kl_loss_weight: float = 1.0,
    ce_loss_weight: float = 1.0,
    hidden_loss_weight: float = 0.0,
    teacher_temperature: float = 1.0,
    reference_hidden_source: ReferenceHiddenSource = "final",
    model_kwargs: dict[str, Any] | None = None,
    checkpoint_every_steps: int | None = None,
    log_every: int = 100,
    measurement_save_interval_seconds: float = 60.0,
    num_draft_sections: int = 5,
) -> TrainGapBridgeOutput:
        ```

#figure(
  image("my-figures/training_window.jpg", width: 105%),
  caption: [How a training window is structured. It is a window of `context_len` tokens divided into `num_draft_sections` sections. The vertical splitting lines indicate starting points where the student continues from how the teacher's KV-cache was at that position. At each position it is getting the teachers t-1 final hidden token, this is the shifted pink array, and the t position last layer hidden vector before the gap.],
) <training_window-img>


The training aims to produce a good drafter for the full model. When running in self-speculation, the drafter will run from where the verifier last stopped. It will do so by continuing from the KV-cache the verifier produced. The training objective is therefore to "cast" a hidden vector through the gap using the input hidden vectors, but to also do so when starting from the verifiers KV-cache. 

To train for this, the teacher runs next-token prediction on the training window and its logits for all positions and the created KV-cache are stored. The window is then conceptually split into multiple sections like @training_window-img shows. The student will do next-token prediction runs on the sections from one starting point to the next. At each boundary, it will start from the KV-cache the teacher has produced at that position. This simulates the objective to start from a verifier prefix and generate from there. 

If the intended block size for the self-speculation is 1-5, then dividing the window into sections of that size would make sense. However, that would be a lot of compute to produce so many versions of the teacher KV-cache history and to run so many small drafter trainings. Therefore, a number of sections that balances compute and realism will have to be selected. The parameter to set the number of sections is `num_draft_sections`. In @training_window-img `num_draft_sections = 5`.

Every HVC-training produces a run.json file that includes training and loss values for every Nth step. These will be used to then plot training convergence and to do analysis. Examples of such values are:

#figure(
  table(
    columns: (auto, 1fr),
    inset: 10pt,
    align: horizon,
    fill: (x, y) => if y == 0 { luma(230) },
    stroke: 0.5pt + luma(200),

    table.header(
      [*Metric*], [*Description / Logic*]
    ),

    [`kl_verifier_to_drafter`], [
      #set text(size: 9pt)
      The KL divergence between the verifier's next-token distribution and the drafter's next-token distribution. Lower values mean that the drafter assigns probability mass more similarly to the verifier over the vocabulary.
    ],

    [`top1_drafter_matches_verifier`], [
      #set text(size: 9pt)
      The fraction of token positions where the drafter and verifier have the same top1 token. This is especially important for greedy self-speculation, since a drafted token is accepted when it matches the verifier's selected token.
    ],

    [`loss_ce_drafter_on_verifier_top1`], [
      #set text(size: 9pt)
      The CE loss of the drafter using the verifier's top1 token as the target. This trains the drafter to put high probability on the token that the verifier would select at the same position.
    ],

    [`loss_bridge_reentry_mse`], [
      #set text(size: 9pt)
      The MSE loss between the hidden vector produced by the HVC bridge and the verifier's target hidden vector at the re-entry point after the skipped gap. Lower values mean that the bridge better casts the skipped hidden vector into the geometry expected by the later layers.
    ],
  ),
  caption: [Summary of HVC training metrics],
) <metric-summary-table>



== ANNH implementation

This project produces an implementation of FlashHead and measures accuracy, clustering times and speedups.

To evaluate ANNH as a replacement for a dense head, three parts are produced: 
- A program to build an ANNH cluster and save that in an index structure that is performant to use. 
- An accuracy-evaluator that can load a saved ANNH and measure how often it finds the correct token. 
- An adapter to run a model with the ANNH.

=== ANNH cluster builder

Clusters of token unembedding vectors are built using spherical k-means with a strict capacity constraint. After each iteration of k-means, vectors in overloaded clusters are greedily moved to clusters that still are a good match until capacity constraint is satisfied for all clusters.

The number of clusters is fixed by the `num_clusters` argument. Before clustering, the builder computes strictly equal cluster capacities whose sum is exactly the vocabulary size. Thus, only values of num_clusters that evenly divide the vocabulary size are accepted.

The algorithm that is used to produce the clustering is the following:

#let kw(it) = strong(it)
#let argmax = math.op("argmax")
#let alg-counter = counter("algorithm-line")

#let alg-num() = {
  alg-counter.step()
  context alg-counter.display()
}

#figure(
  align(left)[
#block(
  fill: luma(250),
  stroke: 0.5pt + luma(150),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  #set text(size: 10pt)
  
  #kw[Algorithm:] Strict Equal-Capacity LM-Head Clustering \
  #line(length: 100%, stroke: 0.5pt + luma(200))
  
  #kw[Input:] LM-head vector table $W in RR^(V times H)$, number of clusters $C$, \
  number of iterations $N$, normalization flag, random seed $S$ \
  #kw[Output:] cluster-to-token table $G$, centroids $mu$

  #alg-counter.update(0)

  #grid(
    columns: (2em, 1fr),
    column-gutter: 0.6em,
    row-gutter: 9pt,

    [#alg-num():], [Let $V$ be the vocabulary size and $H$ the hidden size],
    [#alg-num():], [Require $C >= 1$, $C <= V$, and $V mod C = 0$],
    [#alg-num():], [Set the strict cluster capacity $q <- V / C$],
    [#alg-num():], [#kw[if] normalization is enabled #kw[then] set $z_t <- W_t / ||W_t||_2$ for every token $t$],
    [#alg-num():], [#kw[else] set $z_t <- W_t$],
    [#alg-num():], [Randomly sample $C$ token indices using the seed $S$],
    [#alg-num():], [Initialize centroids $mu_1, ..., mu_C$ from the vectors of the sampled token indices],
    [#alg-num():], [#kw[if] normalization is enabled #kw[then] normalize all centroids],

    [#alg-num():], [#kw[for] iteration $r = 1, ..., N$ #kw[do]],
    [#alg-num():], [#h(1.5em)Compute similarity scores $s_(t,c) <- z_t dot mu_c$ for every token $t$ and cluster $c$],
    [#alg-num():], [#h(1.5em)Assign each token to its nearest centroid: $a_t <- argmax_c s_(t,c)$],
    [#alg-num():], [#h(1.5em)Compute current cluster sizes $n_c <- |{t : a_t = c}|$],
    [#alg-num():], [#h(1.5em)Set eviction list $E <- emptyset$],
    [#alg-num():], [#h(1.5em)#kw[for] each overloaded cluster $c$ with $n_c > q$ #kw[do]],
    [#alg-num():], [#h(3em)Compute regret $rho_t <- s_(t,c) - max_(c' != c) s_(t,c')$ for each token $t$ assigned to $c$],
    [#alg-num():], [#h(3em)Add the $n_c - q$ lowest-regret tokens from $c$ to $E$],
    [#alg-num():], [#h(1.5em)#kw[end for]],
    [#alg-num():], [#h(1.5em)Sort $E$ by increasing regret],
    [#alg-num():], [#h(1.5em)#kw[for] each token $t in E$ #kw[do]],
    [#alg-num():], [#h(3em)Move $t$ to the non-full cluster $c$ with best score $s_(t,c)$ and update sizes],
    [#alg-num():], [#h(1.5em)#kw[end for]],

    [#alg-num():], [#h(1.5em)Recompute each centroid as the mean of its assigned token vectors],
    [#alg-num():], [#h(1.5em)#kw[if] normalization is enabled #kw[then] normalize all centroids],
    [#alg-num():], [#h(1.5em)Compute loss $L <- 1 - frac(1, V) sum_t (z_t dot mu_"assigned for t")$ and report similarity statistics],
    [#alg-num():], [#kw[end for]],

    [#alg-num():], [Run one final strict equal-capacity assignment using the learned centroids],
    [#alg-num():], [Build $G$ by sorting token ids by assigned cluster and reshaping to shape $C times q$],

    [#alg-num():], [#kw[return] token map $a$, cluster table $G$, centroids $mu$, and cluster sizes],
  )
]   ],
  caption: "Algorithm to build FlashHead-like cluster of token vectors.",
  kind: "algorithm",
  supplement: [A],
) <alg:strict-equal-lm-head-clustering>


The algorithm @alg:strict-equal-lm-head-clustering builds an output of this Python structure:
#```python
class BuiltANNHClusters:
    cluster_to_token_ids: Tensor      # [num_clusters, cluster_size], no padding
    centroids: Tensor                 # [num_clusters, hidden_size]
```

// After clustering, that representation is processed to this new structure to be a performant index during inference:

// #```python
// @dataclass(frozen=True, slots=True)
// class FlashHeadIndex:
//     # [hidden_size, num_clusters], transposed centroids for fast cluster scoring
//     centroids_t: Tensor          
//     # [num_clusters, cluster_size], token ids contained in each cluster          
//     cluster_to_token_ids: Tensor     
//     # [num_clusters, cluster_size, hidden_size], LM-head rows grouped by cluster      
//     clustered_lm_head: Tensor           
//     # [num_clusters, cluster_size] or None, matching LM-head bias grouped by cluster   
//     clustered_lm_head_bias: Tensor | None  
//     # number of tokens in each cluster
//     cluster_size: int          
//     # total number of clusters           
//     num_clusters: int    
//     # total number of tokens in the vocabulary                 
//     vocab_size: int  ```

// The compilation from `BuiltFlashHeadClusters` to `FlashHeadIndex` is done with this algorithm:

// #figure(
//   align(left)[
// #block(
//   fill: luma(250),
//   stroke: 0.5pt + luma(150),
//   inset: 12pt,
//   radius: 4pt,
//   width: 100%,
// )[
//   #set text(size: 10pt)
  
//   #kw[Algorithm:] Build Fast FlashHead Index \
//   #line(length: 100%, stroke: 0.5pt + luma(200))
  
//   #kw[Input:] token-to-cluster map $a$, cluster-to-token table $G in NN^(C times q)$, \
//   centroids $mu in RR^(C times H)$, cluster sizes $n in NN^C$, \
//   original LM-head table $W in RR^(V times H)$, optional LM-head bias $b in RR^V$ \
  
//   #kw[where:] $C$ is the number of clusters, $q$ is the number of tokens per cluster, \
//   $V = C q$ is the vocabulary size, and $H$ is the hidden size \
  
//   #kw[Output:] fast index $(mu^T, G, W_"clustered", b_"clustered", q, C, V)$

//   #alg-counter.update(0)

//   #grid(
//     columns: (2em, 1fr),
//     column-gutter: 0.6em,
//     row-gutter: 9pt,

//     [#alg-num():], [Read the number of clusters $C$ and cluster size $q$ from $G in NN^(C times q)$],
//   [#alg-num():], [Transpose the centroids: $mu in RR^(C times H)$ becomes $mu^T <- "transpose"(mu) in RR^(H times C)$],
//   [#alg-num():], [Flatten the clustered token ids: $G in NN^(C times q)$ becomes $g <- "flatten"(G) in NN^(C q)$],
//   [#alg-num():], [Gather LM-head rows in clustered order: $W$ and $g$ give $W_g <- W[g] in RR^(C q times H)$],
//   [#alg-num():], [Reshape $W_g$ into clustered LM-head weights $W_"clustered" in RR^(C times q times H)$],
//   [#alg-num():], [#kw[if] bias $b in RR^V$ is provided #kw[then] gather $b[g] in RR^(C q)$ and reshape it into $b_"clustered" in RR^(C times q)$],
//   [#alg-num():], [#kw[else] set $b_"clustered" <- "None"$],
//   [#alg-num():], [Set $V <- C times q$],
//   [#alg-num():], [#kw[return] $mu^T$, $G$, $W_"clustered"$, $b_"clustered"$, $q$, $C$, and $V$],
//   )
// ]   ],
//   caption: "Algorithm to convert built token clusters into the fast FlashHead inference index.",
//   kind: "algorithm",
//   supplement: [A],
// ) <alg:build-fast-flashhead-index>


To use the built ANNH clusters during inference, a `ANNHModule` is constructed from the stored
`BuiltANNHClusters` and the model's existing LM-head tensors. The module stores the transposed centroids
and the cluster-to-token table, but it does not copy or pre-cluster the full LM-head weight table. Instead, the
original LM-head weight table is borrowed from the model and candidate token rows are gathered from it during
lookup.

The function `find_token(...)` takes a query final hidden vector and searches only the tokens contained in the top-scoring
clusters. Its algorithm is:


#figure(
  align(left)[
#block(
  fill: luma(250),
  stroke: 0.5pt + luma(150),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  #set text(size: 10pt)
  
  #kw[Algorithm:] ANNH Greedy Token Lookup \
  #line(length: 100%, stroke: 0.5pt + luma(200))
  
  #kw[Input:] query final hidden vector $h_q in RR^H$, transposed centroids $mu^T in RR^(H times C)$, \
  cluster-to-token table $G in NN^(C times q)$, LM-head vector table $W in RR^(V times H)$, \
  optional LM-head bias $b in RR^V$, number of clusters to search $K_c$ \

  #kw[where:] $C$ is the number of clusters, $q$ is the cluster size, \
  $V = C q$ is the vocabulary size, and $H$ is the hidden size \

  #kw[Output:] selected token id $y$

  #alg-counter.update(0)

  #grid(
    columns: (2em, 1fr),
    column-gutter: 0.6em,
    row-gutter: 9pt,

    [#alg-num():], [Set $K <- min(K_c, C)$],
    [#alg-num():], [Compute cluster scores $r <- h_q dot mu^T in RR^C$],
    [#alg-num():], [Select the $K$ clusters with largest scores; call their indices $P in NN^K$],
    [#alg-num():], [Gather candidate token ids from the selected clusters: $T <- "flatten"(G[P]) in NN^(K q)$],
    [#alg-num():], [Gather the corresponding LM-head rows from the original table: $W_T <- W[T] in RR^(K q times H)$],
    [#alg-num():], [Compute candidate token scores $ell <- W_T h_q in RR^(K q)$],
    [#alg-num():], [#kw[if] bias $b$ exists #kw[then] add $b[T] in RR^(K q)$ to $ell$],
    [#alg-num():], [Let $m <- max_i ell_i$ be the highest candidate score],
    [#alg-num():], [#kw[return] the smallest token id $T_i$ such that $ell_i = m$],
  )
]   ],
  caption: "Algorithm for approximate greedy best matching token lookup using built FlashHead clusters.",
  kind: "algorithm",
  supplement: [A],
) <alg:flashhead-find-token>

== Inference

The best skip-ablations are used for the body and the ANNH implementation is used for the head. Using those together composes a drafter model for a self-speculative decoding setup.


=== Self-speculative decoding

Self-speculative decoding is used to measure whether an approximated model act as a useful drafter while the original full model acts as the verifier. In this setup, both the drafter and verifier come from the same base model. The verifier uses the original model, while the drafter uses the same model with selected layers skipped and optionally ANNH.

The implementation uses greedy decoding. This means that both the drafter and verifier always select the token with highest probability. A drafted token is accepted if it is exactly equal to the verifier's greedy token at the same position. 

Here is a pseudocode of how the self-speculative decoding works:

#let kw(it) = strong(it)
#let argmax = math.op("argmax")
#let alg-counter = counter("algorithm-line")

#let alg-num() = {
  alg-counter.step()
  context alg-counter.display()
}

#figure(
  align(left)[
#block(
  fill: luma(250),
  stroke: 0.5pt + luma(150),
  inset: 12pt,
  radius: 4pt,
  width: 100%,
)[
  #set text(size: 10pt)
  
  #kw[Algorithm:] Self-Speculative Decoding with an Approximated Drafter \
  #line(length: 100%, stroke: 0.5pt + luma(200))
  
  #kw[Input:] prompt $x$, maximum generation length $T$, draft block size $K$, \
  full model $M$, drafter $tilde(M)$ formed by skipping a layer gap in $M$ and inserting bridge $B$ \
  #kw[Output:] generated sequence $y$

  #alg-counter.update(0)

  #grid(
    columns: (2em, 1fr),
    column-gutter: 0.6em,
    row-gutter: 9pt,

    [#alg-num():], [$y <- x$],
    [#alg-num():], [Run verifier $M$ on $y$],
    [#alg-num():], [Store verifier KV cache $C$],
    [#alg-num():], [Store verifier reference hidden states $H$],
    [#alg-num():], [Append the initial verifier bonus token to $y$],
    [#alg-num():], [#kw[while] number of generated tokens $< T$ #kw[do]],
    [#alg-num():], [#h(1.5em)$d <-$ empty list],
    [#alg-num():], [#h(1.5em)Save verifier cache state $C_0 <- C$],
    [#alg-num():], [#h(1.5em)#kw[for] $i = 1, ..., K$ #kw[do]],
    [#alg-num():], [#h(3em)Construct previous-reference hidden states from $H$],
    [#alg-num():], [#h(3em)Run drafter $tilde(M)$ on the current suffix using $C$; set $d_i <- argmax tilde(M)(C, H)$],
    [#alg-num():], [#h(3em)Append $d_i$ to $d$],
    [#alg-num():], [#h(3em)Temporarily extend $C$ with the drafter cache for the next draft token],
    [#alg-num():], [#h(1.5em)#kw[end for]],
    [#alg-num():], [#h(1.5em)Restore $C <- C_0$],
    [#alg-num():], [#h(1.5em)$y_"candidate" <- "concatenate"(y, d)$],
    [#alg-num():], [#h(1.5em)Run verifier $M$ on the unverified suffix of $y_"candidate"$ using cache $C$],
    [#alg-num():], [#h(1.5em)Let $v_1, ..., v_K$ be the verifier greedy predictions aligned with $d_1, ..., d_K$],
    [#alg-num():], [#h(1.5em)$m <-$ length of the longest prefix such that $d_j = v_j$ for all $j <= m$],
    [#alg-num():], [#h(1.5em)Append accepted draft tokens $d_1, ..., d_m$ to $y$],
    [#alg-num():], [#h(1.5em)#kw[if] $m < K$ #kw[then]],
    [#alg-num():], [#h(3em)Append verifier correction token $v_(m+1)$ to $y$],
    [#alg-num():], [#h(3em)Crop verifier KV cache and reference hidden states to the accepted prefix],
    [#alg-num():], [#h(1.5em)#kw[else]],
    [#alg-num():], [#h(3em)Append verifier bonus token predicted after the full draft block to $y$],
    [#alg-num():], [#h(3em)Keep the full verifier KV cache and reference hidden states],
    [#alg-num():], [#h(1.5em)#kw[end if]],
    [#alg-num():], [#kw[end while]],
    [#alg-num():], [#kw[return] $y$],
  )
]   ],
  caption: "",
  kind: "algorithm",
  supplement: [A],
) <alg:self-spec>

As shown in @finalvsreentry-img, the HVC bridge takes the hidden vector from the previous layer but also a hidden vector from the previous position t-1. During speculation when the draft block has a size of more than 1, the hidden vector at draft step 1 is from the verifier, but from step 2 and forward, it is from the drafter itself.

The real implementation records the generated text, the output token ids, the number of verifier calls, the number of drafted tokens and the number of accepted draft tokens. 

It can optionally also store a token-level trace JSON file for visualization. Each token is marked as either a prompt token, a drafted token or a verifier-produced bonus/correction token. Drafted tokens are additionally marked as accepted or rejected. This makes it possible to inspect where the drafter follows the verifier and where the verifier has to correct the generation.

==== KV-cache

A single KV-cache $C$ is used by both the verifier and the drafter, see @alg:self-spec. The drafter will start where the verifier left off and manipulate $C$ in place. Just before a draft block is started, the length of $C$ is stored as $C_0$. After the drafter has processed a draft block, $C$ will be cropped back to $C_0$ so that then when the verifier runs, it will start from the prefix of $C$ that is guaranteed to be correct. Then when it verifies the proposed draft block, it will update $C$ with the KV-cache up until the mismatch if there is any or for the entire draft block if it is accepted.

This approach gives that only a single KV-cache needs to be stored instead of having one for the verifier and one for the drafter. The KV-cache is mutated in place with cropping. This avoids having memory spikes that would be created if the rollback position $C_0$ was a copy. The KV-cache memory pressure is therefore not higher than running the model normally.

== Benchmarking

The benchmark measures total speedup compared to the normal generation baseline and captures internal details. The self-speculative generation is run in these two variants:

+ _Skipped layers_: the drafter uses the HVC-bridge and skipped-layer body, but still uses the dense LM-head.
+ _Skipped layers + ANNH_: the same drafter body is used, but token selection in the drafter uses the ANNH head instead of the dense LM-head.


=== Prompt sets

The benchmark uses two completion-style prompt sets: a concrete set and a general set. The concrete set contains tasks with relatively unambiguous answers. 

The general set is intended to complement this with prompts that are more open ended and possibly harded to speculate future tokens for the drafter, not because the drafter necessarly is wrong but because it might not be exactly what the verifier would generate.

A short example from the concrete prompt set is:




#```python
[
  ...
  (
        "concrete_math_total_price",
        (
            "Task: Compute the total cost.\n"
            "A notebook costs 4 dollars. A pen costs 2 dollars. Buy 3 notebooks "
            "and 5 pens.\n"
            "Return only the total number of dollars.\n\n"
            "Answer:\n"
        ),
    ),
    ...
    (
        "concrete_extract_email",
        (
            "Task: Extract the email address.\n"
            "Text: Please send the invoice to billing@example.com before Friday.\n"
            "Return only the email address.\n\n"
            "Output:\n"
        ),
    ),
    ...
```

=== Benchmark phases

Each benchmark variant has three phases:

+ _Warmup phase_: the first `N` prompts are run before any reported measurement, where `N` is a benchmark setting. These runs are discarded. Their purpose is to remove any bias for where initial prompts might be processed slower.
+ _Profile phase_: the next `M` prompts are run with internal timings enabled, where `M` is a benchmark setting. These results are saved separately and are used for diagnostic quantities such as verifier cost, drafter cost, drafter body/head/overhead split, ANNH head speedup, and verifier-to-normal ratios. They are not used for the total speedup numbers because internal profiling inserts device synchronizations around sub-operations which can affect total performance.
+ _Speed phase_: the full selected prompt set is run with internal timings disabled. These are the measurements used for acceptance rate, exact-match rate, memory usage, per-prompt speedup histograms, and total per-token speedup. The speed phase begins from the first prompt in the prompt set.


=== Timing measurements

All reported speedups use the speed phase. For both normal generation and self-speculation, timing starts after the prompt has been tokenized and moved to the device, and stops before decoding the output ids back to text. The timer is synchronized with the device at the start and end of the measured region. Normal generation is measured around a greedy `model.generate(...)` call with KV-cache enabled, so the normal time includes both prompt processing and generated-token decoding.

The self-speculative total time is measured around the full self-speculative generation call. This includes the initial verifier pass over the prompt, the first verifier-produced token, all drafter blocks, all verifier calls, token acceptance logic, KV-cache cropping/adoption, and other Python overhead inside the generation loop. In the speed phase, no internal body/head/verifier timers are enabled.

The profile phase uses the same self-speculative generation code, but enables internal timers. The profile records verifier time for verifier calls inside the speculative loop, drafter total time for each drafted block, drafter body time for the skipped-layer backbone forward pass, and drafter head time for either the dense LM-head or the ANNH lookup. The initial verifier prompt pass is included in total self-speculative time but not in the internal `verifier_seconds` bucket. This is why verifier-call profile ratios are computed from verifier calls after the prompt prefill.

Peak GPU memory is measured per prompt by resetting CUDA peak allocation statistics before each normal or self-speculative run and reading the peak allocation after the run. In the speed phase, normal generation results are cached by prompt index so the same normal baseline is reused when both self-speculative variants are compared.



    
= Results


== Skip ablations
To know what layers are the best to skip, different skip-ablations was evaluated. The plot shows different skip ablations in a format like this "`███████··███████`" which would mean that two middle layers are skipped and all other are activated, so a gap-jump. This "`█████████████···`" would mean that we skip the last 3 layers, and thus an early-exit, and so on. By using these, different skip ablations can be presented in a way that gives an intuitive visual overview how they compare. Two different measurements are used to compare to the full model with all layers activated: top1, KL full model to skip ablation and CE gap on dataset tokens. The dataset used to measure the performance of the ablations is `codelion/fineweb-edu-1B`.

*Results for  Mistral-7B-Instruct-v0.3:*


#figure(
  image("my-figures/plots/skip-ablations/layer_ablations_mistralai_Mistral-7B-Instruct-v0.3_20260506_222825_mean_ce_gap_multicolumn.png", width: 100%),
  caption: [Mean next-token cross-entropy increase on dataset tokens relative to the full model for different skip ablations of `mistralai/Mistral-7B-Instruct-v0.3`],
) <mistral-klp-skip-ablations-img>

#figure(
  image("my-figures/plots/skip-ablations/layer_ablations_mistralai_Mistral-7B-Instruct-v0.3_20260506_222825_mean_kl_full_to_masked_multicolumn.png", width: 100%),
  caption: [KL divergence from full model distribution to skipped ablation distribution for different skip-ablations of `mistralai/Mistral-7B-Instruct-v0.3`],
) <mistral-klp-skip-ablations-img>


#figure(
  image("my-figures/plots/skip-ablations/layer_ablations_mistralai_Mistral-7B-Instruct-v0.3_20260506_222825_mean_top1_agreement_multicolumn.png", width: 100%),
  caption: [Mean Top1 agreement compared to full model for different skip-ablations of `mistralai/Mistral-7B-Instruct-v0.3`],
) <mistral-top1-skip-ablations-img>


*Results for  Llama-3.2-1B-Instruct:*

#figure(
  image("my-figures/plots/skip-ablations/layer_ablations_meta-llama_Llama-3.2-1B_20260506_214633_mean_ce_gap_multicolumn.png", width: 100%),
  caption: [Mean next-token cross-entropy increase on dataset tokens relative to the full model for different skip ablations of `mistralai/Mistral-7B-Instruct-v0.3`],
) <mistral-klp-skip-ablations-img>

#figure(
  image("my-figures/plots/skip-ablations/layer_ablations_meta-llama_Llama-3.2-1B_20260506_214633_mean_kl_full_to_masked_multicolumn.png", width: 100%),
  caption: [KL divergence from full model distribution to skipped ablation distribution for different skip-ablations of `mistralai/Mistral-7B-Instruct-v0.3`],
) <mistral-klp-skip-ablations-img>


#figure(
  image("my-figures/plots/skip-ablations/layer_ablations_meta-llama_Llama-3.2-1B_20260506_214633_mean_top1_agreement_multicolumn.png", width: 100%),
  caption: [Mean Top1 agreement compared to full model for different skip-ablations of `mistralai/Mistral-7B-Instruct-v0.3`],
) <mistral-top1-skip-ablations-img>

The results show a clear pattern that skipping a gap in the middle seems to do less damage to the generation quality. By interpreting these graphs it's clear that it should be advantageous to utilize a gap-jump in the middle instead of early-exit or late-start. They also show that patterns that have multiple holes instead of a contiguous gap don't seem to perform better than a gap in the middle. These plots show the results for Llama-3.2-1B-Instruct and Mistral-7B-Instruct-v0.3, but from testing the same pattern appears in other models such as Llama-3.2-3B-Instruct, Qwen2.5 series, and meta-llama/Llama-3.1-8B.

== FlashHead cluster building and evaluation

Here are results regarding building a cluster presented. The most relevant examples are for models that have a large head-to-body ratio, because those are where using FlashHead instead of the full dense LM-head can produce the biggest speedup.

*Results for Llama-3.2-1B-Instruct:*

#figure(
  image("my-figures/plots/clustering/flashhead_llama32_1b_5344c_mean_similarity.png", width: 100%),
  caption: [Mean assigned cosine similarity during FlashHead clustering of `meta-llama/Llama-3.2-1B-Instruct` LM-head vectors over clustering iterations.],
) <clustering-llama3.2-1B-instruct-img>

@clustering-llama3.2-1B-instruct-img shows the process of clustering 5344 clusters for the 128,256 token vectors of Llama-3.2-1B-Instruct. It reaches a plateau after around 15 iterations. The total 40 iterations took around 102 seconds on an Apple M5 chip. The clustering produces these quality metrics:


#```python
cluster quality metrics:
num_clusters                     = 5344
mean_assigned_similarity         = 0.596406
p05_assigned_similarity          = 0.413671
mean_margin_to_best_other        = 0.133069
p05_margin_to_best_other         = -0.002699
fraction_assigned_to_nearest     = 0.947184
min_cluster_size                 = 24
max_cluster_size                 = 24
clustering_time                  = 102 seconds
```

The value `fraction_assigned_to_nearest` of 0.947 signals that most of the vectors belong to a cluster that has a centroid that is the nearest. The `min_cluster_size` and `max_cluster_size` also show that all clusters have the same size and the correct size for 5344 clusters with 128,256 unembedding token vectors.


By running evaluation on the cluster with different number of top-k probings these are the results: 

// #figure(
//   image("my-figures/plots/clustering/flashhead_llama32_1b_5344c_topk_sweep_top1_match_rate.png", width: 105%),
//     caption: [FlashHead evaluation sweep for the cluster built in @clustering-llama3.2-1B-instruct-img, showing top-1 match rate and top-3 containment against the dense LM head as the number of probed clusters is varied. Per top-k step, 30 windows each with 200 tokens from the dataset `HuggingFaceTB/cosmopedia-100k`is used.],
// ) <evaluation-sweep-cluster-llama3.2-1B-instruct-img>



#probing-table(
  title: [Top-1 and Top-3 Match Rate by Probed Clusters],
  model-name: [Llama 3.2 1B Instruct],
  commit-id: "2f4373e",
  windows: 30,
  window-length: 200,
  total-clusters: 5344,
  rows: (
    ([1],    [0.02%],  [0.345394], [0.552429]),
    ([10],   [0.19%],  [0.687772], [0.873367]),
    ([20],   [0.37%],  [0.812228], [0.942211]),
    ([50],   [0.94%],  [0.915578], [0.983417]),
    ([100],  [1.87%],  [0.956449], [0.996147]),
    ([300],  [5.61%],  [0.991457], [0.999497]),
    ([500],  [9.36%],  [0.996315], [0.999832]),
    ([1000], [18.71%], [0.998325], [1.000000]),
  ),
) <evaluation-sweep-cluster-llama32-1B-instruct-table>

@evaluation-sweep-cluster-llama32-1B-instruct-table shows that the probability of getting the true top-1 converges to 1 when the number of top-k probed clusters increases. To find the true top-1 token more than 99% of the time a top-k of around 300 is needed which is 5.61% of all clusters probed. Top-3 containment means that given the top selected token from the cluster, is that within top-3 what the full LM-head would have selected. 

Here is the data for 2.6k, 8k, and 16k clusters:

*2672 clusters:*

#```python
cluster quality metrics:
num_clusters                     = 2672
mean_assigned_similarity         = 0.553953
p05_assigned_similarity          = 0.385145
mean_margin_to_best_other        = 0.119876
p05_margin_to_best_other         = -0.020749
fraction_assigned_to_nearest     = 0.936674
min_cluster_size                 = 48
max_cluster_size                 = 48
clustering_time                  = 57 seconds
```

// #figure(
//   image("my-figures/plots/clustering/flashhead_llama32_1b_2672c_topk_sweep_top1_match_rate.png", width: 105%),
//     caption: [FlashHead evaluation sweep for a 2.6k cluster for Llama 3.2 1B Instruct],
// ) <evaluation-sweep-cluster-2.6k-llama3.2-1B-instruct-img>

#probing-table(
  title: [Top-1 and Top-3 Match Rate by Probed Clusters],
  model-name: [Llama 3.2 1B Instruct],
  commit-id: "dc68e2f",
  windows: 30,
  window-length: 200,
  total-clusters: 2672,
  rows: (
    ([1],    [0.04%],  [0.327471], [0.557119]),
    ([10],   [0.37%],  [0.703685], [0.884255]),
    ([20],   [0.75%],  [0.818258], [0.948911]),
    ([50],   [1.87%],  [0.913400], [0.986935]),
    ([100],  [3.74%],  [0.955946], [0.996817]),
    ([300],  [11.23%], [0.990620], [0.999832]),
    ([500],  [18.71%], [0.995477], [1.000000]),
    ([1000], [37.43%], [0.998492], [1.000000]),
  ),
) <evaluation-sweep-cluster-llama32-1b-instruct-2672-table>


*8016 clusters:*


#```python
cluster quality metrics:
num_clusters                     = 8016
mean_assigned_similarity         = 0.621290
p05_assigned_similarity          = 0.430173
mean_margin_to_best_other        = 0.139982
p05_margin_to_best_other         = -0.005357
fraction_assigned_to_nearest     = 0.944938
min_cluster_size                 = 16
max_cluster_size                 = 16
clustering_time                  = 153 seconds
```

// #figure(
//   image("my-figures/plots/clustering/flashhead_llama32_1b_8016c_topk_sweep_top1_match_rate.png", width: 105%),
//     caption: [FlashHead evaluation sweep for a 8k cluster for Llama 3.2 1B Instruct],
// ) <evaluation-sweep-cluster-8k-llama3.2-1B-instruct-img>


#probing-table(
  title: [Top-1 and Top-3 Match Rate by Probed Clusters],
  model-name: [Llama 3.2 1B Instruct],
  commit-id: "e83192c",
  windows: 30,
  window-length: 200,
  total-clusters: 8016,
  rows: (
    ([1],    [0.01%],  [0.349916], [0.549581]),
    ([10],   [0.12%],  [0.736181], [0.889615]),
    ([20],   [0.25%],  [0.851089], [0.955276]),
    ([50],   [0.62%],  [0.938693], [0.987102]),
    ([100],  [1.25%],  [0.971859], [0.996985]),
    ([300],  [3.74%],  [0.992462], [0.999665]),
    ([500],  [6.24%],  [0.995477], [0.999832]),
    ([1000], [12.48%], [0.998157], [1.000000]),
  ),
) <evaluation-sweep-cluster-llama32-1b-instruct-8016-table>


*16032 clusters:*

#```python
cluster quality metrics:
num_clusters                     = 16032
mean_assigned_similarity         = 0.668838
p05_assigned_similarity          = 0.484650
mean_margin_to_best_other        = 0.155324
p05_margin_to_best_other         = -0.019119
fraction_assigned_to_nearest     = 0.936627
min_cluster_size                 = 8
max_cluster_size                 = 8
clustering_time                  = 309 seconds
```

// #figure(
//   image("my-figures/plots/clustering/flashhead_llama32_1b_16032c_topk_sweep_top1_match_rate.png", width: 105%),
//     caption: [FlashHead evaluation sweep for a 16k cluster for Llama 3.2 1B Instruct],
// ) <evaluation-sweep-cluster-16k-llama3.2-1B-instruct-img>


#probing-table(
  title: [Top-1 and Top-3 Match Rate by Probed Clusters],
  model-name: [Llama 3.2 1B Instruct],
  commit-id: "9551a4d",
  windows: 30,
  window-length: 200,
  total-clusters: 16032,
  rows: (
    ([1],    [0.01%], [0.312060], [0.531658]),
    ([10],   [0.06%], [0.758794], [0.904188]),
    ([20],   [0.12%], [0.861642], [0.957789]),
    ([50],   [0.31%], [0.943886], [0.990787]),
    ([100],  [0.62%], [0.975712], [0.997152]),
    ([300],  [1.87%], [0.992965], [1.000000]),
    ([500],  [3.12%], [0.995812], [1.000000]),
    ([1000], [6.24%], [0.998325], [1.000000]),
  ),
) <evaluation-sweep-cluster-llama32-1b-instruct-16032-table>



*Results for Llama-3.2-3B-Instruct:*

*For 8016 clusters:*

#```python
cluster quality metrics:
num_clusters                     = 8016
mean_assigned_similarity         = 0.621848
p05_assigned_similarity          = 0.430856
mean_margin_to_best_other        = 0.143018
p05_margin_to_best_other         = -0.008053
fraction_assigned_to_nearest     = 0.943270
min_cluster_size                 = 16
max_cluster_size                 = 16
clustering_time                  = 230 seconds
```

// #figure(
//   image("my-figures/plots/clustering/flashhead_llama32_3b_8016c_topk_sweep_top1_match_rate.png", width: 105%),
//     caption: [FlashHead evaluation sweep for a 8k cluster for Llama 3.2 3B Instruct.],
// ) <evaluation-sweep-cluster-8k-llama3.2-3B-instruct-img>

#probing-table(
  title: [Top-1 and Top-3 Match Rate by Probed Clusters],
  model-name: [Llama 3.2 3B Instruct],
  commit-id: "1673707",
  windows: 30,
  window-length: 200,
  total-clusters: 8016,
  rows: (
    ([1],    [0.01%],  [0.300670], [0.479899]),
    ([10],   [0.12%],  [0.805695], [0.932831]),
    ([20],   [0.25%],  [0.910385], [0.981240]),
    ([50],   [0.62%],  [0.965494], [0.996147]),
    ([100],  [1.25%],  [0.983920], [0.997990]),
    ([300],  [3.74%],  [0.994472], [1.000000]),
    ([500],  [6.24%],  [0.997152], [1.000000]),
    ([1000], [12.48%], [0.999162], [1.000000]),
  ),
) <evaluation-sweep-cluster-llama32-3b-instruct-8016-table>


*For 16032 clusters:*

#```python
cluster quality metrics:
num_clusters                     = 16032
mean_assigned_similarity         = 0.669897
p05_assigned_similarity          = 0.486596
mean_margin_to_best_other        = 0.155846
p05_margin_to_best_other         = -0.020911
fraction_assigned_to_nearest     = 0.935980
min_cluster_size                 = 8
max_cluster_size                 = 8
```

// #figure(
//   image("my-figures/plots/clustering/flashhead_llama32_3b_16032c_topk_sweep_top1_match_rate.png", width: 105%),
//     caption: [FlashHead evaluation sweep for a 16k cluster for Llama 3.2 3B Instruct.],
// ) <evaluation-sweep-cluster-16k-llama3.2-3B-instruct-img>

#probing-table(
  title: [Top-1 and Top-3 Match Rate by Probed Clusters],
  model-name: [Llama 3.2 3B Instruct],
  commit-id: "7bce16c",
  windows: 30,
  window-length: 200,
  total-clusters: 16032,
  rows: (
    ([1],    [0.01%], [0.342881], [0.572027]),
    ([10],   [0.06%], [0.819263], [0.940369]),
    ([20],   [0.12%], [0.918593], [0.981910]),
    ([50],   [0.31%], [0.970854], [0.997990]),
    ([100],  [0.62%], [0.987270], [1.000000]),
    ([300],  [1.87%], [0.996482], [1.000000]),
    ([500],  [3.12%], [0.998660], [1.000000]),
    ([1000], [6.24%], [0.999162], [1.000000]),
  ),
) <evaluation-sweep-cluster-llama32-3b-instruct-16032-table>


== Training HVC-bridge


=== Training (1,1) gap

#figure(
  image(
    "my-figures/plots/(1,1) gap training/thesis_gap11__top1_drafter_matches_verifier__train.png",
    width: 90%,
  ),
  caption: [
    Top-1 agreement between drafter and verifier during training for the $(1, 1)$ gap setting.
    Curves show the first 10 points raw, followed by a 5-point centered moving average.
    The right panel reports the final average over the last 20 datapoints.
  ],
) <fig-gap11-training-top1-agreement>

#figure(
  image(
    "my-figures/plots/(1,1) gap training/thesis_gap11__kl_verifier_to_drafter__train.png",
    width: 90%,
  ),
  caption: [
    KL divergence from verifier to drafter during training for the $(1, 1)$ gap setting.
    Curves show the first 10 points raw, followed by a 5-point centered moving average.
    The right panel reports the final average over the last 20 datapoints.
  ],
) <fig-gap11-training-kl-verifier-to-drafter>


=== Training (2,2) gap

#figure(
  image(
    "my-figures/plots/(2,2) gap training/thesis_gap22__top1_drafter_matches_verifier__train.png",
    width: 90%,
  ),
  caption: [
    Top-1 agreement between drafter and verifier during training for the $(2, 2)$ gap setting.
    Curves show the first 10 points raw, followed by a 5-point centered moving average.
    The right panel reports the final average over the last 20 datapoints.
  ],
) <fig-gap22-training-top1-agreement>

#figure(
  image(
    "my-figures/plots/(2,2) gap training/thesis_gap22__kl_verifier_to_drafter__train.png",
    width: 90%,
  ),
  caption: [
    KL divergence from verifier to drafter during training for the $(2, 2)$ gap setting.
    Curves show the first 10 points raw, followed by a 5-point centered moving average.
    The right panel reports the final average over the last 20 datapoints.
  ],
) <fig-gap22-training-kl-verifier-to-drafter>

== Measuring speedups and memory usage

The real world speedup is benchmarked with the produced implementation. Speedup per generated token is as the main measurement but detailed numbers of measured memory usage, acceptance rate, verifier cost, LM-head compute split, ANNH accuracy and ANNH vs full LM-head speedup is also included.

The implementation runs self-speculation for all the prompts in the prompt set and the inference speedup for each prompt is stored. The data is presented as a histogram with a fitted normal curve. The magnitude of each bin in the histogram represents how many prompts that produced a speedup in that range.

The self-speculation is first run with the drafter skipping layers but using the normal LM-head, this is represented in blue. Then it is run with the drafter skipping layers and using ANNH instead of the full LM-head, this is represented in the color magenta.






=== Gap (1,1), Concrete prompt set, block size 2, bfloat16

The prompt set `concrete-completion-style` is used together with a static block size of 2. The models are run in bfloat16 and the HVC-bridge is run in float32. These results are with the gap (1,1) which means that all layers are skipped except the first and the last one. No internal timing is used when measuring speedup to not have syncs that would affect performance.


#figure(
  move(
    dx: -0.2cm,
    image(
      "my-figures/plots/benches/bench_self_spec__llama-3-1-8b-instruct__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260512_190545.png",
      width: 115%,
    ),
  ),
  caption: [
    Per-token self-speculation speedup for Llama 3.1 8B Instruct on the concrete prompt set.
    The blue distribution is speedups for different prompts using skipped layers in the drafter. The magenta distribution is for the drafter with both skipped layers and with ANNH. The speedups reported are per-token while the histogram and normal curves are speedup per prompt.
  ],
) <fig:self-spec-llama-31-8b-concrete>

From @fig:self-spec-llama-31-8b-concrete a speedup of 1.45x with skipped layers and a speedup of 1.56x with skipped layers and ANNH can be seen. The peak memory usage is 15.12 GiB for the normal inference, 15.13 GiB for the self-spec with skipped layers and 15.19 GiB when skipping layers and using ANNH.

The figure shows that the head has a speedup with 7.68x when going from full LM-head to ANNH and an resulting accuracy of 97.6%. This makes the acceptance rate go from 47.6% without ANNH to 46.5% with it. The fraction of when the outputs exactly match normal generation is 43.3% both with and without ANNH. This does not mean that the self-speculation is incorrect or that it is approximate. See the discussion for why this happens even without approximation. 

Before the measured benchmarks, 5 warmup runs and 15 profile runs were performed. The used GPU was a NVIDIA L4. The result shows that a verifier call costs 1.05x of a normal token generation call. The block size is 2, which means that this follows the expectation that verifying two tokens and generating a bonus token is cheaper is significantly cheaper then generation. The result shows that speeding up both the head and the body does increase total speedup which follows the Amdahl's law reasoning. The figure shows that the drafter is 15.1% of the full model in compute with skipped layers, and 9.1% of the full model with skipped layers and ANNH.

Using @selfs-speedup with the observed values $v = 1.05$, $gamma = 2$, $a = 47.6%$, and $d = 15.1%$, the predicted speedup is
$
S = frac(1 + 2 dot 0.476, 1.05 + 2 dot 0.151) = frac(1.952, 1.352) approx 1.444 times,
$
which aligns closely with the measured 1.45x. For the version with ANNH, setting $d = 9.1%$ and $a = 46.5%$ gives $S approx 1.567 times$, again consistent with the measured 1.56x.


#figure(
  move(
    dx: -0.2cm,
    image(
      "my-figures/plots/benches/bench_self_spec__llama-3-2-3b-instruct__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260512_175238.png",
      width: 115%,
    ),
  ),
  caption: [
    Per-token self-speculation speedup for Llama 3.2 3B Instruct on the concrete prompt set.
    The blue distribution is speedups for different prompts using skipped layers in the drafter. The magenta distribution is for the drafter with both skipped layers and with ANNH. The speedups reported are per-token while the histogram and normal curves are speedup per prompt.
  ],
) <fig:self-spec-llama-32-3b-concrete>

The @fig:self-spec-llama-32-3b-concrete shows the same pattern of speedup as @fig:self-spec-llama-31-8b-concrete. The speedups are here smaller, 1.3x and 1.44x respectively. This illustrates the hypothesis of overhead for easy tokens. When the model is smaller, there is less overhead for easy tokens resulting in less gain. The figure shows that the ANNH is 7.02x faster than the normal LM-head and that the memory usage is approximately the same for all three versions

#figure(
  move(
    dx: -0.2cm,
    image(
      "my-figures/plots/benches/bench_self_spec__llama-3-2-1b-instruct__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260512_164626.png",
      width: 115%,
    ),
  ),
  caption: [
    Per-token self-speculation speedup for Llama 3.2 1B Instruct on the concrete prompt set.
    The blue distribution is speedups for different prompts using skipped layers in the drafter. The magenta distribution is for the drafter with both skipped layers and with ANNH. The speedups reported are per-token while the histogram and normal curves are speedup per prompt.
  ],
) <fig:self-spec-llama-32-1b-concrete>


@fig:self-spec-llama-32-1b-concrete shows that the Llama 3.2 1B Instruct also gets speedups of 1.13x and 1.27x. The speedups are here smaller and again following the idea of overhead. The self-speculation runs use approximately the same amount of memory as normal inference.


#figure(
  move(
    dx: -0.2cm,
    image(
      "my-figures/plots/benches/bench_self_spec__mistral-7b-instruct-v0-3__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260512_202037.png",
      width: 115%,
    ),
  ),
  caption: [
    Per-token self-speculation speedup for Mistral 7B Instruct 0.3v on the concrete prompt set.
    The blue distribution is speedups for different prompts using skipped layers in the drafter. The magenta distribution is for the drafter with both skipped layers and with ANNH. The speedups reported are per-token while the histogram and normal curves are speedup per prompt.
  ],
) <fig:self-spec-mistral-7b-concrete>


The Mistral 7B Instruct shows a relatively large speedup of 1.61x with skipped layers but a speedup of 1.60x with skipped layers + ANNH, even though it made the head 3.06x faster. @fig:self-spec-mistral-7b-concrete shows that the LM-head portion is 1.85% of the parameters, so the win from speeding up the head is erased from the slight drop in acceptance rate by doing so.

Using @selfs-speedup with $v = 1.05$, $gamma = 2$, $a = 51.3%$, and $d = 10.3%$ (skipped layers only), the predicted speedup is
$
S = frac(1 + 2 dot 0.513, 1.05 + 2 dot 0.103) = frac(2.026, 1.256) approx 1.613 times,
$
which matches the measured 1.61x closely. With ANNH, substituting $a = 50.0%$ and $d = 9.5%$ gives
$
S = frac(1 + 2 dot 0.500, 1.05 + 2 dot 0.095) = frac(2.000, 1.240) approx 1.613 times,
$
which also roughly matches the measured 1.60x. 


#figure(
  move(
    dx: -0.2cm,
    image(
      "my-figures/plots/benches/bench_self_spec__qwen3-4b-instruct-2507__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260512_210900.png",
      width: 115%,
    ),
  ),
  caption: [
    Per-token self-speculation speedup for Qwen3 4B Instruct on the concrete prompt set.
    The blue distribution is speedups for different prompts using skipped layers in the drafter. The magenta distribution is for the drafter with both skipped layers and with ANNH. The speedups reported are per-token while the histogram and normal curves are speedup per prompt.
  ],
) <fig:self-spec-qwen3-4b-concrete>


@fig:self-spec-qwen3-4b-concrete shows that the implementation also gives speedups for Qwen3. The acceptance rates are relatively low of 36.2% and 35.8%. This manages to result in speedups of 1.25x and 1.34x. A block size of 2 can be to big for this drafter. 

=== float32 debug test

#figure(
  move(
    dx: -0.2cm,
    image(
      "my-figures/plots/benches/debug/bench_self_spec__llama-3-2-1b-instruct__concrete-completion-style__keep-1-1__block-2__max-200__warmup-5__profile-15__both__20260512_190254.png",
      width: 115%,
    ),
  ),
  caption: [
    Per-token self-speculation speedup for Llama 3.2 1B Instruct on the concrete prompt set. This is a debug run to investigate generation correctness. The datatype for the model is float32.
  ],
) <fig:self-spec-llama32-1b-float32-concrete>

@fig:self-spec-llama32-1b-float32-concrete shows a debug run with float32. Here the match between normal generation and self-speculation generation is 100% for both drafter versions. The number of tokens generated is also exactly the same with 16811 in for normal and with the self-speculation implementation. 


= Discussion

== Interpreting the results

=== Layer-skipping ablations
// Why does gap-jump consistently outperform early-exit and late-start?
// Why is the pattern consistent across model families?
// What does this tell us about the role of early vs late layers?

=== HVC bridge training
// How well did the bridge recover generation quality?
// What explains the differences in top-1 agreement across model families?
// Why does (2,2) reach higher top-1 than (1,1) as expected from the notation?

=== FlashHead accuracy and speedup
// What does the accuracy/clusters-probed tradeoff tell us?
// Why does more clusters strictly improve accuracy per percentage probed?
// When is ANNH worth using and when is it not (e.g. Mistral head fraction too small)?

=== Self-speculation speedups
// Why do larger models benefit more than smaller ones?
// Why does the theoretical formula predict measured speedups so closely?
// What does the Amdahl framing tell us about the ceiling for each model?

== Answering the research questions

// RQ1: Which layer-skipping strategy minimizes damage to generation quality?
// RQ2: Can the HVC bridge recover enough quality to serve as an effective drafter?
// RQ3: Does approximating both head and body produce higher speedup than either alone?

== Limitations and sources of error

=== Numerical precision
// bfloat16 vs float32 exact match discrepancy — not a correctness bug, explain why

=== PyTorch implementation
// No custom kernels, speedups are a lower bound on what an optimized stack could achieve

=== Acceptance rate definition
// i.i.d. assumption, distribution drift between draft positions, block size 1 vs 2 comparison

=== Dataset and prompt coverage
// How representative are the prompt sets? English only, completion style only

== Future work
// Larger block sizes
// Larger gaps with stronger HVC
// Adaptive block sizing based on per-prompt acceptance rate
// Optimized CUDA kernels
// Extending to larger models
// Exploring whether HVC generalizes across tasks beyond the training distribution

= Conclusion

== Summary
// 1-2 paragraphs restating the problem, the approach, and the overall finding.

== Contributions
// A concise list of what this thesis produced that didn't exist before.




#pagebreak()
#bibliography("refs.bib")
