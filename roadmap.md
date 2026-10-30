# Project Roadmap: Building a Small Math Language Model with JAX

Our goal is to build a small language model **from scratch in JAX**, train it to solve simple math problems, and improve its performance through three stages:

1. **Pretraining:** Teach the model language, mathematical patterns, and basic arithmetic.
2. **Supervised fine-tuning (SFT):** Teach it how to answer math questions and produce useful solutions.
3. **Reinforcement learning with verifiable rewards (RLVR):** Improve its accuracy by rewarding answers that a deterministic verifier confirms are correct.

We'll target your **12 GB NVIDIA GPU** and keep the project small enough to understand, experiment with, and debug. We'll implement the training pipeline ourselves rather than hiding everything behind a high-level training framework.

One important expectation: training from scratch on a single consumer GPU will impose limits on the model's general capabilities. However, we can build a useful experimental model and investigate how pretraining, SFT, and RLVR affect its mathematical performance.

Here is the plan I'd follow.

# Phase 1: Define the model and establish the environment

## Step 1. Set up the JAX training stack

First, we'll ensure the software works correctly with your NVIDIA GPU.

Our primary components will be:

| Component | Purpose |
|---|---|
| JAX | Tensor operations, automatic differentiation, GPU computation, and compilation |
| Flax NNX | Defining the neural network |
| Optax | Optimizers, learning-rate schedules, and gradient transformations |
| Orbax | Saving and restoring model checkpoints |
| Tokenizers | Training and using a tokenizer |
| NumPy and Python | Data preparation, preprocessing, and evaluation |

We'll use **Flax NNX** rather than introducing both the old and new Flax APIs. NNX is the newer API recommended for new Flax projects, and official examples demonstrate building and training small Transformers with JAX and NNX. [Flax](https://flax.readthedocs.io/en/stable/)

We'll begin by checking:

- Whether JAX detects your NVIDIA GPU.
- Whether a simple matrix multiplication runs on the GPU.
- Whether automatic differentiation produces correct gradients.
- Whether a minimal training loop decreases its loss.
- Whether we can save and reload a checkpoint.

We'll also establish reproducible random seeds, experiment configuration files, and basic logging.

**Why start here?** Debugging model code, CUDA compatibility, and training behavior simultaneously makes problems unnecessarily difficult to isolate.

One hardware detail matters: the appropriate JAX installation depends on your operating system, GPU architecture, and NVIDIA driver. JAX currently documents separate CUDA 12 and CUDA 13 installation options, with different GPU and driver requirements. We'll choose the correct one after identifying your hardware. [JAX Documentation](https://docs.jax.dev/en/latest/installation.html)

## Step 2. Choose our initial model architecture

We'll build a decoder-only Transformer similar in overall design to GPT.

I suggest starting with a model containing approximately **60 million parameters**.

A reasonable initial configuration is:

| Hyperparameter | Initial value |
|---|---:|
| Architecture | Decoder-only Transformer |
| Number of layers | 10 |
| Hidden dimension | 640 |
| Attention heads | 10 |
| Feed-forward dimension | 2,560 |
| Vocabulary size | 16,000 tokens |
| Initial context length | 512 tokens |
| Positional encoding | Rotary positional embeddings (RoPE) |
| Normalization | RMSNorm |
| Parameter count | Approximately 60 million |
| Attention mask | Causal |

These are starting values, not immutable decisions. We'll adjust them after checking actual memory consumption and training throughput.

We'll implement the important components ourselves:

1. Token embeddings.
2. Positional encoding.
3. Causal self-attention.
4. Feed-forward blocks.
5. Normalization and residual connections.
6. The language-model output head.
7. The autoregressive generation function.

We'll also tie the input embedding and output projection weights to avoid unnecessarily increasing the parameter count.

For training, we'll investigate mixed-precision computation, small microbatches, and gradient accumulation. We can consider gradient checkpointing if activation memory becomes a bottleneck.

The objective is to understand the entire model rather than merely invoke a pretrained architecture.

---

# Phase 2: Data and evaluation

## Step 3. Build the dataset pipeline

For a small math-focused model, the quality and structure of the training data will be especially important.

We'll construct three data collections.

**A. General language data**

A relatively small corpus of clean text to teach the model basic language structure, vocabulary, and sentence formation.

**B. Mathematical text**

Examples containing arithmetic, mathematical notation, worked examples, and explanations of basic concepts.

**C. Synthetic mathematical data**

Programmatically generated examples with known answers, covering tasks such as:

- Addition and subtraction.
- Multiplication and division.
- Fractions and decimals.
- Order of operations.
- Missing-number problems.
- Simple word problems.
- Multi-step arithmetic.

For synthetic examples, we'll use programs to generate both the problems and their correct answers.

This is important because it gives us a scalable source of examples whose answers we can verify without relying on another language model.

A reasonable starting experiment would mix general text, mathematical explanations, and synthetic math examples. We'll adjust the proportions based on the model's learning curves and held-out evaluations rather than assume one mixture is optimal.

### Tokenization

We'll train a tokenizer with approximately 8,000–16,000 tokens.

The tokenizer must handle ordinary text, numbers, punctuation, arithmetic operators, and mathematical answer formats.

We'll implement a pipeline that:

1. Cleans and normalizes source data.
2. Trains or loads the tokenizer.
3. Converts examples into token IDs.
4. Packs tokens into training sequences.
5. Creates shifted input and target sequences for next-token prediction.

We'll track where every dataset comes from and ensure that our held-out evaluation examples are not accidentally included in training.

## Step 4. Establish our evaluation benchmarks

Before training, we'll define what success means.

Otherwise, we might produce a model with a decreasing loss but no meaningful improvement at solving math problems.

We'll maintain a fixed evaluation suite containing both automatically generated questions and a recognized math benchmark, such as GSM8K for grade-school word problems.

Our evaluation suite will measure:

| Metric | What it tells us |
|---|---|
| Training loss | Whether the model is learning its training data |
| Validation loss | Whether next-token prediction generalizes |
| Exact-answer accuracy | Whether the final mathematical answer is correct |
| Addition/subtraction accuracy | Performance on basic operations |
| Multiplication/division accuracy | Performance on more challenging arithmetic |
| Word-problem accuracy | Whether it can translate language into mathematical operations |
| Multi-step accuracy | Whether it can solve sequences of related calculations |
| Length generalization | Whether accuracy holds for larger numbers or longer problems |

We'll use a deterministic answer extractor and verifier wherever possible.

For example, if the expected answer is `42`, these responses should be evaluated according to an explicit normalization policy rather than superficial text similarity.

We'll keep the evaluation data separate from training and use the same evaluation suite after each training stage.

**Our central metric will be exact-answer accuracy**, supplemented by measures of how well the model explains its work.

---

# Phase 3: Pretraining

## Step 5. Train the base language model

Now we'll train the Transformer from scratch.

The objective is standard causal language modeling: given a sequence of tokens, predict the next token.

For example:

Input:

> What is 17 + 25? The answer is

The model learns to predict the continuation from the training sequence. At this stage, it is learning token patterns rather than directly optimizing the correctness of its final math answers.

### Training objective

We'll use next-token cross-entropy:

\[
\mathcal{L}_{\text{pretrain}}
=
-\frac{1}{N}
\sum_{t=1}^{N}
\log P_\theta(x_t \mid x_{<t})
\]

Here, \(x_t\) is the current target token and \(x_{<t}\) represents the preceding tokens.

We'll use AdamW, a learning-rate warmup period, a decay schedule, and gradient clipping as the initial training setup.

### How we'll train

We'll proceed incrementally.

**First: a short sanity run.** Confirm that the model trains, loss decreases, GPU memory stays within limits, and checkpoints work.

**Second: a small pilot.** Train on a larger dataset and examine validation loss, generated text, and arithmetic performance.

**Third: a longer training run.** Expand the token budget if the pilot demonstrates meaningful progress.

For planning purposes, we can initially target approximately 100 million tokens for the pilot, then consider several hundred million tokens or more, depending on measured throughput and available training time.

These are experimental targets rather than promises about how quickly the model will converge. We'll use actual tokens-per-second measurements to decide how far to scale.

### Memory considerations

A 60-million-parameter model is a reasonable starting point for a 12 GB GPU, but parameter count alone does not determine peak memory use. Activations, optimizer states, sequence length, batch size, and JAX's compiled execution buffers all matter.

We'll measure peak memory early rather than assume the configuration will fit under every hardware and software combination.

We'll also save periodic checkpoints so that a long training run can resume without starting from scratch.

### What we should have afterward

A checkpoint that contains:

- The model parameters.
- The model configuration.
- The tokenizer.
- The training step and optimizer state.
- The training configuration.

We'll also have baseline evaluation results.

At this point, we have a pretrained language model, but it won't necessarily follow instructions well or solve math problems reliably.

That's what the next stage addresses.

---

# Phase 4: Supervised fine-tuning (SFT)

## Step 6. Teach the model to answer math questions

During SFT, we'll continue training the pretrained model on examples of the kind of behavior we want.

Instead of unstructured text alone, we'll use instruction-response pairs.

For example:

**Prompt**

> What is 37 × 24? Explain your reasoning and give the final answer.

**Target response**

> We can calculate 37 × 24 by splitting 24 into 20 and 4.
>
> 37 × 20 = 740  
> 37 × 4 = 148
>
> Adding these results gives 740 + 148 = 888.
>
> Final answer: 888

We'll create examples for different operations, levels of difficulty, explanation styles, and answer formats.

### Dataset design

Our SFT dataset will contain:

- Basic arithmetic questions.
- Worked mathematical examples.
- Short, multi-step explanations.
- Simple word problems.
- Examples requiring a specific final-answer format.
- Examples that vary number length and problem structure.

Initially, synthetic examples will be useful because we can generate them in large quantities and verify their correctness programmatically.

We should nevertheless inspect randomly sampled examples to catch bugs in our generators. A systematic error in the synthetic data can teach the model consistently incorrect behavior.

### Training objective

We'll retain the language-model cross-entropy objective, but apply it to the response portion of each example.

Conceptually:

\[
\mathcal{L}_{\text{SFT}}
=
-\frac{1}{|\mathcal{R}|}
\sum_{t \in \mathcal{R}}
\log P_\theta(y_t \mid x,y_{<t})
\]

Here, \(x\) is the prompt, \(y_t\) is a response token, and \(\mathcal{R}\) is the set of response-token positions included in the loss.

We'll mask out the prompt tokens so the model learns to produce the desired response rather than being rewarded for predicting text that was already provided.

We'll use a lower learning rate than the initial pretraining run and monitor held-out math accuracy to avoid overfitting.

### What we should have afterward

An instruction-following math model that can attempt problems in a consistent format.

We'll compare its accuracy against the pretrained model, using the same evaluation questions.

That comparison will tell us whether SFT is improving correctness—not just making the responses look more polished.

---

# Phase 5: Reinforcement Learning with Verifiable Rewards (RLVR)

## Step 7. Build a mathematical reward verifier

This is where the project becomes especially interesting.

During SFT, we teach the model to imitate desired answers. During RLVR, we generate answers, verify them, and use the verification results to improve the model's behavior.

For simple mathematics, we have a significant advantage: **we can often evaluate the final answer with ordinary code instead of a learned reward model.**

Suppose the prompt is:

> What is 46 × 17?

The model generates:

> 46 × 17 = 46 × (10 + 7) = 460 + 322 = 782. Final answer: 782.

Our verifier extracts the answer, calculates the expected result independently, and determines whether the answer is correct.

We'll begin with a reward function that distinguishes:

- Correct answers.
- Incorrect answers.
- Malformed or missing final answers.

We can also introduce a modest formatting reward when appropriate.

However, correctness should be the primary signal. We don't want the model to learn that a confident-sounding explanation is more valuable than the right answer.

We'll design the verifier to handle integers, fractions, and decimals according to explicit rules. For generated word problems, the problem generator and answer verifier should be tested independently.

We'll also create unit tests for the verifier itself. An RL system can optimize a faulty reward function extremely effectively, so reward correctness is fundamental.

## Step 8. Implement the reinforcement-learning algorithm

My recommendation is to start with a lightweight, GRPO-style approach.

GRPO stands for **Group Relative Policy Optimization**. Its basic idea is to generate multiple candidate responses to the same prompt and compare their rewards.

A simplified training cycle looks like this:

1. Sample a mathematical question.
2. Generate several candidate answers with the current model.
3. Score each response using the verifier.
4. Calculate a relative advantage based on the rewards within the group.
5. Use that advantage to update the model's policy.
6. Repeat with new questions.

For example:

| Generated answer | Verified correct? | Illustrative reward |
|---|---|---:|
| 46 × 17 = 772 | No | 0 |
| 46 × 17 = 782 | Yes | 1 |
| 46 × 17 = 782 | Yes | 1 |
| 46 × 17 = 712 | No | 0 |

These are illustrative rewards, not a finalized reward specification.

The key is that the training signal comes from verifiable outcomes rather than demonstrations alone.

GRPO uses group-relative rewards to estimate which sampled completions are better than others. Current GRPO documentation also highlights a practical limitation: if every completion in a group receives the same reward, that group supplies no relative learning signal. [GitHub](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md?trk=public_post_comment-text\&utm_source=chatgpt.com)

That limitation will inform our implementation and our choice of training questions.

### How we'll keep RLVR manageable on 12 GB VRAM

We won't begin with a complicated, distributed reinforcement-learning system.

Instead, we'll use small rollout batches, a limited number of completions per question, and short generation lengths.

We'll also need to account for the memory and computational cost of generating answers and calculating policy gradients.

For the first implementation, we should consider:
- A small group of sampled answers per prompt.
- A controlled generation length.
- Gradient accumulation where useful.
- A reference model or equivalent KL regularization to discourage the policy from drifting too far from the SFT model.
- Frequent evaluation checkpoints.

We can implement the training loop using JAX, although the rollout and optimization logic will take more care than the SFT loop.

As an alternative reference, Google's JAX-native Tunix project includes SFT and reinforcement-learning methods such as GRPO. Its initial positioning emphasizes TPU-based training, particularly alongside MaxText, so we'll evaluate its suitability rather than assume it is the best fit for a single consumer NVIDIA GPU. [Google Developers Blog](https://developers.googleblog.com/en/introducing-tunix-a-jax-native-library-for-llm-post-training/)

### The most important experimental question

Does RLVR actually improve math accuracy?

We will test that rather than assume it does.

We'll keep the SFT checkpoint and compare the results before and after RLVR. If the RLVR model becomes better at formatting answers but worse at solving problems, we'll know that the training objective or reward design needs improvement.

---

# Phase 6: Evaluate the results and run controlled experiments

## Step 9. Compare the three training stages

We'll evaluate all three model versions on the same held-out questions.

| Model | Main purpose | What we'll measure |
|---|---|---|
| Pretrained | General language and mathematical patterns | Validation loss and math accuracy |
| SFT | Instruction following and worked solutions | Exact-answer accuracy and response quality |
| RLVR | Improvement through verified rewards | Exact-answer accuracy and generalization |

We'll break down accuracy by problem type.

For example, the model might achieve high addition accuracy but struggle with multiplication involving larger numbers. It might also learn to solve familiar templates without generalizing to differently worded questions.

We want to distinguish genuine improvement from memorization or overfitting.

We'll therefore test:
- Familiar and unfamiliar problem formats.
- Different number lengths.
- Different combinations of arithmetic operations.
- Held-out synthetic word problems.
- A recognized benchmark such as GSM8K.

We'll keep a final test set separate from our training decisions. That lets us judge the final model on examples that haven't influenced our training or hyperparameter choices.

## Step 10. Package the model for inference

Finally, we'll build a small inference interface that can load the model and answer a question.

The interface should:

1. Load the model checkpoint and tokenizer.
2. Accept a mathematical question.
3. Generate a response.
4. Extract the final answer when possible.
5. Optionally run the verifier.
6. Report the result and evaluation information during testing.

We should keep inference generation separate from training code wherever possible.

This makes it easier to compare checkpoints, test generation settings, and investigate whether apparent training improvements actually translate into better answers.

---

# The Recommended Implementation Order

Although this project has three major training stages, we shouldn't implement everything at once.

I'd organize the actual development into these milestones:

| Milestone | Deliverable | Completion criterion |
|---|---|---|
| 1. Environment | Working JAX installation on the GPU | JAX detects and uses the GPU |
| 2. Model | Small decoder-only Transformer | Forward pass, loss, and gradients work |
| 3. Training infrastructure | Training loop and checkpoints | A short run completes and resumes |
| 4. Data pipeline | Tokenizer and packed training sequences | Data loads consistently with no evaluation leakage |
| 5. Pretraining | Trained base model | Training is stable and validation metrics are recorded |
| 6. SFT | Instruction-following checkpoint | Math accuracy and response behavior are measured |
| 7. Verifier | Tested deterministic math evaluator | Correctness labels pass unit tests |
| 8. RLVR | Reward-based training loop | Reward and held-out accuracy are tracked |
| 9. Evaluation | Comparative results across stages | We can tell which stages improve performance |
| 10. Inference | Runnable math model | The final checkpoint answers new questions |

The order is deliberate. We'll establish a working model and training loop before spending substantial time preparing data or running long training jobs.

We'll also save checkpoints at each major milestone so that SFT and RLVR can be repeated without restarting pretraining.

---

# My Proposed Starting Configuration

To make this project concrete, here's the initial plan I'd use.

| Decision | Proposal |
|---|---|
| Framework | JAX + Flax NNX + Optax |
| Model | Decoder-only Transformer |
| Initial size | Approximately 60M parameters |
| GPU | One NVIDIA GPU with 12 GB VRAM |
| Initial context | 512 tokens |
| Tokenizer | 8K–16K vocabulary |
| Pretraining objective | Causal next-token prediction |
| Initial data emphasis | Language, mathematical explanations, and verified synthetic math |
| SFT objective | Response-only cross-entropy |
| RLVR verifier | Deterministic arithmetic and answer checking |
| Initial RL approach | Small-batch, GRPO-style optimization |
| Primary metric | Held-out exact-answer accuracy |
| Main experimental question | How much does each training stage improve mathematical problem-solving? |

One important distinction: our first model and data configuration are hypotheses to test, not guarantees of a particular accuracy. We'll use the initial experiments to determine whether the model has enough capacity, whether the data is useful, and whether the available GPU makes longer training runs practical.

## What we should do first

**Our next step should be to establish the hardware configuration and build the minimal JAX Transformer training skeleton.** Before choosing installation commands, we should identify your GPU model, operating system, and available system RAM. Those details determine the appropriate JAX/CUDA setup and help us choose safe initial batch sizes.

Once the skeleton successfully performs a forward pass, computes loss, updates its parameters, and saves a checkpoint, we'll have the foundation for all three training stages.