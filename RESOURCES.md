# RESOURCES — Microsoft NNI

High-trust sources for learning NNI. (Gathered 2026-08-10; links verified
reachable at capture time. NNI is archived, so the docs are frozen — they will
not drift, which is good for referencing.)

## Primary
- **NNI GitHub (archived)**: https://github.com/microsoft/nni
  - Description: "An open source AutoML toolkit for automate machine learning
    lifecycle, including feature engineering, neural architecture search, model
    compression and hyper-parameter tuning."
  - Note: `archived: true`. Read it as a finished artifact.
- **NNI ReadTheDocs (official docs)**: https://nni.readthedocs.io/en/stable/
  - Frozen at last published version. Covers HPO, NAS, compression, feature
    engineering, experiment management.

## Component reference (from the archived README)
- **HPO tuners**: Grid Search, Random, Anneal (Hyperopt), Evolution, Hyperband,
  PBT, BOHB, DNGO, GP, Metis, SMAC, TPE.
- **NAS strategies**: Grid Search, Policy-Based RL, Random, Regularized
  Evolution, TPE, DARTS, ENAS, FBNet, ProxylessNAS, SPOS.
- **Pruners**: Level, L1-Norm, Taylor FO Weight, Movement, (plus AGP, Slim,
  Lottery Ticket, etc. in docs).
- **Quantizers**: QAT, DoReFa, BNN, LSQ, etc. (INT8 / binary).
- **Assessors (early stopping)**: Median Stop, Curveball, Learning Curve.
- **Advisors**: BOHB, Hyperband, Metis, and the legacy Hyperopt/GP advisors.

## Background papers (for the ideas behind the components)
- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization*,
  JMLR. https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
- Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization (TPE)*,
  NeurIPS. https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
- Zoph & Le (2017), *Neural Architecture Search with Reinforcement Learning*
  (NAS). https://arxiv.org/abs/1611.01578
- Liu et al. (2019), *DARTS: Differentiable Architecture Search*. https://arxiv.org/abs/1806.09055

## Modern alternatives (to compare against, given NNI is archived)
- **Optuna**: https://optuna.org/ — active, lightweight HPO.
- **Ray Tune**: https://docs.ray.io/en/latest/tune/ — scalable HPO/NAS.
- **AutoKeras**: https://autokeras.com/ — Keras-based AutoML (Closely related
  to Keras Tuner).
- **Keras Tuner**: https://keras.io/keras_tuner/ — if I want NNI-style tuning
  but staying inside Keras 3 (which I already pinned for Colab work).

## Community
- NNI GitHub Issues/Discussions (read-only now, but searchable history).
- r/MachineLearning and r/LocalLLaMA for practical AutoML experiences.
- For my R530 hobby: the local-AI/ML self-hosting communities.
