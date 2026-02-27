# Contributing to RL-TBoost

Thank you for your interest! Contributions that improve the RL environment, extend the TDA feature set, or apply RL-TBoost to new clinical datasets are very welcome.

## 🐛 Reporting Bugs

Open a [GitHub Issue](https://github.com/MorillaLab/RL-TBoost/issues) with:
- The notebook or script where the error occurs
- Your environment (OS, Python version, TensorFlow version, giotto-tda version)
- The full error traceback
- A minimal reproducible example if possible

## 💡 Suggesting Features

Open an issue tagged `enhancement`. Good examples:
- Alternative RL algorithms (PPO, A3C, SAC) as agents
- New TDA feature types (cubical complexes, Mapper graphs)
- Extended cohort analysis (3-year survival, graft function)
- Multi-action extensions beyond binary {run, stop}
- Integration with electronic health record systems

## 🔧 Submitting Code

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install flake8 pytest
   ```
3. Keep RL environment code in `Reinforcement_Learning_Model/`,
   TDA feature code in `Representation_Learning/`,
   and shared helpers in `utilities/`.
4. Lint:
   ```bash
   flake8 utilities/ --max-line-length=127
   ```
5. Clear notebook outputs before committing.
6. Open a pull request against `main` with a clear technical and clinical motivation.

## 📋 RL Environment Guidelines

When modifying or extending the RL environment:
- The reward signal must remain grounded in a topologically meaningful criterion
- Document the `state`, `action`, and `reward` spaces clearly in the class docstring
- Include a test that verifies the environment complies with the OpenAI Gym interface: `env.check_env()`

## 📜 License

By contributing, you agree your work will be released under GPL-3.0.
