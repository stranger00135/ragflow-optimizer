# Contributing to RAGFlow Optimizer

Thanks for your interest in contributing! 🎉

## How to Contribute

### Bug Reports & Feature Requests
- Open an [issue](https://github.com/stranger00135/ragflow-optimizer/issues) with a clear description
- Include your RAGFlow version, Python version, and OS
- For bugs: include the error message and steps to reproduce

### Pull Requests
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Test with `python main.py test-api` and a sample run
5. Submit a PR with a clear description

### Areas We'd Love Help With
- **New chunking presets** for different document types
- **Additional LLM providers** (Azure OpenAI, local models, etc.)
- **Visualization** of optimization results
- **Benchmark datasets** for reproducible comparisons
- **Translations** of documentation

## Development Setup

```bash
git clone https://github.com/stranger00135/ragflow-optimizer.git
cd ragflow-optimizer
pip install -r requirements.txt
cp .env.example .env
# Fill in your credentials in .env
```

## Code Style
- Python 3.10+
- Keep functions focused and well-documented
- Add docstrings for public functions

## Questions?
Open a discussion or issue — we're happy to help!
