# Contributing to ML Experiment Toolkit

Thank you for your interest in contributing to ML Experiment Toolkit! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/ml-experiment-toolkit.git
   cd ml-experiment-toolkit
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Run tests**:
   ```bash
   pytest
   ```

4. **Format your code**:
   ```bash
   black mlexp/ tests/ examples/
   ```

5. **Check for linting issues**:
   ```bash
   flake8 mlexp/ tests/ examples/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: description of your changes"
   ```

7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** on GitHub

## Coding Standards

### Code Style

- **Follow PEP 8** style guidelines
- **Use Black** for code formatting (line length: 100)
- **Add type hints** where appropriate
- **Write docstrings** for all public functions and classes

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """
    Brief description of the function.

    Longer description if needed, explaining what the function does,
    any important details, or usage examples.

    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, default=10
        Description of param2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> function_name("test", 20)
    True
    """
    pass
```

### Testing

- **Write tests** for all new functionality
- **Aim for good coverage** (80%+)
- **Test edge cases** and error conditions
- **Use descriptive test names**: `test_function_name_scenario`

### Commit Messages

Use clear, descriptive commit messages:

- **Add**: for new features
- **Fix**: for bug fixes
- **Update**: for updates to existing features
- **Refactor**: for code refactoring
- **Docs**: for documentation changes
- **Test**: for test additions/changes

Example:
```
Add: support for custom metrics in model comparison
Fix: handle missing values in data profiler
Update: improve visualization styling
```

## Areas for Contribution

We welcome contributions in the following areas:

### Features

- Additional data quality checks
- Support for more ML frameworks (XGBoost, LightGBM, etc.)
- Time series analysis support
- Model explainability integration
- Export functionality (CSV, JSON, HTML)
- Interactive dashboards

### Improvements

- Performance optimizations
- Better error handling
- More comprehensive tests
- Documentation improvements
- Code refactoring

### Bug Fixes

- Report bugs via GitHub Issues
- Include steps to reproduce
- Provide error messages and stack traces

## Pull Request Process

1. **Update documentation** if you've changed functionality
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG.md** (if applicable)
5. **Request review** from maintainers

## Questions?

If you have questions or need help, please:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the documentation

Thank you for contributing to ML Experiment Toolkit! 🎉

