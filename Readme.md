# Python Project Template

A modern Python project template with built-in support for code quality tools and VS Code integration.

## Features

- 🐍 Python development environment
- ✨ Code formatting with Black
- 🔍 Linting with Pylint
- 📝 VS Code integration with recommended extensions
- 🐋 Optional Docker support
- 📓 Optional Jupyter Notebook support

## Prerequisites

- Python 3.x
- Visual Studio Code
- Git

## Getting Started

1. Clone this template repository
2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## VS Code Extensions

This template comes with recommended VS Code extensions for Python development:

- Python (ms-python.python)
- Pylint (ms-python.pylint)
- Black Formatter (ms-python.black-formatter)
- AutoDocstring (njpwerner.autodocstring)
- Jupyter (ms-toolsai.jupyter) - Optional
- Docker (ms-azuretools.vscode-docker) - Optional

The extensions will be automatically suggested when you open the project in VS Code.

## Code Quality Tools

### Formatting

- Black formatter with a line length of 90 characters
- Automatic format on save
- Visual ruler at 90 characters

### Linting

- Pylint integration
- Customized line length to match Black formatter
- Automatic lint on save

## Project Structure

```
├── .vscode/                # VS Code configuration
├── src/                    # Source code
├── .gitignore             # Git ignore rules
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Development Settings

The template includes optimized VS Code settings for Python development:

- Auto-save on focus change
- Integrated terminal configuration
- Git integration
- Code navigation features
- Syntax highlighting
- IntelliSense and auto-completion
- Debug configuration


