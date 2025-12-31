# Chapter 7: Hands-on Project 1: Building a Smart Code Assistant

In the previous chapters, we've explored various prompt engineering techniques, patterns, and tools for developing LLM-powered applications. Now it's time to put that knowledge into practice by building a practical tool that can help streamline your daily coding tasks. In this chapter, we'll create a Smart Code Assistant that leverages LLMs to automate common coding tasks.

## 7.1 Project Overview

### 7.1.1 Problem Statement

Software development involves numerous repetitive tasks that consume valuable time and mental energy:

- Writing boilerplate code for new classes, functions, or modules
- Creating comprehensive documentation for existing code
- Refactoring code for better readability or performance
- Understanding unfamiliar code or complex algorithms
- Writing unit tests for existing code
- Converting code between different programming languages

While integrated development environments (IDEs) offer some assistance, they often lack the flexibility and contextual understanding that LLMs can provide.

### 7.1.2 Solution: The Smart Code Assistant

We'll build a Python-based Smart Code Assistant that leverages the power of LLMs to:

1. Generate boilerplate code based on natural language specifications
2. Analyze and explain existing code
3. Suggest refactoring improvements for small functions
4. Generate unit tests for given functions
5. Assist with code documentation
6. Provide language conversion between Python, JavaScript, and Java

### 7.1.3 Technical Requirements

- Python 3.8+ environment
- OpenAI API key (or equivalent for another LLM provider)
- Command-line interface for easy integration with existing workflows
- Simple, modular architecture for future extensions
- Option to save results to files or copy to clipboard

## 7.2 Setting Up the Project

Let's begin by setting up our project structure and installing the necessary dependencies.

### 7.2.1 Project Structure

```
smart_code_assistant/
├── __init__.py
├── main.py              # Entry point for the command-line interface
├── code_assistant.py    # Core functionality
├── prompt_library.py    # Prompt templates
├── utils/
│   ├── __init__.py
│   ├── clipboard.py     # Clipboard utilities
│   ├── file_utils.py    # File handling utilities
│   └── token_counter.py # Token counting utilities
├── config.py            # Configuration handling
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

### 7.2.2 Installing Dependencies

Create a `requirements.txt` file with the following dependencies:

```
openai>=1.0.0
typer>=0.9.0
rich>=13.5.0
pyperclip>=1.8.2
tiktoken>=0.5.0
python-dotenv>=1.0.0
```

Install these dependencies using pip:

```bash
pip install -r requirements.txt
```

### 7.2.3 Configuration Setup

Create a `config.py` file to handle API keys and other settings:

```python
import os
import dotenv
from pathlib import Path

# Load environment variables from .env file
dotenv.load_dotenv()

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Application paths
APP_DIR = Path.home() / ".smart_code_assistant"
CACHE_DIR = APP_DIR / "cache"
OUTPUT_DIR = APP_DIR / "output"

# Ensure directories exist
APP_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Default languages supported
SUPPORTED_LANGUAGES = [
    "python", "javascript", "typescript", "java", "c++", "csharp", "go", "rust"
]
```

Create a `.env` file in the project root to store your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
DEFAULT_MODEL=gpt-3.5-turbo
```

## 7.3 Building the Core Components

### 7.3.1 Prompt Library

First, let's create our prompt library with templates for different coding tasks. Create a `prompt_library.py` file:

```python
class PromptTemplate:
    def __init__(self, template, required_params=None):
        self.template = template
        self.required_params = required_params or []
    
    def format(self, **kwargs):
        # Ensure all required parameters are provided
        missing = [param for param in self.required_params if param not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")
        
        # Format the template with the provided parameters
        return self.template.format(**kwargs)


class CodePromptLibrary:
    def __init__(self):
        self.prompts = {}
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        # Code generation prompts
        self.prompts["generate_function"] = PromptTemplate(
            """You are an expert software developer. Write a {language} function that {description}.

Requirements:
{requirements}

Your function should be well-documented with comments explaining the logic.
Only return the code with no additional explanations.
""",
            ["language", "description", "requirements"]
        )
        
        # Code explanation prompts
        self.prompts["explain_code"] = PromptTemplate(
            """Explain the following {language} code in detail:

```{language}
{code}
```

Include in your explanation:
1. What the code does
2. The key components and their purpose
3. Any algorithms or patterns used
4. Potential edge cases or limitations
""",
            ["language", "code"]
        )
        
        # Refactoring prompts
        self.prompts["refactor_code"] = PromptTemplate(
            """Refactor the following {language} code to improve its {focus}:

```{language}
{code}
```

Provide the refactored code and explain what improvements you made.
Focus specifically on improving {focus} while maintaining the same functionality.
""",
            ["language", "code", "focus"]
        )
        
        # Unit test generation prompts
        self.prompts["generate_tests"] = PromptTemplate(
            """Write comprehensive unit tests for the following {language} function:

```{language}
{code}
```

The tests should:
1. Cover normal cases, edge cases, and potential errors
2. Be well-structured and properly named
3. Use {test_framework} as the testing framework
4. Include comments explaining the purpose of each test case
""",
            ["language", "code", "test_framework"]
        )
        
        # Documentation generation prompts
        self.prompts["generate_docs"] = PromptTemplate(
            """Generate comprehensive documentation for the following {language} code:

```{language}
{code}
```

The documentation should:
1. Follow {doc_style} documentation style
2. Include parameter descriptions, return values, and exceptions
3. Provide a clear overview of what the code does and how to use it
4. Include usage examples where appropriate
""",
            ["language", "code", "doc_style"]
        )
        
        # Code conversion prompts
        self.prompts["convert_code"] = PromptTemplate(
            """Convert the following {source_language} code to {target_language} while maintaining the same functionality:

```{source_language}
{code}
```

Ensure the converted code:
1. Follows the idiomatic conventions of {target_language}
2. Preserves the original functionality and logic
3. Includes equivalent error handling
4. Is well-commented to explain any non-trivial conversions
""",
            ["source_language", "target_language", "code"]
        )
    
    def get_prompt(self, prompt_name, **kwargs):
        """Get a formatted prompt by name with the provided parameters"""
        if prompt_name not in self.prompts:
            raise ValueError(f"Unknown prompt: {prompt_name}")
        
        return self.prompts[prompt_name].format(**kwargs)
```

### 7.3.2 Core Code Assistant Implementation

Now, let's create the core `code_assistant.py` file that will handle interactions with the LLM:

```python
import openai
import tiktoken
import json
import time
from pathlib import Path

import config
from prompt_library import CodePromptLibrary

class SmartCodeAssistant:
    def __init__(self, api_key=None, model=None):
        """Initialize the Smart Code Assistant"""
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.DEFAULT_MODEL
        self.prompt_library = CodePromptLibrary()
        
        # Configure OpenAI client
        openai.api_key = self.api_key
        
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.encoding_for_model(self.model)
    
    def _send_request(self, prompt, temperature=None, max_tokens=None):
        """Send a request to the OpenAI API"""
        temperature = temperature if temperature is not None else config.TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error calling OpenAI API: {str(e)}")
    
    def count_tokens(self, text):
        """Count the number of tokens in the given text"""
        return len(self.tokenizer.encode(text))
    
    def generate_function(self, description, language="python", requirements=""):
        """Generate code based on a natural language description"""
        prompt = self.prompt_library.get_prompt(
            "generate_function",
            language=language,
            description=description,
            requirements=requirements
        )
        return self._send_request(prompt, temperature=0.2)
    
    def explain_code(self, code, language="python"):
        """Explain the given code in detail"""
        prompt = self.prompt_library.get_prompt(
            "explain_code",
            language=language,
            code=code
        )
        return self._send_request(prompt)
    
    def refactor_code(self, code, focus="readability", language="python"):
        """Refactor code to improve a specific aspect"""
        prompt = self.prompt_library.get_prompt(
            "refactor_code",
            language=language,
            code=code,
            focus=focus
        )
        return self._send_request(prompt)
    
    def generate_tests(self, code, language="python", test_framework="pytest"):
        """Generate unit tests for the given code"""
        prompt = self.prompt_library.get_prompt(
            "generate_tests",
            language=language,
            code=code,
            test_framework=test_framework
        )
        return self._send_request(prompt)
    
    def generate_docs(self, code, language="python", doc_style="Google"):
        """Generate documentation for the given code"""
        prompt = self.prompt_library.get_prompt(
            "generate_docs",
            language=language,
            code=code,
            doc_style=doc_style
        )
        return self._send_request(prompt)
    
    def convert_code(self, code, source_language="python", target_language="javascript"):
        """Convert code from one language to another"""
        prompt = self.prompt_library.get_prompt(
            "convert_code",
            source_language=source_language,
            target_language=target_language,
            code=code
        )
        return self._send_request(prompt)
```

### 7.3.3 Utility Functions

Create a utility module for file operations and clipboard interaction. First, create the `utils/file_utils.py`:

```python
import os
from pathlib import Path

def read_file(file_path):
    """Read content from a file"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path, content):
    """Write content to a file"""
    path = Path(file_path)
    
    # Create directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    return path

def get_language_from_extension(file_path):
    """Determine language from file extension"""
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'c++',
        '.cc': 'c++',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'c++',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin'
    }
    
    extension = Path(file_path).suffix.lower()
    return extension_map.get(extension, 'text')
```

Now, create `utils/clipboard.py`:

```python
import pyperclip

def copy_to_clipboard(text):
    """Copy text to clipboard"""
    try:
        pyperclip.copy(text)
        return True
    except Exception as e:
        print(f"Failed to copy to clipboard: {e}")
        return False

def paste_from_clipboard():
    """Paste text from clipboard"""
    try:
        return pyperclip.paste()
    except Exception as e:
        print(f"Failed to paste from clipboard: {e}")
        return ""
```

Create `utils/token_counter.py`:

```python
import tiktoken
import config

def count_tokens(text, model=None):
    """Count tokens in the given text"""
    model = model or config.DEFAULT_MODEL
    
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to approximate token count if encoding fails
        return len(text) // 4  # Rough approximation: ~4 characters per token
```

Create an empty `__init__.py` in the utils directory to make it a proper package:

```python
# This file makes the utils directory a Python package
```

## 7.4 Building the Command-Line Interface

Let's create a command-line interface using Typer to make our tool easily accessible. Create the `main.py` file:

```python
import typer
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.syntax import Syntax

import config
from code_assistant import SmartCodeAssistant
from utils.file_utils import read_file, write_file, get_language_from_extension
from utils.clipboard import copy_to_clipboard, paste_from_clipboard
from utils.token_counter import count_tokens

# Initialize Typer app and Rich console
app = typer.Typer(help="Smart Code Assistant - Your AI-powered coding companion")
console = Console()

# Initialize code assistant
assistant = SmartCodeAssistant()

def print_code(code, language):
    """Print code with syntax highlighting"""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)

@app.command("generate")
def generate_function(
    description: str = typer.Argument(..., help="Description of the function to generate"),
    language: str = typer.Option("python", "--language", "-l", help="Programming language"),
    requirements: str = typer.Option("", "--requirements", "-r", help="Additional requirements"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    copy: bool = typer.Option(False, "--copy", "-c", help="Copy result to clipboard")
):
    """Generate code based on a natural language description"""
    console.print(f"[bold blue]Generating {language} code for:[/bold blue] {description}")
    
    try:
        result = assistant.generate_function(description, language, requirements)
        
        # Print the result
        print_code(result, language)
        
        # Save to file if requested
        if output_file:
            write_file(output_file, result)
            console.print(f"[green]Code saved to:[/green] {output_file}")
        
        # Copy to clipboard if requested
        if copy:
            copy_to_clipboard(result)
            console.print("[green]Code copied to clipboard![/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("explain")
def explain_code(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File containing code to explain"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Programming language"),
    from_clipboard: bool = typer.Option(False, "--clipboard", "-c", help="Read code from clipboard"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Explain code in detail"""
    # Get the code from file or clipboard
    if file:
        code = read_file(file)
        language = language or get_language_from_extension(file)
    elif from_clipboard:
        code = paste_from_clipboard()
        if not code:
            console.print("[bold red]No code found in clipboard![/bold red]")
            raise typer.Exit(code=1)
    else:
        # Interactive mode - read from stdin
        console.print("[bold blue]Enter code to explain (Ctrl+D to finish):[/bold blue]")
        code = sys.stdin.read().strip()
        if not code:
            console.print("[bold red]No code provided![/bold red]")
            raise typer.Exit(code=1)
    
    language = language or "python"  # Default to Python if not specified
    
    console.print(f"[bold blue]Explaining {language} code...[/bold blue]")
    
    try:
        explanation = assistant.explain_code(code, language)
        
        # Print the explanation
        console.print("[bold green]Explanation:[/bold green]")
        console.print(explanation)
        
        # Save to file if requested
        if output_file:
            write_file(output_file, explanation)
            console.print(f"[green]Explanation saved to:[/green] {output_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("refactor")
def refactor_code(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File containing code to refactor"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Programming language"),
    focus: str = typer.Option("readability", "--focus", help="What to focus on improving (e.g., readability, performance)"),
    from_clipboard: bool = typer.Option(False, "--clipboard", "-c", help="Read code from clipboard"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    copy: bool = typer.Option(False, "--copy", help="Copy result to clipboard")
):
    """Refactor code to improve a specific aspect"""
    # Get the code from file or clipboard
    if file:
        code = read_file(file)
        language = language or get_language_from_extension(file)
    elif from_clipboard:
        code = paste_from_clipboard()
        if not code:
            console.print("[bold red]No code found in clipboard![/bold red]")
            raise typer.Exit(code=1)
    else:
        # Interactive mode - read from stdin
        console.print(f"[bold blue]Enter code to refactor (focus: {focus}, Ctrl+D to finish):[/bold blue]")
        code = sys.stdin.read().strip()
        if not code:
            console.print("[bold red]No code provided![/bold red]")
            raise typer.Exit(code=1)
    
    language = language or "python"  # Default to Python if not specified
    
    console.print(f"[bold blue]Refactoring {language} code to improve {focus}...[/bold blue]")
    
    try:
        refactored = assistant.refactor_code(code, focus, language)
        
        # Print the refactored code
        console.print("[bold green]Refactored code:[/bold green]")
        print_code(refactored, language)
        
        # Save to file if requested
        if output_file:
            write_file(output_file, refactored)
            console.print(f"[green]Refactored code saved to:[/green] {output_file}")
        
        # Copy to clipboard if requested
        if copy:
            copy_to_clipboard(refactored)
            console.print("[green]Refactored code copied to clipboard![/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("test")
def generate_tests(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File containing code to test"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Programming language"),
    framework: str = typer.Option("pytest", "--framework", help="Testing framework to use"),
    from_clipboard: bool = typer.Option(False, "--clipboard", "-c", help="Read code from clipboard"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Generate unit tests for code"""
    # Get the code from file or clipboard
    if file:
        code = read_file(file)
        language = language or get_language_from_extension(file)
    elif from_clipboard:
        code = paste_from_clipboard()
        if not code:
            console.print("[bold red]No code found in clipboard![/bold red]")
            raise typer.Exit(code=1)
    else:
        # Interactive mode - read from stdin
        console.print(f"[bold blue]Enter code to generate tests for (using {framework}, Ctrl+D to finish):[/bold blue]")
        code = sys.stdin.read().strip()
        if not code:
            console.print("[bold red]No code provided![/bold red]")
            raise typer.Exit(code=1)
    
    language = language or "python"  # Default to Python if not specified
    
    console.print(f"[bold blue]Generating {framework} tests for {language} code...[/bold blue]")
    
    try:
        tests = assistant.generate_tests(code, language, framework)
        
        # Print the tests
        console.print("[bold green]Generated tests:[/bold green]")
        print_code(tests, language)
        
        # Save to file if requested
        if output_file:
            write_file(output_file, tests)
            console.print(f"[green]Tests saved to:[/green] {output_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("docs")
def generate_docs(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File containing code to document"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Programming language"),
    style: str = typer.Option("Google", "--style", help="Documentation style (Google, NumPy, JSDoc, etc.)"),
    from_clipboard: bool = typer.Option(False, "--clipboard", "-c", help="Read code from clipboard"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Generate documentation for code"""
    # Get the code from file or clipboard
    if file:
        code = read_file(file)
        language = language or get_language_from_extension(file)
    elif from_clipboard:
        code = paste_from_clipboard()
        if not code:
            console.print("[bold red]No code found in clipboard![/bold red]")
            raise typer.Exit(code=1)
    else:
        # Interactive mode - read from stdin
        console.print(f"[bold blue]Enter code to document (using {style} style, Ctrl+D to finish):[/bold blue]")
        code = sys.stdin.read().strip()
        if not code:
            console.print("[bold red]No code provided![/bold red]")
            raise typer.Exit(code=1)
    
    language = language or "python"  # Default to Python if not specified
    
    console.print(f"[bold blue]Generating {style} documentation for {language} code...[/bold blue]")
    
    try:
        docs = assistant.generate_docs(code, language, style)
        
        # Print the documentation
        console.print("[bold green]Generated documentation:[/bold green]")
        print_code(docs, language)
        
        # Save to file if requested
        if output_file:
            write_file(output_file, docs)
            console.print(f"[green]Documentation saved to:[/green] {output_file}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("convert")
def convert_code(
    source_language: str = typer.Option(..., "--from", "-f", help="Source programming language"),
    target_language: str = typer.Option(..., "--to", "-t", help="Target programming language"),
    file: Optional[Path] = typer.Option(None, "--file", help="File containing code to convert"),
    from_clipboard: bool = typer.Option(False, "--clipboard", "-c", help="Read code from clipboard"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    copy: bool = typer.Option(False, "--copy", help="Copy result to clipboard")
):
    """Convert code from one language to another"""
    # Get the code from file or clipboard
    if file:
        code = read_file(file)
    elif from_clipboard:
        code = paste_from_clipboard()
        if not code:
            console.print("[bold red]No code found in clipboard![/bold red]")
            raise typer.Exit(code=1)
    else:
        # Interactive mode - read from stdin
        console.print(f"[bold blue]Enter {source_language} code to convert to {target_language} (Ctrl+D to finish):[/bold blue]")
        code = sys.stdin.read().strip()
        if not code:
            console.print("[bold red]No code provided![/bold red]")
            raise typer.Exit(code=1)
    
    console.print(f"[bold blue]Converting code from {source_language} to {target_language}...[/bold blue]")
    
    try:
        converted = assistant.convert_code(code, source_language, target_language)
        
        # Print the converted code
        console.print("[bold green]Converted code:[/bold green]")
        print_code(converted, target_language)
        
        # Save to file if requested
        if output_file:
            write_file(output_file, converted)
            console.print(f"[green]Converted code saved to:[/green] {output_file}")
        
        # Copy to clipboard if requested
        if copy:
            copy_to_clipboard(converted)
            console.print("[green]Converted code copied to clipboard![/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)

@app.command("info")
def show_info():
    """Show information about the Smart Code Assistant"""
    console.print("[bold blue]Smart Code Assistant[/bold blue]")
    console.print("Your AI-powered coding companion")
    console.print("\n[bold green]Available commands:[/bold green]")
    console.print("  generate    - Generate code from a description")
    console.print("  explain     - Explain code in detail")
    console.print("  refactor    - Refactor code to improve specific aspects")
    console.print("  test        - Generate unit tests for code")
    console.print("  docs        - Generate documentation for code")
    console.print("  convert     - Convert code between languages")
    console.print("  info        - Show this information")
    
    console.print("\n[bold green]Configuration:[/bold green]")
    console.print(f"  Model: {config.DEFAULT_MODEL}")
    console.print(f"  Max tokens: {config.MAX_TOKENS}")
    console.print(f"  Temperature: {config.TEMPERATURE}")
    console.print(f"  Supported languages: {', '.join(config.SUPPORTED_LANGUAGES)}")

if __name__ == "__main__":
    app()
```

## 7.5 Project Usage Examples

Let's explore how to use our Smart Code Assistant for various tasks.

### 7.5.1 Generate a Function

```bash
# Generate a binary search function in Python
python main.py generate "implement a binary search algorithm for a sorted list" --language python --requirements "Must handle edge cases like empty lists and include proper documentation"
```

### 7.5.2 Explain Code

```bash
# Explain code from a file
python main.py explain --file complex_algorithm.py

# Explain code from clipboard
python main.py explain --clipboard --language javascript
```

### 7.5.3 Refactor Code

```bash
# Refactor code from a file to improve performance
python main.py refactor --file slow_function.py --focus performance --output improved_function.py

# Refactor code from clipboard to improve readability
python main.py refactor --clipboard --focus readability --language python
```

### 7.5.4 Generate Tests

```bash
# Generate tests for a function in a file
python main.py test --file my_function.py --framework pytest --output test_my_function.py

# Generate tests for code in clipboard
python main.py test --clipboard --language javascript --framework jest
```

### 7.5.5 Generate Documentation

```bash
# Generate documentation for a file
python main.py docs --file undocumented_code.py --style Google --output documented_code.py

# Generate documentation for code in clipboard
python main.py docs --clipboard --language typescript --style JSDoc
```

### 7.5.6 Convert Code Between Languages

```bash
# Convert Python code to JavaScript
python main.py convert --from python --to javascript --file algorithm.py --output algorithm.js

# Convert JavaScript code from clipboard to Python
python main.py convert --from javascript --to python --clipboard --copy
```

## 7.6 Enhancing the Smart Code Assistant

Now that we have the core functionality in place, let's explore some ways to enhance our tool.

### 7.6.1 Adding a Simple Caching Mechanism

To avoid unnecessary API calls and reduce costs, let's implement a simple caching mechanism:

```python
# Add to code_assistant.py

import hashlib
import json
import os
from pathlib import Path
import time

class SimpleCache:
    def __init__(self, cache_dir=None, ttl=3600):
        """Initialize the cache with a directory and time-to-live in seconds"""
        self.cache_dir = Path(cache_dir or config.CACHE_DIR)
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, prompt, model):
        """Create a hash key from the prompt and model"""
        key_str = f"{prompt}:{model}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, key):
        """Get the file path for a cache key"""
        return self.cache_dir / f"{key}.json"
    
    def get(self, prompt, model):
        """Get cached response if available and not expired"""
        key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Check if cache has expired
        if time.time() - cache_path.stat().st_mtime > self.ttl:
            os.remove(cache_path)
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                return cache_data["response"]
        except:
            return None
    
    def set(self, prompt, model, response):
        """Cache a response"""
        key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(key)
        
        cache_data = {
            "prompt": prompt,
            "model": model,
            "response": response,
            "timestamp": time.time()
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
```

Update the `SmartCodeAssistant` class to use the cache:

```python
# Update _send_request method in code_assistant.py

def __init__(self, api_key=None, model=None, use_cache=True):
    # ... existing code ...
    self.use_cache = use_cache
    self.cache = SimpleCache() if use_cache else None

def _send_request(self, prompt, temperature=None, max_tokens=None):
    """Send a request to the OpenAI API with caching"""
    temperature = temperature if temperature is not None else config.TEMPERATURE
    max_tokens = max_tokens if max_tokens is not None else config.MAX_TOKENS
    
    # Try to get from cache if enabled
    if self.use_cache:
        cached_response = self.cache.get(prompt, self.model)
        if cached_response:
            return cached_response
    
    try:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = response.choices[0].message.content
        
        # Store in cache if enabled
        if self.use_cache:
            self.cache.set(prompt, self.model, content)
        
        return content
    except Exception as e:
        raise Exception(f"Error calling OpenAI API: {str(e)}")
```

### 7.6.2 Adding Progressive Enhancement with File Context

Let's enhance our code assistant to consider surrounding file context when processing partial code:

```python
# Add to code_assistant.py

def extract_file_context(self, file_path, target_lines=None, context_lines=5):
    """Extract context from a file around the target lines"""
    with open(file_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    if target_lines is None:
        return "".join(all_lines)
    
    # Convert target_lines to a range if it's a single number
    if isinstance(target_lines, int):
        target_start = max(0, target_lines - 1)
        target_end = target_start + 1
    else:
        target_start = max(0, target_lines[0] - 1)
        target_end = min(len(all_lines), target_lines[1])
    
    # Extract the target code
    target_code = "".join(all_lines[target_start:target_end])
    
    # Get context before target
    context_before_start = max(0, target_start - context_lines)
    context_before = "".join(all_lines[context_before_start:target_start])
    
    # Get context after target
    context_after_end = min(len(all_lines), target_end + context_lines)
    context_after = "".join(all_lines[target_end:context_after_end])
    
    # Compile everything with markers
    result = ""
    if context_before:
        result += "/* Context before target code */\n" + context_before
    
    result += "/* Target code */\n" + target_code
    
    if context_after:
        result += "/* Context after target code */\n" + context_after
    
    return result

def refactor_with_context(self, code, file_path, target_lines, focus="readability", language=None):
    """Refactor code with surrounding file context"""
    if not language:
        language = get_language_from_extension(file_path)
    
    # Extract code with context
    code_with_context = self.extract_file_context(file_path, target_lines, context_lines=5)
    
    prompt = self.prompt_library.get_prompt(
        "refactor_with_context",
        language=language,
        code=code_with_context,
        focus=focus
    )
    
    return self._send_request(prompt)
```

Add the new prompt template to the `PromptLibrary`:

```python
# Add to _initialize_prompts in prompt_library.py

self.prompts["refactor_with_context"] = PromptTemplate(
    """Refactor the target code in the following {language} code to improve its {focus}.
The file contains context before and after the target code to help you understand its purpose.
Only modify the code between the "/* Target code */" markers.

```{language}
{code}
```

Provide ONLY the refactored target code portion and explain what improvements you made.
The surrounding context is for reference only and should not be included in your response.
Focus specifically on improving {focus} while maintaining the same functionality.
""",
    ["language", "code", "focus"]
)
```

### 7.6.3 Adding a Project-Level Assistant

Let's extend our code assistant to understand project-level context:

```python
# Add to code_assistant.py

def analyze_project_structure(self, project_dir, max_files=20, file_extensions=None):
    """Analyze project structure to provide context for code generation"""
    project_path = Path(project_dir)
    if not project_path.exists() or not project_path.is_dir():
        raise ValueError(f"Invalid project directory: {project_dir}")
    
    # Default file extensions to analyze
    if file_extensions is None:
        file_extensions = [".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp"]
    
    # Find relevant files
    all_files = []
    for ext in file_extensions:
        all_files.extend(project_path.glob(f"**/*{ext}"))
    
    # Limit the number of files to analyze
    all_files = all_files[:max_files]
    
    # Extract file names and structures
    project_structure = {
        "project_name": project_path.name,
        "files": [],
        "imports": [],
        "classes": [],
        "functions": []
    }
    
    for file_path in all_files:
        rel_path = file_path.relative_to(project_path)
        
        try:
            content = read_file(file_path)
            
            # Extract high-level info from the file
            file_info = {
                "path": str(rel_path),
                "extension": file_path.suffix,
                "size_bytes": file_path.stat().st_size
            }
            
            project_structure["files"].append(file_info)
            
            # Very simple extraction of Python imports, classes, and functions
            # In a real implementation, use AST parsing or other proper code analysis
            if file_path.suffix == ".py":
                # Simple regex-based extraction
                import re
                
                # Find imports
                imports = re.findall(r'^import\s+(.+?)$|^from\s+(.+?)\s+import', content, re.MULTILINE)
                for imp in imports:
                    imp_name = imp[0] or imp[1]
                    if imp_name:
                        project_structure["imports"].append(imp_name)
                
                # Find classes
                classes = re.findall(r'^class\s+([A-Za-z0-9_]+)', content, re.MULTILINE)
                for cls in classes:
                    project_structure["classes"].append({
                        "name": cls,
                        "file": str(rel_path)
                    })
                
                # Find functions
                functions = re.findall(r'^def\s+([A-Za-z0-9_]+)', content, re.MULTILINE)
                for func in functions:
                    project_structure["functions"].append({
                        "name": func,
                        "file": str(rel_path)
                    })
            
        except Exception as e:
            print(f"Error analyzing file {rel_path}: {e}")
    
    return project_structure

def generate_code_with_project_context(self, description, project_dir, language=None):
    """Generate code with project context"""
    try:
        # Analyze project structure
        project_structure = self.analyze_project_structure(project_dir)
        
        # Determine language from project if not specified
        if language is None:
            # Simple heuristic: use most common language in project
            extensions = [f["extension"] for f in project_structure["files"]]
            if extensions:
                from collections import Counter
                most_common_ext = Counter(extensions).most_common(1)[0][0]
                language = get_language_from_extension(f"file{most_common_ext}")
            else:
                language = "python"  # Default
        
        # Create prompt with project context
        prompt = f"""You are an expert software developer. 
I want you to generate {language} code based on the following description:

{description}

The code will be part of an existing project with the following structure:
Project name: {project_structure['project_name']}
Files: {', '.join(f['path'] for f in project_structure['files'][:10])}

Key classes in the project: {', '.join(cls['name'] for cls in project_structure['classes'][:10])}
Key functions in the project: {', '.join(func['name'] for func in project_structure['functions'][:10])}
Common imports: {', '.join(project_structure['imports'][:10])}

Generate code that follows the style and conventions of this existing project.
Only return the code with minimal explanatory comments.
"""
        
        return self._send_request(prompt, temperature=0.2)
    
    except Exception as e:
        raise Exception(f"Error generating code with project context: {str(e)}")
```

## 7.7 Practical Use Cases

Here are some practical use cases for our Smart Code Assistant:

### 7.7.1 Automating Repetitive Coding Tasks

**Task**: Creating REST API endpoint handlers

```bash
python main.py generate "create a Flask REST API endpoint for user registration that validates email, username, and password" --language python --requirements "Must include input validation, error handling, and follow RESTful principles"
```

**Task**: Generating database models

```bash
python main.py generate "create a SQLAlchemy model for a blog post with title, content, author, publication date, and tags" --language python
```

### 7.7.2 Understanding Legacy Code

**Task**: Explaining complex algorithms

```bash
python main.py explain --file legacy_algorithm.py
```

**Task**: Documenting undocumented functions

```bash
python main.py docs --file undocumented_module.py --style Google
```

### 7.7.3 Improving Code Quality

**Task**: Refactoring for performance

```bash
python main.py refactor --file slow_function.py --focus performance
```

**Task**: Creating unit tests for existing code

```bash
python main.py test --file data_processor.py --framework pytest
```

### 7.7.4 Cross-Language Development

**Task**: Converting Python utility to JavaScript

```bash
python main.py convert --from python --to javascript --file utils.py --output utils.js
```

## 7.8 Best Practices and Limitations

### 7.8.1 Best Practices

1. **Always review the generated code**: While LLMs can provide good starting points, always review the code for correctness, security issues, and alignment with your needs.

2. **Break down complex tasks**: For better results, break complex coding tasks into smaller, more manageable pieces.

3. **Provide clear requirements**: The more specific your descriptions and requirements are, the better the generated code will be.

4. **Use project context**: When working on existing projects, providing project-level context will help generate more consistent and compatible code.

5. **Cache responses**: To reduce API costs and improve response times, implement caching for frequently requested tasks.

### 7.8.2 Limitations

1. **Code accuracy**: LLMs may generate code with logical errors or incorrect implementations, especially for complex algorithms.

2. **Security considerations**: Generated code might contain security vulnerabilities, so always review it carefully.

3. **Context limits**: LLMs have context window limitations, so they might struggle with understanding very large codebases or files.

4. **Language limitations**: Performance varies across programming languages, with better results typically for popular languages like Python and JavaScript.

5. **API costs**: Extensive use of LLMs can incur significant API costs, so monitor usage carefully.

## 7.9 Future Enhancements

Our Smart Code Assistant is just the beginning. Here are some potential future enhancements:

1. **IDE integration**: Develop plugins for popular IDEs like VS Code, PyCharm, and IntelliJ.

2. **More advanced project understanding**: Implement deeper static analysis of project structures and coding patterns.

3. **Code review capabilities**: Add features to review code changes and suggest improvements.

4. **Custom fine-tuning**: Train models on specific codebases to better match company coding styles and patterns.

5. **Collaborative features**: Allow teams to share and rate prompt templates and responses.

6. **Version control integration**: Integrate with Git to understand code history and changes over time.

## 7.10 Conclusion

In this chapter, we've built a practical Smart Code Assistant that demonstrates how prompt engineering can be applied to solve real-world coding challenges. By leveraging LLMs, we've created a tool that can generate code, explain existing code, refactor for improvements, create tests, and assist with documentation.

The key takeaway is that effective prompt engineering allows us to guide LLMs to produce valuable coding assistance. By structuring prompts with clear instructions, relevant context, and specific requirements, we can obtain high-quality results across a range of coding tasks.

As you continue your prompt engineering journey, consider how you might extend and customize this tool for your specific development needs. The techniques and patterns demonstrated here can be applied to a wide range of software development tasks beyond what we've covered.

In the next chapter, we'll build on these skills to create another practical application: an LLM-powered ML Model Explainer and Debugger.
