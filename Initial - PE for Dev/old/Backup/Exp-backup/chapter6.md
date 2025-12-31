# Chapter 6: Building Effective Developer Tooling for LLM Applications

In the previous chapters, we've explored the fundamentals of prompt engineering and various techniques to create effective prompts. Now, it's time to take our skills to the next level by implementing robust developer tooling for LLM applications. As LLMs become integral parts of modern software systems, proper tooling becomes essential for maintainability, scalability, and reliability.

## 6.1 Prompt Libraries and Reuse Patterns

### 6.1.1 The Need for Prompt Management

As your project grows, managing prompts becomes increasingly challenging. Without proper organization, you might face:

- Duplicate prompts across different parts of your application
- Inconsistent prompting styles and formats
- Difficulty in tracking which prompts work best for specific tasks
- Challenges in version control and prompt evolution

### 6.1.2 Building a Prompt Library

Let's create a simple yet effective prompt library in Python:

```python
# A basic prompt library implementation

class PromptTemplate:
    def __init__(self, template, required_variables=None):
        self.template = template
        self.required_variables = required_variables or []
    
    def format(self, **kwargs):
        # Check if all required variables are provided
        missing_vars = [var for var in self.required_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
        
        # Format the template with the provided variables
        return self.template.format(**kwargs)

class PromptLibrary:
    def __init__(self):
        self.prompts = {}
    
    def add_prompt(self, name, template, required_variables=None):
        self.prompts[name] = PromptTemplate(template, required_variables)
        
    def get_prompt(self, name, **kwargs):
        if name not in self.prompts:
            raise KeyError(f"Prompt '{name}' not found in the library")
        return self.prompts[name].format(**kwargs)
```

Usage example:

```python
# Initialize the library
prompt_lib = PromptLibrary()

# Add prompts with templates
prompt_lib.add_prompt(
    "code_explanation",
    "Explain the following {language} code:\n\n```{language}\n{code}\n```\n\nProvide a detailed explanation including:",
    ["language", "code"]
)

prompt_lib.add_prompt(
    "bug_fix",
    "Fix the following {language} code that has a bug:\n\n```{language}\n{code}\n```\n\nError message: {error}\n\nProvide the corrected code and explain the fix.",
    ["language", "code", "error"]
)

# Use the prompt template
python_code = "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"
formatted_prompt = prompt_lib.get_prompt("code_explanation", language="python", code=python_code)
```

### 6.1.3 Advanced Prompt Organization Patterns

For larger projects, consider organizing prompts hierarchically:

```python
# Domain-specific prompt libraries
class CodePromptLibrary(PromptLibrary):
    def __init__(self):
        super().__init__()
        self._initialize_code_prompts()
    
    def _initialize_code_prompts(self):
        self.add_prompt("generate_function", "Write a {language} function that {requirement}.", ["language", "requirement"])
        self.add_prompt("optimize_code", "Optimize the following {language} code for {optimization_goal}:\n\n```{language}\n{code}\n```", ["language", "optimization_goal", "code"])
        # More code-related prompts...
```

## 6.2 Debugging Tools for LLM Applications

### 6.2.1 Prompt Debugging

Debugging LLM applications presents unique challenges compared to traditional software. Let's implement a simple prompt debugger:

```python
import json
from datetime import datetime

class PromptDebugger:
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.history = []
    
    def log_interaction(self, prompt, response, metadata=None):
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {}
        }
        self.history.append(interaction)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(interaction) + "\n")
        
        return interaction
    
    def analyze_token_usage(self, interaction):
        if 'token_usage' in interaction['metadata']:
            usage = interaction['metadata']['token_usage']
            return f"Prompt tokens: {usage.get('prompt_tokens', 'N/A')}, " \
                   f"Completion tokens: {usage.get('completion_tokens', 'N/A')}, " \
                   f"Total tokens: {usage.get('total_tokens', 'N/A')}"
        return "Token usage data not available"
    
    def compare_interactions(self, interaction1_idx, interaction2_idx):
        if interaction1_idx >= len(self.history) or interaction2_idx >= len(self.history):
            return "Invalid interaction indices"
        
        int1 = self.history[interaction1_idx]
        int2 = self.history[interaction2_idx]
        
        # Compare prompts
        prompt_diff = self._simple_diff(int1['prompt'], int2['prompt'])
        
        # Compare responses (simplified)
        response_similarity = self._calculate_similarity(int1['response'], int2['response'])
        
        return {
            "prompt_differences": prompt_diff,
            "response_similarity": f"{response_similarity:.2f}%"
        }
    
    def _simple_diff(self, text1, text2):
        # A very simple diff implementation
        # In a real application, use a proper diff library
        if text1 == text2:
            return "No differences"
        
        # Basic character-by-character comparison
        diffs = []
        for i, (c1, c2) in enumerate(zip(text1, text2)):
            if c1 != c2:
                diffs.append(f"Pos {i}: '{c1}' vs '{c2}'")
                
        if len(text1) != len(text2):
            diffs.append(f"Length difference: {len(text1)} vs {len(text2)}")
            
        return diffs[:10]  # Only show first 10 differences
    
    def _calculate_similarity(self, text1, text2):
        # Simple similarity calculation
        # In a real application, use a more sophisticated algorithm
        common_chars = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
        total_length = max(len(text1), len(text2))
        return (common_chars / total_length) * 100 if total_length > 0 else 100
```

Example usage with OpenAI's API:

```python
import openai

debugger = PromptDebugger(log_file="llm_debug_log.jsonl")

def query_llm(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    content = response.choices[0].message.content
    
    # Log the interaction with metadata
    debugger.log_interaction(
        prompt=prompt,
        response=content,
        metadata={
            "model": model,
            "token_usage": response.usage._asdict() if hasattr(response, "usage") else None,
            "finish_reason": response.choices[0].finish_reason
        }
    )
    
    return content
```

### 6.2.2 Visualizing LLM Behavior

To understand how changes in prompts affect LLM responses, visualization tools can be invaluable:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_token_usage(debugger, last_n=10):
    """Visualize token usage for the last N interactions"""
    if len(debugger.history) == 0:
        return "No history available"
    
    # Get data for the last n interactions
    history = debugger.history[-last_n:]
    
    prompt_tokens = []
    completion_tokens = []
    labels = []
    
    for i, interaction in enumerate(history):
        metadata = interaction.get('metadata', {})
        usage = metadata.get('token_usage', {})
        
        prompt_tokens.append(usage.get('prompt_tokens', 0))
        completion_tokens.append(usage.get('completion_tokens', 0))
        labels.append(f"Query {i+1}")
    
    # Create stacked bar chart
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(labels, prompt_tokens, width, label='Prompt Tokens')
    ax.bar(labels, completion_tokens, width, bottom=prompt_tokens, label='Completion Tokens')
    
    ax.set_ylabel('Token Count')
    ax.set_title('Token Usage by Query')
    ax.legend()
    
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig('token_usage.png')
    plt.close()
    
    return "Token usage visualization saved as 'token_usage.png'"
```

## 6.3 Performance Profiling and Optimization

### 6.3.1 Measuring LLM Application Performance

Performance in LLM applications involves several metrics:

```python
import time
import statistics
from functools import wraps

class LLMProfiler:
    def __init__(self):
        self.metrics = {
            "latency": [],
            "token_throughput": [],
            "success_rate": {"success": 0, "failure": 0},
            "cost": []
        }
    
    def profile_request(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            response = None
            
            try:
                response = func(*args, **kwargs)
                self.metrics["success_rate"]["success"] += 1
            except Exception as e:
                error = e
                self.metrics["success_rate"]["failure"] += 1
                
            end_time = time.time()
            latency = end_time - start_time
            self.metrics["latency"].append(latency)
            
            # Calculate token throughput if possible
            if response and hasattr(response, "usage"):
                total_tokens = response.usage.total_tokens
                tokens_per_second = total_tokens / latency if latency > 0 else 0
                self.metrics["token_throughput"].append(tokens_per_second)
                
                # Calculate approximate cost (example for GPT-3.5-turbo)
                # Rates as of 2023, adjust as needed
                prompt_cost = response.usage.prompt_tokens * 0.0015 / 1000  # $0.0015 per 1K tokens
                completion_cost = response.usage.completion_tokens * 0.002 / 1000  # $0.002 per 1K tokens
                total_cost = prompt_cost + completion_cost
                self.metrics["cost"].append(total_cost)
            
            if error:
                raise error
                
            return response
            
        return wrapper
    
    def get_summary(self):
        latency_stats = {
            "min": min(self.metrics["latency"]) if self.metrics["latency"] else None,
            "max": max(self.metrics["latency"]) if self.metrics["latency"] else None,
            "avg": statistics.mean(self.metrics["latency"]) if self.metrics["latency"] else None,
            "p95": self._percentile(self.metrics["latency"], 95),
            "p99": self._percentile(self.metrics["latency"], 99)
        }
        
        throughput_stats = {
            "avg": statistics.mean(self.metrics["token_throughput"]) if self.metrics["token_throughput"] else None
        }
        
        success_rate = (
            self.metrics["success_rate"]["success"] / 
            (self.metrics["success_rate"]["success"] + self.metrics["success_rate"]["failure"])
            if (self.metrics["success_rate"]["success"] + self.metrics["success_rate"]["failure"]) > 0
            else 0
        ) * 100
        
        total_cost = sum(self.metrics["cost"])
        
        return {
            "latency_ms": {k: v*1000 if v is not None else None for k, v in latency_stats.items()},
            "throughput_tokens_per_sec": throughput_stats,
            "success_rate_percent": success_rate,
            "total_cost_usd": total_cost,
            "request_count": len(self.metrics["latency"])
        }
    
    def _percentile(self, data, percentile):
        if not data:
            return None
        sorted_data = sorted(data)
        index = int(len(sorted_data) * (percentile / 100))
        return sorted_data[index]
```

### 6.3.2 Optimizing Prompt Performance

We can optimize prompts in several ways:

1. **Prompt Compression Techniques:**

```python
def compress_prompt(prompt, max_length=None):
    """Compress a prompt by removing redundancies while preserving meaning"""
    # Simple compression techniques
    compressed = prompt
    
    # Remove redundant whitespace
    compressed = " ".join(compressed.split())
    
    # Replace common verbose phrases
    replacements = {
        "Please provide a detailed explanation of": "Explain",
        "I would like you to": "",
        "It would be great if you could": "",
        "Can you please": "",
    }
    
    for verbose, concise in replacements.items():
        compressed = compressed.replace(verbose, concise)
    
    # If a maximum length is specified, truncate while preserving key instructions
    if max_length and len(compressed) > max_length:
        # This is a simplistic approach - a real implementation would be more sophisticated
        lines = compressed.split('. ')
        result = []
        current_length = 0
        
        # Always include the first line (assumed to contain the main instruction)
        result.append(lines[0])
        current_length += len(lines[0])
        
        # Add as many additional lines as fit within max_length
        for line in lines[1:]:
            if current_length + len(line) + 2 <= max_length:  # +2 for the '. '
                result.append(line)
                current_length += len(line) + 2
            else:
                break
                
        compressed = '. '.join(result)
        if not compressed.endswith('.'):
            compressed += '.'
    
    return compressed
```

2. **Caching LLM Responses:**

```python
import hashlib
import json
import os
import pickle

class LLMResponseCache:
    def __init__(self, cache_dir="llm_cache", ttl_seconds=86400):
        """Initialize the cache with a directory and time-to-live"""
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt, model, temperature):
        """Create a unique cache key from the request parameters"""
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, key):
        """Get the file path for a cache key"""
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get(self, prompt, model, temperature=0):
        """Retrieve a response from the cache if it exists and is still valid"""
        key = self._get_cache_key(prompt, model, temperature)
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
            
        # Check if cache has expired
        cache_age = time.time() - os.path.getmtime(cache_path)
        if cache_age > self.ttl_seconds:
            os.remove(cache_path)  # Remove expired cache
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def set(self, prompt, model, temperature, response):
        """Store a response in the cache"""
        key = self._get_cache_key(prompt, model, temperature)
        cache_path = self._get_cache_path(key)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(response, f)
```

Example usage of caching:

```python
cache = LLMResponseCache()

def query_llm_with_cache(prompt, model="gpt-3.5-turbo", temperature=0):
    # Try to get from cache first
    cached_response = cache.get(prompt, model, temperature)
    if cached_response:
        print("Cache hit!")
        return cached_response
    
    # If not in cache, make the API call
    print("Cache miss, calling API...")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    
    # Store in cache for future use
    cache.set(prompt, model, temperature, response)
    
    return response
```

## 6.4 Integration with Existing Development Workflows

### 6.4.1 Command Line Tools for Prompt Engineering

Creating a simple CLI tool for prompt engineering:

```python
#!/usr/bin/env python
import argparse
import sys
import json
import openai
from pathlib import Path

def setup_argparser():
    parser = argparse.ArgumentParser(description="LLM Prompt Engineering CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Query LLM command
    query_parser = subparsers.add_parser("query", help="Query an LLM with a prompt")
    query_parser.add_argument("--prompt", "-p", help="The prompt to send")
    query_parser.add_argument("--prompt-file", "-f", help="File containing the prompt")
    query_parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="LLM model to use")
    query_parser.add_argument("--output", "-o", help="Save output to file")
    query_parser.add_argument("--temperature", "-t", type=float, default=0, help="Temperature setting")
    
    # Test prompt variations command
    test_parser = subparsers.add_parser("test-variations", help="Test different prompt variations")
    test_parser.add_argument("--variations-file", required=True, help="JSON file with prompt variations")
    test_parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="LLM model to use")
    test_parser.add_argument("--output-dir", "-o", default="./results", help="Directory to save results")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Process a batch of prompts")
    batch_parser.add_argument("--batch-file", required=True, help="JSON file with prompts to process")
    batch_parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="LLM model to use")
    batch_parser.add_argument("--output-dir", "-o", default="./results", help="Directory to save results")
    
    return parser

def main():
    parser = setup_argparser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize OpenAI API (assuming OPENAI_API_KEY environment variable is set)
    if not openai.api_key:
        print("Error: OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    if args.command == "query":
        handle_query_command(args)
    elif args.command == "test-variations":
        handle_test_variations_command(args)
    elif args.command == "batch":
        handle_batch_command(args)

def handle_query_command(args):
    # Get prompt from arguments or file
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read()
    else:
        print("Error: Either --prompt or --prompt-file must be specified")
        sys.exit(1)
    
    # Query the LLM
    response = openai.ChatCompletion.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature
    )
    
    # Process the response
    content = response.choices[0].message.content
    
    # Output handling
    if args.output:
        with open(args.output, 'w') as f:
            f.write(content)
        print(f"Response saved to {args.output}")
    else:
        print("\n--- Response ---\n")
        print(content)
        print("\n---------------\n")
    
    # Print usage statistics
    print(f"Token usage: {response.usage.total_tokens} tokens")
    print(f"  - Prompt: {response.usage.prompt_tokens} tokens")
    print(f"  - Completion: {response.usage.completion_tokens} tokens")

# Additional handler functions omitted for brevity

if __name__ == "__main__":
    main()
```

### 6.4.2 Integrating with VS Code Extensions

For VS Code integration, consider creating a simple extension that enables developers to interact with LLMs directly from their editor.

Key features might include:
- Prompt templates accessible via snippets
- Highlighted code selection to LLM processing
- Preview window for LLM responses
- Context-aware suggestions based on the current file

## 6.5 Testing Frameworks for LLM-Powered Features

### 6.5.1 Unit Testing LLM Prompts

Creating a testing framework for prompts:

```python
import unittest
from unittest.mock import patch
import json

class PromptTest(unittest.TestCase):
    """Base class for testing prompt templates and their expected responses"""
    
    def setUp(self):
        # Set up mock for OpenAI API
        self.openai_patcher = patch('openai.ChatCompletion.create')
        self.mock_openai = self.openai_patcher.start()
    
    def tearDown(self):
        self.openai_patcher.stop()
        
    def assert_prompt_contains(self, prompt, required_elements):
        """Assert that a prompt contains all required elements"""
        for element in required_elements:
            self.assertIn(element, prompt, f"Prompt should contain '{element}'")
    
    def assert_prompt_format(self, prompt, expected_format):
        """Assert that a prompt follows the expected format structure"""
        # This is a simplified check - real implementation would be more sophisticated
        sections = expected_format.split("[section]")
        last_pos = 0
        
        for section in sections[1:]:  # Skip the first empty section
            section = section.strip()
            pos = prompt.find(section, last_pos)
            self.assertGreater(pos, -1, f"Prompt missing expected section: '{section}'")
            last_pos = pos + len(section)
    
    def mock_llm_response(self, response_content, usage=None):
        """Helper to set up a mock LLM response"""
        if usage is None:
            usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            
        # Create a response object structure similar to OpenAI's
        response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {'content': response_content}),
                    'finish_reason': 'stop'
                })
            ],
            'usage': type('obj', (object,), usage),
            'model': 'gpt-3.5-turbo'
        })
        
        self.mock_openai.return_value = response

# Example test class
class TestCodeGenerationPrompts(PromptTest):
    def test_python_function_prompt(self):
        from my_prompt_lib import get_function_generation_prompt
        
        # Test specific prompt generation
        prompt = get_function_generation_prompt(
            language="python",
            function_name="calculate_discount",
            description="Calculate the final price after applying a discount percentage",
            parameters=["price", "discount_percentage"]
        )
        
        # Verify prompt structure
        self.assert_prompt_contains(prompt, ["python", "calculate_discount", "price", "discount_percentage"])
        self.assert_prompt_format(prompt, "[section]Task[section]Parameters[section]Requirements")
        
        # Mock the LLM response
        expected_code = "def calculate_discount(price, discount_percentage):\n    return price * (1 - discount_percentage / 100)"
        self.mock_llm_response(expected_code)
        
        # Test the full flow from prompt to response
        from my_llm_service import generate_code
        response = generate_code(prompt)
        
        # Verify the response handling
        self.assertEqual(response, expected_code)
```

### 6.5.2 Integration Testing for LLM Applications

For integration tests:

```python
class LLMIntegrationTest(unittest.TestCase):
    """Base class for integration testing of LLM-powered features"""
    
    def setUp(self):
        # Real API calls but with a special test API key
        # Could use a staging/test environment for the API
        import os
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_TEST_API_KEY")
        
        # Set lower temperature for more consistent results in tests
        self.default_test_params = {
            "temperature": 0.0,
            "max_tokens": 100  # Limit tokens for faster tests
        }
    
    def assert_response_matches_criteria(self, response, criteria):
        """Assert that an LLM response meets a set of criteria"""
        for criterion, expected in criteria.items():
            if criterion == "contains":
                for phrase in expected:
                    self.assertIn(phrase, response, f"Response should contain '{phrase}'")
            elif criterion == "excludes":
                for phrase in expected:
                    self.assertNotIn(phrase, response, f"Response should not contain '{phrase}'")
            elif criterion == "length_range":
                min_len, max_len = expected
                self.assertTrue(min_len <= len(response) <= max_len, 
                               f"Response length {len(response)} outside range {min_len}-{max_len}")
            # Add more criteria types as needed
```

## 6.6 Cost Optimization Techniques

### 6.6.1 Token Counting and Budget Management

Implement a token budget manager:

```python
import tiktoken

class TokenBudgetManager:
    """Manages token usage and budgets for LLM applications"""
    
    def __init__(self, model_name="gpt-3.5-turbo", monthly_budget=None):
        self.model_name = model_name
        self.monthly_budget = monthly_budget
        self.encoding = tiktoken.encoding_for_model(model_name)
        
        # Cost per 1K tokens (adjust based on current pricing)
        self.cost_per_1k_tokens = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06}
        }.get(model_name, {"input": 0.0015, "output": 0.002})
        
        self.current_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0.0
        }
    
    def count_tokens(self, text):
        """Count the number of tokens in a text string"""
        if not text:
            return 0
        token_ids = self.encoding.encode(text)
        return len(token_ids)
    
    def estimate_cost(self, input_text, estimated_output_length=None):
        """Estimate the cost of an LLM request"""
        input_tokens = self.count_tokens(input_text)
        
        # If output length not provided, estimate based on input length
        if estimated_output_length is None:
            estimated_output_tokens = input_tokens * 1.5  # Simple heuristic
        else:
            estimated_output_tokens = self.count_tokens(estimated_output_length)
        
        input_cost = (input_tokens / 1000) * self.cost_per_1k_tokens["input"]
        output_cost = (estimated_output_tokens / 1000) * self.cost_per_1k_tokens["output"]
        
        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "total_estimated_cost": input_cost + output_cost
        }
    
    def track_usage(self, input_text, output_text):
        """Track actual token usage and cost"""
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        
        input_cost = (input_tokens / 1000) * self.cost_per_1k_tokens["input"]
        output_cost = (output_tokens / 1000) * self.cost_per_1k_tokens["output"]
        total_cost = input_cost + output_cost
        
        # Update running totals
        self.current_usage["input_tokens"] += input_tokens
        self.current_usage["output_tokens"] += output_tokens
        self.current_usage["total_cost"] += total_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "running_totals": self.current_usage.copy()
        }
    
    def check_budget_status(self):
        """Check status against monthly budget"""
        if self.monthly_budget is None:
            return {"has_budget": False}
            
        remaining_budget = self.monthly_budget - self.current_usage["total_cost"]
        usage_percentage = (self.current_usage["total_cost"] / self.monthly_budget) * 100
        
        return {
            "has_budget": True,
            "monthly_budget": self.monthly_budget,
            "current_usage": self.current_usage["total_cost"],
            "remaining_budget": remaining_budget,
            "usage_percentage": usage_percentage,
            "status": "OK" if usage_percentage < 90 else "WARNING" if usage_percentage < 100 else "EXCEEDED"
        }
```

### 6.6.2 Implementing Smart Caching

We've already covered a basic caching implementation earlier. Here's an enhancement with smarter invalidation strategies:

```python
class SemanticCache:
    """A cache that uses semantic similarity to match similar prompts"""
    
    def __init__(self, embedding_model="text-embedding-ada-002", similarity_threshold=0.95):
        self.cache = {}  # Maps embedding hash to (response, original_prompt)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
    
    def _get_embedding(self, text):
        """Get embedding vector for text"""
        response = openai.Embedding.create(
            model=self.embedding_model,
            input=text
        )
        return response["data"][0]["embedding"]
    
    def _compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between embeddings"""
        import numpy as np
        
        # Convert to numpy arrays for vector operations
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    def get(self, prompt, model):
        """Try to retrieve a cached response based on semantic similarity"""
        try:
            prompt_embedding = self._get_embedding(prompt)
            
            best_match = None
            highest_similarity = 0
            
            for cache_key, (cached_response, original_prompt, cached_model) in self.cache.items():
                # Skip if models don't match
                if model != cached_model:
                    continue
                    
                # Calculate similarity with the cached prompt
                similarity = self._compute_similarity(prompt_embedding, cache_key)
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = (cached_response, original_prompt, similarity)
            
            # Return the best match if it meets the threshold
            if best_match and highest_similarity >= self.similarity_threshold:
                return {"response": best_match[0], 
                        "original_prompt": best_match[1],
                        "similarity": highest_similarity}
                
            return None
        
        except Exception as e:
            print(f"Error in semantic cache: {e}")
            return None
    
    def set(self, prompt, response, model):
        """Store a response in the cache"""
        try:
            prompt_embedding = self._get_embedding(prompt)
            # Use the embedding vector as key
            embedding_key = tuple(prompt_embedding)  # Convert to tuple so it's hashable
            self.cache[embedding_key] = (response, prompt, model)
        except Exception as e:
            print(f"Error setting semantic cache: {e}")
```

## 6.7 Conclusion

In this chapter, we've explored various tools and techniques for building effective developer tooling for LLM applications. From prompt libraries and reuse patterns to debugging tools, performance optimization, testing frameworks, and cost management, we've covered the essential components needed to develop robust LLM-powered applications.

As you implement these tools in your projects, remember that the field of prompt engineering is rapidly evolving. Stay flexible and be prepared to adapt your tools and approaches as new best practices emerge. In the next chapter, we'll put these tools into practice with a hands-on project building a Smart Code Assistant.

## 6.8 Further Reading

- "Design Patterns for LLM Applications" by Various Authors
- "Efficient Natural Language Processing" by Various Authors
- "Software Engineering for AI-Powered Systems" by Various Authors
- OpenAI API Documentation
- LangChain and LlamaIndex Documentation for advanced tooling options
