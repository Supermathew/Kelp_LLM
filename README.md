# Kelp_LLM

## Article Analyzer & Structured Categorizer

##  Installation

### Prerequisites

1. **Python 3.8+** must be installed
2. **Google Chrome** must be installed
3. **ChromeDriver** must be in your system PATH (see below)

---

### Clone the Repository

```bash
git clone https://github.com/Supermathew/Kelp_LLM.git
cd Kelp_LLM/src
````

---

###  Create Virtual Environment (Windows)

```bash
python -m venv kelpenv
kelpenv\Scripts\activate
```

---

###  Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ⚙️ ChromeDriver Setup (Windows)

#### Option 1: Automatically Managed by Selenium 4.x ✅

If using Selenium v4+, it automatically downloads and manages `ChromeDriver`.

#### Option 2: Manual Installation

1. Download from: [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
2. Extract `chromedriver.exe` and place it in a folder.
3. Add that folder to your system PATH (Environment Variables → Path).

---

###  API Key Setup (Windows)

1. **Get API Key**:
   [Google AI Studio](https://makersuite.google.com/app/apikey)

2. **Set Environment Variable**:

#### Option 1: Temporary (current session only)

```cmd
set GEMINI_API_KEY=your_api_key_here
```
#### Option 2: Create .env file

1. create a .env file in the same file directory and add variable GEMINI_API_KEY="value"
---

#### Option 3: Permanent (recommended)

1. Search for "Environment Variables" in Windows
2. Add `GEMINI_API_KEY` under **User variables** with your API key

---

## Usage

### Basic Example

```bash
python analyzer.py
```

### Process URLs from File

```bash
python analyzer.py --file urls.txt
```

Create `urls.txt` file like this:

```
https://www.example.com
https://www.company.com/about
https://www.news-site.com/article
```

---

### Advanced Options

```bash
# Save results to file
python analyzer.py --file urls.txt --output results.json

# Use multiple workers
python analyzer.py --file urls.txt --workers 4

# Help
python analyzer.py --help
```

---

## Output Format

```json
{
  "URL": "https://example.com",
  "site_type": "company website",
  "extracted_web_content": "Full cleaned text content...",
  "content": [
    {
      "About Us": {
        "links": ["https://example.com/about"],
        "text": "We are a tech company..."
      }
    }
  ],
  "errors": null
}
```

* Results saved to `analysis_results.json` (or custom output file)

---

## Implementation Details

### Content Extraction Strategy

* JavaScript Rendering: Uses Selenium Chrome WebDriver to execute JavaScript and wait for dynamic content
* Cleans up navigation, ads, popups, etc.
* Main Content Identification: Multi-strategy approach:

* Semantic HTML5 elements
* Common content class names
* Text density analysis
* Fallback to body content

## LLM Integration

The tool makes two separate API calls to the Gemini LLM:

1. **Category Identification**: Analyzes content to identify present categories
2. **Site Type Classification**: Determines the overall website type

### Cost Optimization Strategies

- Content truncation for category identification (3000 chars)
- we can also summarize the content we have extracted from the website using any python package or using the summarizer model and summarize into less words and use it for LLM gemini process.
- Minimal content for site type identification (1000 chars)
- Structured prompts for consistent JSON output
- Retry logic with exponential backoff

## Content Splitting Heuristic

The Python script uses sophisticated heuristics to extract category-specific text:

1. **Keyword Matching**: Each category has predefined keywords
2. **Sentence Analysis**: Finds sentences containing category keywords
3. **Section Header Detection**: Identifies content sections by headers
4. **Paragraph Analysis**: Fallback to relevant paragraphs

### Limitations

- Relies on keyword presence, may miss contextual content
- May not handle complex layouts or unconventional content organization
- Text extraction quality depends on HTML structure

## Link Association Heuristic

Links are associated with categories based on:

1. **URL Path Analysis**: Checks if URL contains category keywords
2. **Anchor Text Matching**: Analyzes visible link text (when available)
3. **Category-Specific Rules**: Special handling for common patterns

### Limitations

- May miss semantically related links without obvious keywords
- Cannot analyze link destinations or anchor text from JavaScript-generated links
- Heuristic-based approach may have false positives/negatives

## Error Handling

- **Network Errors**: Timeout handling, retry logic
- **WebDriver Issues**: Browser automation error recovery
- **API Failures**: Exponential backoff, graceful degradation
- **Content Parsing**: Fallback extraction methods
- **JSON Parsing**: Robust handling of malformed LLM responses

##  Troubleshooting (Windows)

### ChromeDriver Not Found

* Ensure it's installed and available in PATH
* Or use Selenium 4.x to auto-manage

### API Key Not Detected

```cmd
echo %GEMINI_API_KEY%
```

---

##  Limitations

* JavaScript-heavy pages may fail
* Only English language content supported
* Heuristics may miss certain semantic categories

---


