import os
import json
import logging
import requests
import asyncio
import aiohttp
import re
from bs4 import BeautifulSoup, Comment
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Optional, Tuple, Any
from dotenv import load_dotenv
import argparse
import asyncio
import json
import logging
import os
import sys
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleAnalyzer:
    ## This is the class that will initially the api_key, max workers that will concurently resolve and fetch the urls, and all construct other process like LLM and the content extraction.
    def __init__(self, api_key: str, max_workers: int = 3):
        self.api_key = api_key
        self.max_workers = max_workers
        self.llm_processor = LLMProcessor(api_key=self.api_key)
        self.content_processor = ContentProcessor()

    async def load_urls_from_file(self, file_path: str) -> List[str]:
        ## this will read the file and extract the urls present in the file##
        try:
            with open(file_path, 'r') as file:
                return [line.strip() for line in file if line.strip()]
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return []

    @property
    def default_urls(self) -> List[str]:
        """Default URLs for testing."""
        return [
            "https://ramkrishnaforgings.com"
        ]

    async def process_url(self, url: str, semaphore: asyncio.Semaphore) -> Dict:
        ## This will process the single url one by one##
        async with semaphore:
            if validate_url(url):
                return await process_url(url, self.llm_processor, self.content_processor)
            else:
                return {
                    "URL": url,
                    "site_type": "other",
                    "extracted_web_content": "",
                    "content": [],
                    "errors": "Invalid or unreachable URL"
                }

    async def process_urls(self, urls: List[str]) -> List[Dict]:
        ##This will process the url concurrently and compute the result##
        semaphore = asyncio.Semaphore(self.max_workers)
        tasks = [self.process_url(url, semaphore) for url in urls]
        return await asyncio.gather(*tasks)

    async def cleanup(self):
        ## This will clean by the resourses and release it##
        await self.llm_processor.cleanup()

class LLMProcessor:
    ## This class is responsible for LLMprocess for making api call to gemini ##
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        self.session = None

        self.category_keywords = {
            "About Us": ["about", "company", "mission", "vision", "story", "history", "overview"],
            "Products & Services": ["product", "service", "solution", "offering", "feature", "technology"],
            "Leadership/Team": ["team", "leadership", "staff", "employee", "founder", "ceo", "management"],
            "Blog/News/Press Release": ["blog", "news", "press", "article", "announcement", "update"],
            "Contact/Support": ["contact", "support", "help", "phone", "email", "address", "location"],
            "Privacy/Legal": ["privacy", "legal", "terms", "policy", "compliance", "gdpr"],
            "Careers/Jobs": ["career", "job", "hiring", "employment", "work", "position", "vacancy"],
            "Other": []
        }

    async def _make_api_call(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        ## This will make the api call to the gemini and compute the result##
        if not self.session:
            self.session = aiohttp.ClientSession()

        headers = {
            'Content-Type': 'application/json',
        }

        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            }
        }

        url = f"{self.base_url}?key={self.api_key}"

        for attempt in range(max_retries):
                try:
                    async with self.session.post(url, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            if 'candidates' in result and result['candidates']:
                                content = result['candidates'][0]['content']['parts'][0]['text']
                                return {"content": content}
                        else:
                            error_text = await response.text()
                            logger.warning(f"API call failed with status {response.status}, response: {error_text}, attempt {attempt + 1}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    logger.error(f"API call error on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                return None

    def _create_category_identification_prompt(self, content: str) -> str:
        ## The content which is passed from the parament we are only selecting the first 3000 part and performing computing on it##
        truncated_content = content[:3000] if len(content) > 3000 else content

        prompt = f"""
Analyze the following web content and identify which categories from the predefined list are present.
Only include categories that have clear, relevant content in the text.

Predefined categories:
- About Us
- Products & Services
- Leadership/Team
- Blog/News/Press Release
- Contact/Support
- Privacy/Legal
- Careers/Jobs
- Other (only if significant content doesn't fit other categories)

Content to analyze:
{truncated_content}

Return ONLY a JSON array with the identified categories in this exact format:
[
    {{"category_name": "About Us", "text": ""}},
    {{"category_name": "Products & Services", "text": ""}}
]

Important:
- Only include categories that are actually present in the content
- The "text" field should be empty - it will be populated later
- Return valid JSON only, no additional text or explanations
"""
        return prompt

    def _create_site_type_prompt(self, content: str) -> str:
        truncated_content = content[:1000] if len(content) > 1000 else content

        prompt = f"""
Based on the following web content, determine the type of website.

Content:
{truncated_content}

Return ONLY a JSON object with the site type in this exact format:
{{"site_type": "TYPE"}}

Where TYPE must be one of: news, blog, e-commerce, company website, educational, forum, portfolio, other

Important: Return valid JSON only, no additional text or explanations.
"""
        return prompt

    async def identify_categories(self, content: str) -> List[Dict[str, str]]:
        prompt = self._create_category_identification_prompt(content)

        try:
            result = await self._make_api_call(prompt)
            if result and 'content' in result:
                categories_json = result['content'].strip()
                categories_json = re.sub(r'```json\s*|\s*```', '', categories_json)
                categories = json.loads(categories_json)

                if isinstance(categories, list):
                    return categories
                else:
                    logger.error("LLM returned non-list categories")
                    return []
            else:
                logger.error("Failed to get valid response from LLM for categories")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse categories JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in category identification: {e}")
            return []

    async def identify_site_type(self, content: str) -> str:
        prompt = self._create_site_type_prompt(content)

        try:
            result = await self._make_api_call(prompt)
            if result and 'content' in result:
                site_type_json = result['content'].strip()
                site_type_json = re.sub(r'```json\s*|\s*```', '', site_type_json)
                site_type_data = json.loads(site_type_json)

                if isinstance(site_type_data, dict) and 'site_type' in site_type_data:
                    return site_type_data['site_type']
                else:
                    logger.error("LLM returned invalid site type format")
                    return "other"
            else:
                logger.error("Failed to get valid response from LLM for site type")
                return "other"
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse site type JSON: {e}")
            return "other"
        except Exception as e:
            logger.error(f"Error in site type identification: {e}")
            return "other"

    async def cleanup(self):
        """Clean up aiohttp session"""
        if self.session:
            await self.session.close()

class ContentProcessor:

    def __init__(self):
        self.category_keywords = {
            "About Us": ["about", "company", "mission", "vision", "story", "history", "overview", "founded"],
            "Products & Services": ["product", "service", "solution", "offering", "feature", "technology", "platform"],
            "Leadership/Team": ["team", "leadership", "staff", "employee", "founder", "ceo", "management", "director"],
            "Blog/News/Press Release": ["blog", "news", "press", "article", "announcement", "update", "post"],
            "Contact/Support": ["contact", "support", "help", "phone", "email", "address", "location", "reach"],
            "Privacy/Legal": ["privacy", "legal", "terms", "policy", "compliance", "gdpr", "cookie"],
            "Careers/Jobs": ["career", "job", "hiring", "employment", "work", "position", "vacancy", "apply"],
            "Other": []
        }

    def _extract_category_text(self, content: str, category_name: str) -> str:
        content_lower = content.lower()
        keywords = self.category_keywords.get(category_name, [])
        sentences = re.split(r'[.!?]+', content)

        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in keywords:
                if keyword in sentence_lower:
                    relevant_sentences.append(sentence.strip())
                    break

        if relevant_sentences:
            return '. '.join(relevant_sentences[:3]).strip() + '.'

        lines = content.split('\n')
        section_start = None

        for i, line in enumerate(lines):
            line_lower = line.lower()
            for keyword in keywords:
                if keyword in line_lower and (len(line) < 100):
                    section_start = i
                    break
            if section_start is not None:
                break

        if section_start is not None:
            section_text = []
            for i in range(section_start, min(section_start + 10, len(lines))):
                if lines[i].strip():
                    section_text.append(lines[i].strip())
                if i > section_start and any(kw in lines[i].lower() for kw_list in self.category_keywords.values() for kw in kw_list):
                    break

            if section_text:
                result = ' '.join(section_text)
                if len(result) > 300:
                    result = result[:300] + '...'
                return result

        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            for keyword in keywords:
                if keyword in paragraph_lower:
                    if len(paragraph) > 300:
                        return paragraph[:300] + '...'
                    return paragraph.strip()

        return f"Content related to {category_name} found in the document."

    def _associate_links(self, category_name: str, category_text: str, internal_links: List[str]) -> List[str]:
        ## This will associate the link with the particular category ##

        associated_links = []
        keywords = self.category_keywords.get(category_name, [])
        category_words = category_name.lower().replace('/', ' ').split()
        keywords.extend(category_words)

        for link in internal_links:
            link_lower = link.lower()
            for keyword in keywords:
                if keyword in link_lower:
                    associated_links.append(link)
                    break

            if category_name == "About Us" and any(word in link_lower for word in ['about', 'company', 'story']):
                if link not in associated_links:
                    associated_links.append(link)
            elif category_name == "Contact/Support" and any(word in link_lower for word in ['contact', 'support', 'help']):
                if link not in associated_links:
                    associated_links.append(link)
            elif category_name == "Careers/Jobs" and any(word in link_lower for word in ['career', 'job', 'work']):
                if link not in associated_links:
                    associated_links.append(link)

        return associated_links

    def process_categories(self, categories: List[Dict[str, str]], content: str, internal_links: List[str]) -> List[Dict[str, Any]]:

        processed_categories = []

        for category in categories:
            category_name = category.get('category_name', '')
            category_text = self._extract_category_text(content, category_name)
            category_links = self._associate_links(category_name, category_text, internal_links)

            processed_category = {
                category_name: {
                    "links": category_links,
                    "text": category_text
                }
            }

            processed_categories.append(processed_category)

        return processed_categories

class ContentExtractor:
    ## this is the extraction of the content from the url website through headless manner ##
    def __init__(self, headless: bool = True, timeout: int = 30):
        self.headless = headless
        self.timeout = timeout
        self.driver = None

    def _setup_driver(self) -> bool:
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WebDriver: {e}")
            return False

    def extract_content(self, url: str) -> Tuple[str, str, List[str], Optional[str]]:
        error = None
        try:
            if not self._setup_driver():
                return url, "", [], "Failed to initialize WebDriver"

            logger.info(f"Fetching content from: {url}")
            self.driver.get(url)

            WebDriverWait(self.driver, self.timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            final_url = self.driver.current_url
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')


            extracted_content = self._extract_main_content(soup)
            ## we can also summarize the extracted content into small paragraph and pass that as a paraphrase keyword to the LLM gemini but i tried but could make it because of package issue
            internal_links = self._extract_internal_links(soup, final_url)

            logger.info(f"Extracted {len(extracted_content)} characters and {len(internal_links)} links")
            return final_url, extracted_content, internal_links, error

        except TimeoutException:
            error = "Page load timeout"
            logger.error(f"Timeout while loading {url}")
        except WebDriverException as e:
            error = f"WebDriver error: {str(e)}"
            logger.error(f"WebDriver error for {url}: {e}")
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected error for {url}: {e}")
        finally:
            self.cleanup()

        return url, "", [], error

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        main_selectors = [
            'article', 'main', '[role="main"]',
            '.article-content', '.post-content', '.entry-content',
            '.content', '#content', '.article-body', '.story-body', '.post-body'
        ]

        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                largest_element = max(elements, key=lambda x: len(x.get_text()))
                text = largest_element.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    return text

        return soup.get_text(separator=' ', strip=True)

    def _extract_internal_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        internal_links = set()
        base_domain = urlparse(base_url).netloc

        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            if not href or href.startswith('#'):
                continue

            absolute_url = urljoin(base_url, href)
            parsed_url = urlparse(absolute_url)

            if parsed_url.netloc == base_domain:
                clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                if parsed_url.query:
                    clean_url += f"?{parsed_url.query}"

                internal_links.add(clean_url)

        return list(internal_links)

    def cleanup(self):
        ## this will clean up the webdriver resources ##
        if self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                logger.warning(f"Error during driver cleanup: {e}")

def validate_url(url: str) -> bool:
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        response = requests.head(url, timeout=5)
        return response.status_code < 400
    except requests.RequestException:
        return False

def read_urls_from_file(file_path: str) -> List[str]:
    """Read URLs from a text file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

async def process_url(url: str, llm_processor: LLMProcessor, content_processor: ContentProcessor) -> Dict:
    extractor = ContentExtractor()
    final_url, extracted_content, internal_links, error = extractor.extract_content(url)

    if error:
        return {
            "URL": url,
            "site_type": "other",
            "extracted_web_content": "",
            "content": [],
            "errors": error
        }

    categories = await llm_processor.identify_categories(extracted_content)
    site_type = await llm_processor.identify_site_type(extracted_content)

    content = content_processor.process_categories(categories, extracted_content, internal_links)

    return {
        "URL": final_url,
        "site_type": site_type,
        "extracted_web_content": extracted_content,
        "content": content,
        "errors": None
    }


async def main():
    ## This is the main function which is the entry point of the script and it will take the file as the parameter ,the output file (default is mentioned),and number of worker for concurrent execution ##

    parser = argparse.ArgumentParser(description='Article Analyzer & Structured Categorizer')
    parser.add_argument('--file', '-f', help='Path to file containing URLs (one per line)')
    parser.add_argument('--output', '-o', default='analysis_results.json', help='Output JSON file')
    parser.add_argument('--workers', '-w', type=int, default=3, help='Number of concurrent workers')
    args = parser.parse_args()

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY='your_api_key_here'")
        sys.exit(1)

    analyzer = ArticleAnalyzer(api_key, max_workers=args.workers)

    try:
        if args.file:
            urls = await analyzer.load_urls_from_file(args.file)
            if not urls:
                print(f"No URLs found in file: {args.file}")
                sys.exit(1)
        else:
            urls = analyzer.default_urls
            print("Using default URLs for testing")

        print(f"Processing {len(urls)} URLs...")

        results = await analyzer.process_urls(urls)

        print("\n" + "="*80)
        print("ANALYSIS RESULTS")
        print("="*80)

        for result in results:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("-" * 80)

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {args.output}")
        print(f"Processed {len(results)} URLs successfully")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}")
    finally:
        await analyzer.cleanup()

if __name__ == "__main__":
    asyncio.run(main())