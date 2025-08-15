import requests
from typing import Dict, Tuple, Optional

class NewsFactChecker:
    
    api_key = "62f819f2173c4068b3d9a4644b487d0c"
    country = "us"
    news_api_url = url =f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={api_key}"
        
    def _perform_api_request(self, article_text: str) -> Optional[Dict]:
        """
        Makes request to News API and returns processed response
        """
        params = {
            "q": article_text[:100],  # Limit query to first 100 chars
            "language": "en",
            "country": "us",
            "sortBy": "relevancy",
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.news_api_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Request Error: {e}")
            return None
            
    def classify_news(self, article_text: str) -> str:
        """
        Performs basic classification of news text
        """
        if not isinstance(article_text, str):
            article_text = str(article_text)
            
        if not article_text.strip():
            return "⚠️ No text provided for analysis"
            
        suspicious_terms = ['conspiracy', 'shocking truth', 'they dont want you to know',
                          'miracle cure', '100% guaranteed', 'secret they dont want you to know']
        
        article_lower = article_text.lower()
        for term in suspicious_terms:
            if term in article_lower:
                return "❌ Potentially Misleading"
                
        return "✅ No Red Flags Detected"
        
    def fact_check_article(self, article_text: str) -> Tuple[str, Dict]:
        """
        Cross-references article with News API and returns verification status
        """
        if not isinstance(article_text, str):
            article_text = str(article_text)
            
        # Initialize default verification results
        default_results = {
            "similar_articles_found": 0,
            "top_sources": [],
            "publication_dates": []
        }
            
        api_response = self._perform_api_request(article_text)
        
        if not api_response:
            return "Error: Unable to verify with external sources", default_results
            
        try:
            verification_results = {
                "similar_articles_found": api_response.get("totalResults", 0),
                "top_sources": [article.get("source", {}).get("name", "Unknown Source") 
                              for article in api_response.get("articles", [])[:3]],
                "publication_dates": [article.get("publishedAt", "Unknown Date") 
                                    for article in api_response.get("articles", [])[:3]]
            }
        except Exception as e:
            print(f"Error processing API response: {e}")
            return "Error: Unable to process verification results", default_results
        
        if verification_results["similar_articles_found"] > 0:
            status = "✓ Content verified with multiple sources"
        else:
            status = "⚠️ Unable to find corroborating sources"
            
        return status, verification_results

    def analyze_article(self, article_text: str = None) -> str:
        """
        Main function to analyze and fact-check an article
        """
        try:
            if article_text is None:
                article_text = str(input("Please enter the article text: "))
                
            if not article_text.strip():
                return "Error: No text provided for analysis"
                
            # Step 1: Classification
            classification = self.classify_news(article_text)
            
            # Step 2: Fact Checking
            fact_check_status, verification_details = self.fact_check_article(article_text)
            
            # Prepare detailed report
            report = [
                "News Analysis Report",
                "=" * 20,
                f"Classification: {classification}",
                f"Fact Check Status: {fact_check_status}"
            ]
            
            # Add verification details only if available
            if verification_details:
                report.extend([
                    "\nVerification Details:",
                    f"- Similar articles found: {verification_details.get('similar_articles_found', 0)}",
                    "- Top sources referenced:"
                ])
                
                for source in verification_details.get("top_sources", []):
                    report.append(f"  * {source}")
                    
            return "\n".join(report)
            
        except Exception as e:
            return f"Error analyzing article: {str(e)}"

# Usage example
if __name__ == "__main__":
    API_KEY = "62f819f2173c4068b3d9a4644b487d0c"
    
    checker = NewsFactChecker()
    
    # Example usage with direct text input
    sample_text = """Disney World was going to lower the drinking age to 18."""
    
    result = checker.analyze_article(sample_text)
    print(result)