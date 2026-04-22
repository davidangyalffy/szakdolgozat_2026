
import requests
from bs4 import BeautifulSoup
import time
import random
import json
import csv
from datetime import datetime, timedelta
from urllib.parse import urljoin
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IndexHuScraper:
    def __init__(self, base_url="https://index.hu"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.articles = []
    
    def random_delay(self, min_seconds=2, max_seconds=8):
        delay = random.uniform(min_seconds, max_seconds)
        logging.info(f"Waiting {delay:.2f} seconds before next request...")
        time.sleep(delay)
    
    def scrape_article_list(self, url):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links - adjust selectors based on actual HTML structure
            article_links = []
            
            # Common selectors for article listings (adjust as needed)
            articles = soup.find_all('article') or soup.find_all('div', class_='cikk')
            
            for article in articles:
                link = article.find('a', href=True)
                if link:
                    article_url = urljoin(self.base_url, link['href'])
                    article_links.append(article_url)
            
            logging.info(f"Found {len(article_links)} articles")
            return article_links
            
        except Exception as e:
            logging.error(f"Error scraping article list: {e}")
            return []
    
    def scrape_article_content(self, url):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article data - adjust selectors based on actual HTML
            article_data = {
                'url': url,
                'title': '',
                'date': '',
                'author': '',
                'content': '',
                'scraped_at': datetime.now().isoformat()
            }
            
            # Title
            title_tag = soup.find('h1')
            if title_tag:
                article_data['title'] = title_tag.get_text(strip=True)
            
            # Date
            date_tag = soup.find('time') or soup.find(class_='date')
            if date_tag:
                article_data['date'] = date_tag.get_text(strip=True)
            
            # Author
            author_tag = soup.find(class_='author') or soup.find('span', class_='szerzo')
            if author_tag:
                article_data['author'] = author_tag.get_text(strip=True)
            
            # Content
            content_tag = soup.find('article') or soup.find(class_='cikk-torzs')
            if content_tag:
                paragraphs = content_tag.find_all('p')
                article_data['content'] = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            logging.info(f"Successfully scraped: {article_data['title'][:50]}...")
            return article_data
            
        except Exception as e:
            logging.error(f"Error scraping article {url}: {e}")
            return None
    
    def scrape_with_random_delays(self, list_url, min_delay=2, max_delay=8, max_articles=None):
        logging.info(f"Starting scrape of {list_url}")
        
        # Get article list
        article_urls = self.scrape_article_list(list_url)
        
        if not article_urls:
            logging.warning("No articles found!")
            return
        
        # Limit to first N articles if specified
        if max_articles is not None:
            article_urls = article_urls[:max_articles]
            logging.info(f"Limiting scrape to first {len(article_urls)} articles")
        else:
            logging.info(f"Scraping all {len(article_urls)} articles found")
        
        # Scrape each article with random delays
        for i, article_url in enumerate(article_urls, 1):
            logging.info(f"Scraping article {i}/{len(article_urls)}")
            
            article_data = self.scrape_article_content(article_url)
            
            if article_data:
                self.articles.append(article_data)
            
            # Random delay before next request (skip on last article)
            if i < len(article_urls):
                self.random_delay(min_delay, max_delay)
        
        logging.info(f"Scraping complete! Collected {len(self.articles)} articles")
    
    def save_to_json(self, filename='index_hu_articles.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)
        logging.info(f"Articles saved to {filename}")
    
    def save_to_csv(self, filename='index_hu_articles.csv'):
        if not self.articles:
            logging.warning("No articles to save!")
            return
        
        keys = self.articles[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.articles)
        logging.info(f"Articles saved to {filename}")
    
    def generate_date_range(self, start_date, end_date):
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_list = []
        current_date = start_date
        
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)
        
        return date_list
    
    def build_url(self, date, category='belfold'):
        date_str = date.strftime('%Y-%m-%d')
        url = f"https://index.hu/24ora/?s=&tol={date_str}&ig={date_str}&profil=&rovat={category}&cimke=&word=1&pepe=1"
        return url
    
    def scrape_date_range(self, start_date, end_date, category='belfold', 
                         min_delay=3, max_delay=8, max_articles=None,
                         delay_between_days=5):
        dates = self.generate_date_range(start_date, end_date)
        
        logging.info(f"Starting scrape for {len(dates)} days from {start_date} to {end_date}")
        
        for i, date in enumerate(dates, 1):
            date_str = date.strftime('%Y-%m-%d')
            logging.info(f"\n{'='*60}")
            logging.info(f"DAY {i}/{len(dates)}: {date_str}")
            logging.info(f"{'='*60}")
            
            url = self.build_url(date, category)
            
            # Scrape this day's articles
            self.scrape_with_random_delays(url, min_delay, max_delay, max_articles)
            
            # Extra delay between days (skip on last day)
            if i < len(dates):
                logging.info(f"Day complete. Waiting {delay_between_days} seconds before next day...")
                time.sleep(delay_between_days)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"ALL DAYS COMPLETE!")
        logging.info(f"Total articles collected: {len(self.articles)}")
        logging.info(f"{'='*60}")


# Example usage
if __name__ == "__main__":
    # Initialize scraper
    scraper = IndexHuScraper()
    
    # OPTION 1: Scrape all of December 2021
    scraper.scrape_date_range(
        start_date='2021-12-01',
        end_date='2021-12-31',
        category='belfold',
        min_delay=3,           # Random delay between articles: 3-10 seconds
        max_delay=8,
        max_articles=None,     # No limit - scrape all articles per day
        delay_between_days=5   # 5 second pause between days
    )
    
    # Save results with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scraper.save_to_json(f'articles_dec2021_{timestamp}.json')
    scraper.save_to_csv(f'articles_dec2021_{timestamp}.csv')
    
    
    # OPTION 2: Scrape a single day (uncomment to use)
    # scraper_single = IndexHuScraper()
    # url = "https://index.hu/24ora/?s=&tol=2019-01-01&ig=2019-01-01&profil=&rovat=belfold&cimke=&word=1&pepe=1"
    # scraper_single.scrape_with_random_delays(url, min_delay=3, max_delay=10)
    # scraper_single.save_to_json('articles_single_day.json')
    # scraper_single.save_to_csv('articles_single_day.csv')
