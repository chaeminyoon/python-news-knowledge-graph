from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time
from datetime import datetime
import re
import os

# Constants
CATEGORIES = {
    '정치': 'https://news.naver.com/section/100',
    '경제': 'https://news.naver.com/section/101',
    '사회': 'https://news.naver.com/section/102',
    '생활/문화': 'https://news.naver.com/section/103',
    'IT/과학': 'https://news.naver.com/section/105',
    '세계': 'https://news.naver.com/section/104'
}

NUM_ARTICLES_PER_CATEGORY = 10

def get_article_links(driver, category_url, num_articles):
    driver.get(category_url)
    time.sleep(3)

    article_links = []

    try:
        selectors = [
            'a.sa_text_lede',
            'a.sa_text_strong',
            '.sa_text a',
            '.cluster_text_headline a',
            '.cluster_text_lede a'
        ]

        for selector in selectors:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                url = element.get_attribute('href')
                if (url and 'news.naver.com' in url and '/article/' in url
                    and '/comment/' not in url  # 댓글 페이지만 제외
                    and url not in article_links):
                    article_links.append(url)
                    if len(article_links) >= num_articles:
                        break
            if len(article_links) >= num_articles:
                break

        print(f"✓ {len(article_links)}개의 기사 링크 수집 완료")

    except Exception as e:
        print(f"✗ 기사 링크 수집 실패: {e}")

    return article_links[:num_articles]

def parse_article_detail(driver, article_url, category):
    driver.get(article_url)
    time.sleep(1.5)

    article_data = {
        'article_id': '',
        'title': '',
        'content': '',
        'url': article_url,
        'published_date': '',
        'source': '',
        'author': '',
        'category': category
    }

    try:
        # 기사 ID 생성 (URL에서 추출)
        match = re.search(r'article/(\d+)/(\d+)', article_url)
        if match:
            article_data['article_id'] = f"ART_{match.group(1)}_{match.group(2)}"
        else:
            article_data['article_id'] = f"ART_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 제목
        title_selectors = [
            '#title_area span',
            '#ct .media_end_head_headline',
            '.media_end_head_headline',
            'h2#title_area',
            '.news_end_title'
        ]

        for selector in title_selectors:
            try:
                title_element = driver.find_element(By.CSS_SELECTOR, selector)
                if title_element.text.strip():
                    article_data['title'] = title_element.text.strip()
                    break
            except:
                continue

        # 본문
        content_selectors = [
            '#dic_area',
            'article#dic_area',
            '.go_trans._article_content',
            '._article_body_contents'
        ]

        for selector in content_selectors:
            try:
                content_element = driver.find_element(By.CSS_SELECTOR, selector)
                if content_element.text.strip():
                    article_data['content'] = content_element.text.strip()
                    break
            except:
                continue

        # 언론사
        try:
            source_element = driver.find_element(By.CSS_SELECTOR, 'a.media_end_head_top_logo img')
            article_data['source'] = source_element.get_attribute('alt')
        except:
            try:
                source_element = driver.find_element(By.CSS_SELECTOR, '.media_end_head_top_logo_text')
                article_data['source'] = source_element.text.strip()
            except:
                pass

        # 발행일
        try:
            date_element = driver.find_element(By.CSS_SELECTOR, 'span.media_end_head_info_datestamp_time, span[data-date-time]')
            date_text = date_element.get_attribute('data-date-time') or date_element.text
            article_data['published_date'] = date_text.strip()
        except:
            article_data['published_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')

        # 기자명
        try:
            author_element = driver.find_element(By.CSS_SELECTOR, 'em.media_end_head_journalist_name, span.byline_s')
            article_data['author'] = author_element.text.strip()
        except:
            pass

    except Exception as e:
        print(f"  ✗ 파싱 오류: {e}")

    return article_data

def main():
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # options.add_argument('--headless') # Uncomment if you want to run in headless mode

    driver = webdriver.Chrome(service=service, options=options)

    try:
        # 전체 기사 데이터를 저장할 리스트
        all_articles = []

        # 카테고리별 크롤링
        for category_name, category_url in CATEGORIES.items():
            print(f"\n{'='*60}")
            print(f"[{category_name}] 카테고리 수집 시작...")
            print(f"{'='*60}")

            # 1단계: 기사 링크 수집
            article_links = get_article_links(driver, category_url, NUM_ARTICLES_PER_CATEGORY)

            # 2단계: 각 기사 상세 정보 수집
            for idx, article_url in enumerate(article_links, 1):
                print(f"  [{idx}/{len(article_links)}] {article_url}")
                article_data = parse_article_detail(driver, article_url, category_name)

                if article_data['title']:  # 제목이 있는 경우만 추가
                    all_articles.append(article_data)
                    print(f"  ✓ 수집 완료: {article_data['title'][:50]}...")
                else:
                    print(f"  ✗ 수집 실패 - 제목을 찾을 수 없습니다.")

                time.sleep(0.5)

        # 5. 수집 결과 확인 및 저장
        if all_articles:
            df_articles = pd.DataFrame(all_articles)
            
            # Excel 파일로 저장
            output_filename = f"Articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df_articles.to_excel(output_filename, index=False, engine='openpyxl')
            print(f"\n✓ 데이터가 '{output_filename}' 파일로 저장되었습니다.")
            print(df_articles.head())
        else:
            print("\n✗ 수집된 기사가 없습니다.")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
