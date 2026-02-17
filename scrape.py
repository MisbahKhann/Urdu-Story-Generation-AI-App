from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import json
import re

def extract_story_paragraphs(driver):
    """Extract ONLY story paragraphs using the specific XPath pattern"""
    
    paragraphs = []
    
    try:
        parent_xpath = '//*[@id="main_content"]/div[1]/div[2]/div[8]/div[2]'
        
        try:
            story_container = driver.find_element(By.XPATH, parent_xpath)
        except:
            story_container = driver.find_element(By.XPATH, '//*[@id="main_content"]//div[contains(@class, "clear")]')
        
        script = """
        var element = arguments[0];
        var textNodes = [];
        
        for (var i = 0; i < element.childNodes.length; i++) {
            var node = element.childNodes[i];
            if (node.nodeType === Node.TEXT_NODE) {
                var text = node.textContent.trim();
                if (text.length > 20) {
                    textNodes.push(text);
                }
            }
        }
        
        return textNodes;
        """
        
        text_nodes = driver.execute_script(script, story_container)
        
        for text in text_nodes:
            if re.search(r'[\u0600-\u06FF]', text):
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 20:
                    paragraphs.append(text)
        
        return paragraphs
    
    except Exception as e:
        print(f"      Error extracting paragraphs: {e}")
        return []

def main():
    driver = webdriver.Chrome()
    
    # Set longer page load timeout
    driver.set_page_load_timeout(30)  # 30 seconds timeout
    
    all_stories = []
    story_links = []
    
    # Step 1: Loop through pages and collect story links
    print("📚 Collecting story links from all pages...\n")
    
    for page_num in range(1, 21):
        if page_num == 1:
            url = "https://www.urdupoint.com/kids/category/moral-stories.html"
        else:
            url = f"https://www.urdupoint.com/kids/category/moral-stories-page{page_num}.html"
        
        print(f"📄 Page {page_num}: {url}")
        
        try:
            driver.get(url)
            time.sleep(3)
            
            story_boxes = driver.find_elements(By.CSS_SELECTOR, 'a.sharp_box')
            
            page_count = 0
            for box in story_boxes:
                story_url = box.get_attribute('href')
                urdu_title = box.find_element(By.CLASS_NAME, 'title_ur').text
                
                if story_url not in [s['url'] for s in story_links]:
                    story_links.append({'title': urdu_title, 'url': story_url})
                    page_count += 1
            
            print(f"   ✅ Found {page_count} stories (Total: {len(story_links)})\n")
        
        except TimeoutException:
            print(f"   ⚠️ Page load timeout - skipping page {page_num}\n")
            continue
        except Exception as e:
            print(f"   ❌ Error: {e}\n")
    
    print("="*70)
    print(f"✅ Total stories collected: {len(story_links)}\n")
    print("="*70 + "\n")
    
    # Step 2: Extract each story with retry logic
    failed_stories = []
    
    for idx, story_info in enumerate(story_links, 1):
        print(f"[{idx}/{len(story_links)}] {story_info['title']}")
        
        # Try up to 3 times for each story
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                driver.get(story_info['url'])
                time.sleep(3)  # Increased delay
                
                wait = WebDriverWait(driver, 15)  # Increased wait time
                wait.until(EC.presence_of_element_located((By.ID, "main_content")))
                
                try:
                    title_elem = driver.find_element(By.CLASS_NAME, 'detail_heading')
                    title = title_elem.text.strip()
                except:
                    title = story_info['title']
                
                paragraphs = extract_story_paragraphs(driver)
                
                if not paragraphs:
                    print(f"   ⚠️ No story content extracted")
                    success = True  # Don't retry for empty content
                    break
                
                full_text = '\n\n'.join(paragraphs)
                
                print(f"   ✅ {len(paragraphs)} paragraphs extracted")
                
                story_data = {
                    'title': title,
                    'paragraphs': paragraphs,
                    'full_text': full_text
                }
                
                all_stories.append(story_data)
                
                if paragraphs:
                    preview = paragraphs[0][:80] + "..." if len(paragraphs[0]) > 80 else paragraphs[0]
                    print(f"   📖 Preview: {preview}")
                
                success = True
                
                # Longer delay between stories (be more polite)
                time.sleep(3)
            
            except TimeoutException:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"   ⏳ Timeout - Retry {retry_count}/{max_retries}")
                    time.sleep(5)  # Wait before retry
                else:
                    print(f"   ❌ Timeout after {max_retries} retries - skipping")
                    failed_stories.append(story_info)
            
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"   ⏳ Error - Retry {retry_count}/{max_retries}: {e}")
                    time.sleep(5)
                else:
                    print(f"   ❌ Failed after {max_retries} retries: {e}")
                    failed_stories.append(story_info)
        
        # Restart driver every 50 stories to prevent memory issues
        if idx % 50 == 0:
            print(f"\n   🔄 Restarting browser (processed {idx} stories)...\n")
            driver.quit()
            time.sleep(3)
            driver = webdriver.Chrome()
            driver.set_page_load_timeout(30)
    
    # Step 3: Save to JSON
    print("\n" + "="*70)
    print("💾 Saving to JSON...")
    
    output_file = 'urdu_stories_clean.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_stories, f, ensure_ascii=False, indent=2)
    
    total_paragraphs = sum(len(story['paragraphs']) for story in all_stories)
    total_words = sum(len(story['full_text'].split()) for story in all_stories)
    
    print(f"✅ Saved {len(all_stories)} stories to: {output_file}")
    print(f"📊 Statistics:")
    print(f"   - Total stories: {len(all_stories)}")
    print(f"   - Failed stories: {len(failed_stories)}")
    print(f"   - Total paragraphs: {total_paragraphs}")
    print(f"   - Total words: {total_words:,}")
    if all_stories:
        print(f"   - Avg paragraphs/story: {total_paragraphs/len(all_stories):.1f}")
        print(f"   - Avg words/story: {total_words/len(all_stories):.1f}")
    
    # Save failed stories list
    if failed_stories:
        print(f"\n⚠️ Saving list of failed stories...")
        with open('failed_stories.json', 'w', encoding='utf-8') as f:
            json.dump(failed_stories, f, ensure_ascii=False, indent=2)
        print(f"   Saved to: failed_stories.json")
    
    print("="*70)
    
    driver.quit()

if __name__ == "__main__":
    main()
