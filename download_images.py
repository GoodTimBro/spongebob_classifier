from icrawler.builtin import BingImageCrawler
import os

def download_images(query_list, target_dir, max_num=100):
    os.makedirs(target_dir, exist_ok=True)
    for query in query_list:
        print(f"  🔍 Searching: {query}")
        crawler = BingImageCrawler(storage={"root_dir": target_dir})
        crawler.crawl(keyword=query, max_num=max_num)

if __name__ == "__main__":
    targets = {
        "spongebob": [
            "SpongeBob Squarepants full body PNG",
            "SpongeBob character transparent background",
            "SpongeBob standing pose",
            "SpongeBob front view art"
        ],
        "patrick": [
            "Patrick Star full body PNG",
            "Patrick Star character art",
            "Patrick Star standing transparent"
        ],
        "squidward": [
            "Squidward Tentacles full body PNG",
            "Squidward character art",
            "Squidward standing transparent"
        ],
        "sandy": [
            "Sandy Cheeks full body PNG",
            "Sandy Cheeks character art",
            "Sandy Cheeks astronaut suit"
        ],
        "mr_krabs": [
            "Mr. Krabs full body PNG",
            "Mr. Krabs character transparent",
            "Mr. Krabs cartoon pose"
        ],
        "plankton": [
            "Plankton character full body PNG",
            "Plankton character art",
            "Plankton transparent character"
        ]
    }

    for folder, query_list in targets.items():
        print(f"\n📥 Downloading images for '{folder}'...")
        download_images(query_list, f"./data/train/{folder}", max_num=100)  # 每個關鍵字抓 50 張
