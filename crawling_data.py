import csv
import glob
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import unquote, urlparse
import requests
from playwright.sync_api import sync_playwright


def sanitize_text(text: str) -> str:
    """Loại bỏ ký tự xuống dòng và tab để không làm vỡ cấu trúc tệp TSV."""
    if not text:
        return ""
    text = (
        text.replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def resolve_short_url(url: str) -> str:
    """Giải mã link rút gọn (như shp.ee, shope.ee, ti.ki, s.lazada.vn) sang link gốc."""
    short_domains = ["shp.ee", "shope.ee", "ti.ki", "s.lazada.vn", "bit.ly", "t.co"]
    if any(domain in url for domain in short_domains):
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            res = requests.head(url, headers=headers, allow_redirects=True, timeout=8)
            return res.url
        except Exception as e:
            print(f"[!] Không thể mở rộng link rút gọn {url}: {e}")
    return url


def parse_product_url(raw_url: str) -> dict:
    """
    Tự động phân tích đường link sản phẩm để trích xuất platform, shop_id, item_id và tên sản phẩm.
    Hỗ trợ Shopee, Tiki và Lazada.
    """
    url = resolve_short_url(raw_url.strip())
    parsed = urlparse(url)
    path = parsed.path

    # ==========================
    # 1. SHOPEE
    # ==========================
    if "shopee.vn" in parsed.netloc:
        shop_id = None
        item_id = None
        product_name = ""

        m1 = re.search(r"/product/(\d+)/(\d+)", path)
        if m1:
            shop_id = int(m1.group(1))
            item_id = int(m1.group(2))

        m2 = re.search(r"/(.*?)-?i\.(\d+)\.(\d+)", path)
        if m2:
            slug = m2.group(1)
            shop_id = int(m2.group(2))
            item_id = int(m2.group(3))
            slug_clean = unquote(slug).replace("-", " ").strip()
            if slug_clean:
                product_name = slug_clean

        if shop_id and item_id:
            if not product_name:
                product_name = f"Shopee_Item_{item_id}"

            return {
                "platform": "shopee",
                "shop_id": shop_id,
                "item_id": item_id,
                "name": product_name,
                "url": url,
            }

    # ==========================
    # 2. TIKI
    # ==========================
    if "tiki.vn" in parsed.netloc:
        item_id = None
        product_name = ""

        m1 = re.search(r"/(.*?)-p(\d+)\.html", path)
        if m1:
            slug = m1.group(1)
            item_id = int(m1.group(2))
            slug_clean = unquote(slug).replace("-", " ").strip()
            if slug_clean:
                product_name = slug_clean.capitalize()

        m2 = re.search(r"/p/(\d+)", path)
        if m2:
            item_id = int(m2.group(1))

        if not item_id:
            m3 = re.search(r"[?&](?:spid|product_id)=(\d+)", parsed.query)
            if m3:
                item_id = int(m3.group(1))

        if item_id:
            try:
                info_url = f"https://tiki.vn/api/v2/products/{item_id}"
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                r = requests.get(info_url, headers=headers, timeout=5)
                if r.status_code == 200:
                    api_name = r.json().get("name")
                    if api_name:
                        product_name = api_name
            except Exception:
                pass

            if not product_name:
                product_name = f"Tiki_Item_{item_id}"

            return {
                "platform": "tiki",
                "shop_id": None,
                "item_id": item_id,
                "name": product_name,
                "url": url,
            }

    # ==========================
    # 3. LAZADA
    # ==========================
    if "lazada.vn" in parsed.netloc:
        item_id = None
        sku_id = None
        product_name = ""

        m_item = re.search(r"-i(\d+)", path)
        if m_item:
            item_id = int(m_item.group(1))

        m_sku = re.search(r"-s(\d+)", path)
        if m_sku:
            sku_id = int(m_sku.group(1))

        m_slug = re.search(r"/products/(.*?)-i\d+", path)
        if m_slug:
            slug = m_slug.group(1)
            if slug and slug != "pdp":
                product_name = unquote(slug).replace("-", " ").strip().capitalize()

        if not product_name:
            m_q = re.search(r"query(?:%253A|%3A|:)(.*?)(?:%253B|%3B|;|&|$)", parsed.query)
            if m_q:
                q_val = unquote(unquote(m_q.group(1))).replace("+", " ").strip()
                if q_val:
                    product_name = q_val.capitalize()

        if not product_name and item_id:
            product_name = f"Lazada_Item_{item_id}"

        if item_id:
            return {
                "platform": "lazada",
                "shop_id": None,
                "item_id": item_id,
                "sku_id": sku_id,
                "name": product_name,
                "url": url,
            }

    return {}


def load_cookie_list(cookie_file: str = "input/cookie.txt") -> list[dict]:
    """Đọc và chuyển đổi chuỗi cookie trong input/cookie.txt thành danh sách cookie cho Playwright."""
    if not os.path.exists(cookie_file):
        return []

    with open(cookie_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        return []

    raw_cookie = "".join(lines)
    cookies = []
    for item in raw_cookie.split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name:
            cookies.append({
                "name": name,
                "value": value,
                "domain": ".shopee.vn",
                "path": "/",
            })
    return cookies


class ReviewCrawler:

    def __init__(self, output_tsv_path: str = "output/reviews_raw.tsv", headless: bool = False):
        self.output_tsv_path = output_tsv_path
        self.headless = headless
        self.seen_review_ids = set()
        self.fieldnames = [
            "review_id",
            "platform",
            "item_id",
            "item_category",
            "product_name",
            "product_variant",
            "rating_star",
            "raw_comment",
            "created_at",
            "crawled_at",
        ]
        self._init_tsv_file()

    def _init_tsv_file(self):
        """Khởi tạo header cho tệp TSV, tự động lọc sạch dữ liệu trùng lặp cũ và nạp ID duy nhất vào bộ nhớ."""
        dir_name = os.path.dirname(self.output_tsv_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if not os.path.exists(self.output_tsv_path):
            with open(self.output_tsv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=self.fieldnames, delimiter="\t"
                )
                writer.writeheader()
        else:
            # Đọc toàn bộ file cũ để lọc trùng và lấy danh sách review_id đã có
            try:
                with open(self.output_tsv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    rows = list(reader)

                unique_rows = []
                for r in rows:
                    rev_id = r.get("review_id", "").strip()
                    if rev_id and rev_id not in self.seen_review_ids:
                        self.seen_review_ids.add(rev_id)
                        unique_rows.append(r)

                if len(unique_rows) < len(rows):
                    print(f"[*] Phát hiện và tự động loại bỏ {len(rows) - len(unique_rows)} bản ghi trùng lặp trong '{self.output_tsv_path}'.")
                    with open(self.output_tsv_path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=self.fieldnames, delimiter="\t")
                        writer.writeheader()
                        writer.writerows(unique_rows)

                print(f"[*] Cơ sở dữ liệu hiện có: {len(self.seen_review_ids)} đánh giá duy nhất.")
            except Exception as e:
                print(f"[!] Lỗi khi nạp file cũ: {e}")

    def save_records(self, records: list[dict]):
        """Ghi nối tiếp các bản ghi CHƯA TỪNG CÓ vào tệp TSV, tự động ngăn chặn trùng lặp."""
        new_records = []
        for r in records:
            rev_id = r.get("review_id")
            if rev_id and rev_id not in self.seen_review_ids:
                self.seen_review_ids.add(rev_id)
                new_records.append(r)

        if not new_records:
            return

        with open(self.output_tsv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self.fieldnames, delimiter="\t"
            )
            for record in new_records:
                writer.writerow(record)
        print(f"  -> Đã lưu thêm {len(new_records)} review mới vào {self.output_tsv_path}")

    # =========================================================================
    # CRAWLER 1: SHOPEE BẰNG CHROME THẬT + TÀI KHOẢN + CHỐNG BỊ CAPTCHA
    # =========================================================================
    def crawl_shopee_playwright(
        self,
        url: str,
        shop_id: int,
        item_id: int,
        product_name: str,
        item_category: str = "general",
        max_reviews: int = 150,
    ):
        print(f"\n[*] Đang khởi động Google Chrome cào Shopee | ID: {item_id}")
        print(f"    Tên sản phẩm: {product_name[:60]}...")

        collected = 0
        seen_cmtid = set()
        profile_dir = os.path.abspath("browser_profile")
        os.makedirs(profile_dir, exist_ok=True)

        with sync_playwright() as p:
            launch_args = {
                "user_data_dir": profile_dir,
                "headless": self.headless,
                "viewport": {"width": 1366, "height": 850},
                "locale": "vi-VN",
                "ignore_default_args": ["--enable-automation"],
                "args": [
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-infobars",
                    "--start-maximized",
                ],
            }

            try:
                context = p.chromium.launch_persistent_context(
                    channel="chrome",
                    **launch_args
                )
            except Exception:
                context = p.chromium.launch_persistent_context(**launch_args)

            user_cookies = load_cookie_list("input/cookie.txt")
            if user_cookies:
                try:
                    context.add_cookies(user_cookies)
                except Exception:
                    pass

            page = context.pages[0] if context.pages else context.new_page()

            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                window.navigator.chrome = {
                    runtime: {},
                };
            """)

            def on_response(response):
                nonlocal collected
                if "get_ratings" in response.url and response.status == 200:
                    try:
                        res_json = response.json()
                        ratings = res_json.get("data", {}).get("ratings", [])
                        if ratings:
                            batch = []
                            crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            for r in ratings:
                                cmtid = str(r.get("cmtid", ""))
                                rev_id = f"sp_{cmtid}"
                                if cmtid in seen_cmtid or rev_id in self.seen_review_ids:
                                    continue
                                seen_cmtid.add(cmtid)

                                raw_comment = sanitize_text(r.get("comment", ""))
                                if len(raw_comment) < 15:
                                    continue

                                product_items = r.get("product_items", [])
                                variant = ""
                                if product_items and isinstance(product_items, list):
                                    variant = product_items[0].get("model_name", "")

                                ctime = r.get("ctime", 0)
                                created_at = (
                                    datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
                                    if ctime
                                    else ""
                                )

                                record = {
                                    "review_id": rev_id,
                                    "platform": "shopee",
                                    "item_id": str(item_id),
                                    "item_category": item_category,
                                    "product_name": sanitize_text(product_name),
                                    "product_variant": sanitize_text(variant),
                                    "rating_star": int(r.get("rating_star", 5)),
                                    "raw_comment": raw_comment,
                                    "created_at": created_at,
                                    "crawled_at": crawled_at,
                                }
                                batch.append(record)
                                collected += 1
                                if collected >= max_reviews:
                                    break

                            if batch:
                                self.save_records(batch)
                    except Exception:
                        pass

            page.on("response", on_response)

            try:
                print("  [1/4] Đang mở trang sản phẩm...")
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(4000)

                if "verify/captcha" in page.url or "captcha" in page.url.lower():
                    print("\n  ⚠️ [CHÚ Ý] Shopee yêu cầu xác minh Captcha!")
                    print("  👉 Vui lòng nhìn lên cửa sổ trình duyệt Chrome vừa mở:")
                    print("     - Kéo mảnh ghép xác minh.")
                    print("     - Bạn có 35 giây để hoàn tất xác minh...")

                    for sec in range(35, 0, -5):
                        if "verify/captcha" not in page.url:
                            print("  [✓] Đã vượt qua Captcha thành công!")
                            break
                        print(f"     ... Còn {sec} giây để hoàn tất captcha...")
                        page.wait_for_timeout(5000)

                    if "verify/captcha" not in page.url:
                        page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        page.wait_for_timeout(3000)

                try:
                    page.keyboard.press("Escape")
                except Exception:
                    pass

                print("  [2/4] Đang trích xuất đánh giá...")
                offset = 0
                limit = 50
                while collected < max_reviews:
                    js_code = f"""
                    async () => {{
                        try {{
                            const res = await fetch("https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&shopid={shop_id}&limit={limit}&offset={offset}&type=0");
                            if (res.status === 200) {{
                                return await res.json();
                            }}
                            return {{ status: res.status }};
                        }} catch (e) {{
                            return {{ error: e.toString() }};
                        }}
                    }}
                    """
                    result = page.evaluate(js_code)
                    if isinstance(result, dict) and "data" in result:
                        ratings = result.get("data", {}).get("ratings", [])
                        if not ratings:
                            break
                        batch = []
                        crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for r in ratings:
                            cmtid = str(r.get("cmtid", ""))
                            rev_id = f"sp_{cmtid}"
                            if cmtid in seen_cmtid or rev_id in self.seen_review_ids:
                                continue
                            seen_cmtid.add(cmtid)

                            raw_comment = sanitize_text(r.get("comment", ""))
                            if len(raw_comment) < 15:
                                continue

                            product_items = r.get("product_items", [])
                            variant = ""
                            if product_items and isinstance(product_items, list):
                                variant = product_items[0].get("model_name", "")

                            ctime = r.get("ctime", 0)
                            created_at = (
                                datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
                                if ctime
                                else ""
                            )

                            record = {
                                "review_id": rev_id,
                                "platform": "shopee",
                                "item_id": str(item_id),
                                "item_category": item_category,
                                "product_name": sanitize_text(product_name),
                                "product_variant": sanitize_text(variant),
                                "rating_star": int(r.get("rating_star", 5)),
                                "raw_comment": raw_comment,
                                "created_at": created_at,
                                "crawled_at": crawled_at,
                            }
                            batch.append(record)
                            collected += 1
                            if collected >= max_reviews:
                                break

                        if batch:
                            self.save_records(batch)
                        offset += limit
                        page.wait_for_timeout(1000)
                    else:
                        break

                if collected < max_reviews:
                    print("  [3/4] Đang cuộn trang xuống phần Đánh giá sản phẩm...")
                    for _ in range(8):
                        page.evaluate("window.scrollBy(0, 600)")
                        page.wait_for_timeout(800)

                    print(f"  [4/4] Đã thu thập: {collected}/{max_reviews} đánh giá. Đang quét thêm trang...")
                    page_step = 1
                    while collected < max_reviews and page_step < 20:
                        next_btn = page.query_selector(
                            ".shopee-page-controller .shopee-icon-button--right, button[aria-label='next page'], .shopee-icon-button--right"
                        )
                        if not next_btn or not next_btn.is_enabled():
                            break
                        next_btn.click()
                        page.wait_for_timeout(2500)
                        page_step += 1

            except Exception as e:
                print(f"  [!] Lỗi khi cào bằng Playwright: {e}")
            finally:
                context.close()

        print(f"[✓] Hoàn thành Shopee ID {item_id}: Tổng cộng thu thập được {collected} đánh giá hợp lệ.")

    # =========================================================================
    # CRAWLER 2: TIKI API (Rất nhanh và ổn định)
    # =========================================================================
    def crawl_tiki(
        self,
        item_id: int,
        product_name: str,
        item_category: str = "general",
        max_reviews: int = 150,
    ):
        print(f"\n[*] Đang cào Tiki | ID: {item_id} | Tên: {product_name[:50]}...")
        api_url = "https://tiki.vn/api/v2/reviews"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

        collected = 0
        page = 1
        limit = 20

        while collected < max_reviews:
            params = {
                "product_id": item_id,
                "sort": "score|desc",
                "page": page,
                "limit": limit,
                "include": "comments",
            }

            try:
                res = requests.get(api_url, headers=headers, params=params, timeout=10)
                if res.status_code != 200:
                    print(f"  [!] Tiki trả về HTTP: {res.status_code}")
                    break

                data = res.json().get("data", [])
                if not data:
                    print("  [*] Đã hết bình luận Tiki.")
                    break

                batch = []
                crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for r in data:
                    rev_id = f"tk_{r.get('id', '')}"
                    # Chống trùng lặp với các lần chạy trước
                    if rev_id in self.seen_review_ids:
                        continue

                    content = r.get("content", "")
                    title = r.get("title", "")
                    full_text = f"{title}. {content}" if title else content
                    raw_comment = sanitize_text(full_text)

                    if len(raw_comment) < 15:
                        continue

                    created_at_val = r.get("created_at")
                    if isinstance(created_at_val, int):
                        created_at = datetime.fromtimestamp(created_at_val).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    else:
                        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    record = {
                        "review_id": rev_id,
                        "platform": "tiki",
                        "item_id": str(item_id),
                        "item_category": item_category,
                        "product_name": sanitize_text(product_name),
                        "product_variant": "",
                        "rating_star": int(r.get("rating", 5)),
                        "raw_comment": raw_comment,
                        "created_at": created_at,
                        "crawled_at": crawled_at,
                    }
                    batch.append(record)
                    collected += 1
                    if collected >= max_reviews:
                        break

                self.save_records(batch)
                page += 1
                time.sleep(1.2)

            except Exception as e:
                print(f"  [!] Lỗi kết nối Tiki: {e}")
                break

        print(f"[✓] Hoàn thành Tiki ID {item_id}: Thu thập thêm {collected} đánh giá mới.")

    # =========================================================================
    # CRAWLER 3: LAZADA API (Hỗ trợ cào đánh giá Lazada)
    # =========================================================================
    def crawl_lazada(
        self,
        item_id: int,
        product_name: str,
        item_category: str = "general",
        max_reviews: int = 150,
        url: str = "",
    ):
        print(f"\n[*] Đang cào Lazada | ID: {item_id} | Tên: {product_name[:50]}...")
        endpoints = [
            f"https://member.lazada.vn/pdp/review/getReviewList?itemId={item_id}&pageSize=20&filter=0&sort=0",
            f"https://my.lazada.vn/pdp/review/getReviewList?itemId={item_id}&pageSize=20&filter=0&sort=0",
            f"https://www.lazada.vn/pdp/review/getReviewList?itemId={item_id}&pageSize=20&filter=0&sort=0",
        ]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": url or f"https://www.lazada.vn/products/i{item_id}.html",
        }

        collected = 0
        page = 1

        while collected < max_reviews:
            success_in_batch = False
            for api_base in endpoints:
                api_url = f"{api_base}&pageNo={page}"
                try:
                    res = requests.get(api_url, headers=headers, timeout=10)
                    if res.status_code != 200:
                        continue
                    res_json = res.json()
                    data = res_json.get("data") or res_json.get("model") or {}
                    items = data.get("items", [])
                    if not items and isinstance(data, list):
                        items = data

                    if items:
                        batch = []
                        crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        for it in items:
                            rev_raw_id = str(it.get("reviewId") or it.get("id") or "")
                            rev_id = f"lz_{rev_raw_id}"
                            if not rev_raw_id or rev_id in self.seen_review_ids:
                                continue

                            raw_comment = sanitize_text(it.get("reviewContent") or it.get("content") or "")
                            if len(raw_comment) < 10:
                                continue

                            star = int(it.get("buyerRating") or it.get("rating") or 5)
                            created_at = it.get("reviewTime") or it.get("createTime") or ""
                            variant = it.get("skuInfo") or ""

                            record = {
                                "review_id": rev_id,
                                "platform": "lazada",
                                "item_id": str(item_id),
                                "item_category": item_category,
                                "product_name": sanitize_text(product_name),
                                "product_variant": sanitize_text(variant),
                                "rating_star": star,
                                "raw_comment": raw_comment,
                                "created_at": created_at,
                                "crawled_at": crawled_at,
                            }
                            batch.append(record)
                            collected += 1
                            if collected >= max_reviews:
                                break

                        if batch:
                            self.save_records(batch)
                            success_in_batch = True
                            break
                except Exception:
                    pass

            if not success_in_batch:
                print("  [*] Đã hết bình luận hoặc API Lazada cần xác thực thêm.")
                break

            page += 1
            time.sleep(1.5)

        print(f"[✓] Hoàn thành Lazada ID {item_id}: Thu thập được {collected} đánh giá.")


def load_urls_from_input_folder(input_folder: str = "input") -> list[dict]:
    """
    Đọc tất cả file .txt trong thư mục input/.
    """
    if not os.path.exists(input_folder):
        os.makedirs(input_folder, exist_ok=True)
        return []

    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    txt_files = [f for f in txt_files if not os.path.basename(f).lower().startswith("cookie")]

    if not txt_files:
        print(f"[!] Không tìm thấy file danh sách link nào trong '{input_folder}/'.")
        return []

    tasks = []
    for filepath in txt_files:
        print(f"[*] Đang đọc file input: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split("|")]
                raw_url = parts[0]
                category = parts[1] if len(parts) > 1 and parts[1] else "general"
                max_reviews = 150
                if len(parts) > 2:
                    try:
                        max_reviews = int(parts[2])
                    except ValueError:
                        pass

                info = parse_product_url(raw_url)
                if not info:
                    print(f"  [!] Bỏ qua dòng {line_no}: Không nhận diện được sàn hoặc ID từ URL: {raw_url}")
                    continue

                tasks.append({
                    "platform": info["platform"],
                    "shop_id": info.get("shop_id"),
                    "item_id": info["item_id"],
                    "sku_id": info.get("sku_id"),
                    "name": info["name"],
                    "category": category,
                    "max": max_reviews,
                    "url": info.get("url", raw_url),
                })

    return tasks


# =========================================================================
# CHƯƠNG TRÌNH CHÍNH
# =========================================================================
if __name__ == "__main__":
    INPUT_DIR = "input"
    OUTPUT_FILE = "output/reviews_raw.tsv"

    crawler = ReviewCrawler(output_tsv_path=OUTPUT_FILE, headless=False)
    tasks = load_urls_from_input_folder(INPUT_DIR)

    if not tasks:
        print(f"\n[!] Chưa có link nào trong thư mục '{INPUT_DIR}/urls.txt'.")
    else:
        print(f"\n=== TÌM THẤY {len(tasks)} SẢN PHẨM CẦN CÀO ===")
        for i, t in enumerate(tasks, 1):
            print(f"{i}. [{t['platform'].upper()}] {t['name']} (ID: {t['item_id']}) -> Tối đa {t['max']} reviews")

        print("\n=== BẮT ĐẦU CÀO DỮ LIỆU ===")
        for item in tasks:
            if item["platform"] == "shopee":
                crawler.crawl_shopee_playwright(
                    url=item["url"],
                    shop_id=item["shop_id"],
                    item_id=item["item_id"],
                    product_name=item["name"],
                    item_category=item["category"],
                    max_reviews=item["max"],
                )
            elif item["platform"] == "tiki":
                crawler.crawl_tiki(
                    item_id=item["item_id"],
                    product_name=item["name"],
                    item_category=item["category"],
                    max_reviews=item["max"],
                )
            elif item["platform"] == "lazada":
                crawler.crawl_lazada(
                    item_id=item["item_id"],
                    product_name=item["name"],
                    item_category=item["category"],
                    max_reviews=item["max"],
                    url=item["url"],
                )
            time.sleep(2)

        print(f"\n🎉 HOÀN THÀNH TẤT CẢ! Dữ liệu TSV đã được lưu tại: {OUTPUT_FILE}")