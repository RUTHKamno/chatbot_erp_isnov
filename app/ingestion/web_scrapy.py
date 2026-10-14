# Fichier : app/ingestion/web_scraper.py
import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

import asyncio
import logging
from typing import Dict, List, Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

log = logging.getLogger(__name__)

# 1. Regex pour [Texte](URL)
MD_LINK_REGEX = re.compile(r"\[([^\]]+)\]\(((?:https?://|/)[^\s\)]+)\)")
# 2. Regex pour les autolinks <https://...> ou URLs brutes
RAW_URL_REGEX = re.compile(r"(?:<)?(https?://[^\s>\"\']+)(?:>)?")


# Erreur personnalisée pour le scraping impossible
class ScrapingBlockedException(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def extract_clean_text(html_content: str) -> str:
    """Extrait proprement le contenu textuel en évitant la suppression excessive du contenu principal."""
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Ne supprimer QUE les éléments parasites stricts (SANS supprimer form/div/aside de documentation)
    for element in soup(["script", "style", "noscript", "svg", "iframe"]):
        element.decompose()

    # 2. Cibler les conteneurs principaux courants
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(
            "div", class_=re.compile(r"content|main|page|body|container|doc", re.I)
        )
        or soup.find(
            "div", id=re.compile(r"content|main|page|body|container|doc", re.I)
        )
        or soup.body
    )

    if not main_content:
        return ""

    # 3. Extraction structurée ligne par ligne (en filtrant les lignes vides ou trop courtes sans contenu)
    lines = []
    for line in main_content.get_text(separator="\n").splitlines():
        clean_line = line.strip()
        if clean_line:
            lines.append(clean_line)

    return "\n".join(lines)


def run_crawl4ai_sync(
    start_url: str,
    output_txt_path: Path,
    max_pages: int = 5,
) -> Path:
    """
    Wrapper synchrone pour exécuter le crawler asynchrone Crawl4AI dans les workers FastAPI/ThreadPool.
    """
    return asyncio.run(_crawl_process(start_url, output_txt_path, max_pages))


async def _crawl_process(
    start_url: str,
    output_txt_path: Path,
    max_pages: int,
) -> Path:
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    # Configuration du navigateur Chromium intégré à Crawl4AI
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage"],
    )

    # Configuration du scraping (Markdown RAG + contournement)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,  # Ignore le bruit/headers vides
        remove_overlay_elements=True,  # Supprime modales & bannières de cookies
        process_iframes=False,
    )

    all_markdown_blocks = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        log.info(f"[Crawl4AI] Début du scraping pour : {start_url}")

        # Crawl de la page principale
        result = await crawler.arun(url=start_url, config=run_config)

        if not result.success:
            raise ScrapingBlockedException(
                f"Échec du chargement de la page ({result.error_message or 'Accès refusé/Timeout'})."
            )

        # Vérification des blocages Anti-Bot
        status_code = getattr(result, "status_code", 200)
        if status_code in (401, 403, 429, 503):
            raise ScrapingBlockedException(
                f"Accès refusé par le serveur cible (HTTP {status_code}). Protection anti-bot active."
            )

        # Extraction du Markdown propre généré automatiquement par Crawl4AI
        main_md = (
            result.markdown.raw_markdown
            if hasattr(result.markdown, "raw_markdown")
            else str(result.markdown)
        )

        if not main_md or len(main_md.strip()) < 50:
            raise ScrapingBlockedException(
                "Le contenu extrait est vide ou protégé par un pare-feu/CAPTCHA."
            )

        all_markdown_blocks.append(f"--- [Page 1 : {start_url}] ---\n{main_md.strip()}")

        # Récupération des liens internes pour la pagination / documentation multi-pages
        internal_links = [
            link["href"]
            for link in result.links.get("internal", [])
            if link.get("href") and start_url in link.get("href")
        ]

        # Crawl des sous-pages de documentation (limité à max_pages)
        page_count = 1
        for link in internal_links:
            if page_count >= max_pages:
                break

            log.info(
                f"[Crawl4AI] Crawl sous-page {page_count + 1}/{max_pages} : {link}"
            )
            sub_res = await crawler.arun(url=link, config=run_config)

            if sub_res.success:
                sub_md = (
                    sub_res.markdown.raw_markdown
                    if hasattr(sub_res.markdown, "raw_markdown")
                    else str(sub_res.markdown)
                )
                if sub_md and len(sub_md.strip()) > 50:
                    page_count += 1
                    all_markdown_blocks.append(
                        f"--- [Page {page_count} : {link}] ---\n{sub_md.strip()}"
                    )

    # Sauvegarde dans le fichier temporaire .txt au format universel
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_markdown_blocks))

    log.info(
        f"[Crawl4AI] Scraping terminé. {len(all_markdown_blocks)} page(s) enregistrée(s) dans {output_txt_path}"
    )
    return output_txt_path


def scrape_web_documentation_advanced(
    start_url: str,
    output_txt_path: Path,
    max_pages: int = 10,
    cookies: list[dict] | None = None,
) -> Path:
    """
    Scrappe une documentation web avec Playwright (gestion JS, pagination, cookies).
    Lève ScrapingBlockedException si le site bloque le scraping ou nécessite un CAPTCHA/Auth avancée.
    """
    from playwright.sync_api import (
        sync_playwright,
        TimeoutError as PlaywrightTimeoutError,
    )

    visited_urls = set()
    current_url = start_url
    all_texts = []

    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # Lancement du navigateur Headless Chromium
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Ajout des cookies personnalisés si fournis
        if cookies:
            context.add_cookies(cookies)

        page = context.new_page()

        for page_idx in range(1, max_pages + 1):
            if current_url in visited_urls:
                break
            visited_urls.add(current_url)

            log.info(f"[Scraper] Navigation vers la page {page_idx} : {current_url}")

            try:
                # Tentative de chargement de la page
                response = page.goto(
                    current_url, wait_until="networkidle", timeout=30000
                )

                # 1. Vérification des codes d'erreur HTTP
                if response and response.status in (401, 403, 429, 503):
                    raise ScrapingBlockedException(
                        f"Accès refusé par le serveur cible (Code HTTP {response.status}). Le site protège son contenu contre le scraping automatique."
                    )

                # 2. Détection des pare-feux / CAPTCHA (Cloudflare, Incapsula, etc.)
                html_content = page.content()
                html_lower = html_content.lower()

                if any(
                    bad_token in html_lower
                    for bad_token in [
                        "just a moment...",
                        "cloudflare",
                        "enable javascript",
                        "captcha",
                        "access denied",
                        "ddos protection",
                    ]
                ):
                    raise ScrapingBlockedException(
                        "Protection anti-bot détectée (Cloudflare/CAPTCHA). Le site nécessite une validation humaine ou un accès interactif impossible à automatiser."
                    )

                # 3. Traitement des banderoles de cookies (Cookie Banner / Consentement)
                try:
                    cookie_btn = page.query_selector(
                        "button:has-text('Accept'), button:has-text('Accepter'), button:has-text('Allow all')"
                    )
                    if cookie_btn:
                        cookie_btn.click()
                        page.wait_for_timeout(1000)
                except Exception:
                    pass  # Pas de banderole bloquante

                # 4. Extraction du texte de la page courante
                extracted = extract_clean_text(html_content)
                if extracted:
                    all_texts.append(
                        f"--- [Page {page_idx} : {current_url}] ---\n{extracted}"
                    )

                # 5. Détection renforcée du bouton "Suivant / Next"
                next_button = (
                    page.query_selector("li.next a")
                    or page.query_selector("a:has-text('Next')")
                    or page.query_selector("a:has-text('Suivant')")
                    or page.query_selector("a:has-text('›')")
                    or page.query_selector("a:has-text('→')")
                    or page.query_selector("a[rel='next']")
                    or page.query_selector(".pagination .next a")
                )

                if next_button:
                    next_href = next_button.get_attribute("href")
                    if next_href:
                        # Résolution robuste des URL relatives (ex: 'catalogue/page-2.html')
                        current_url = urljoin(page.url, next_href)
                    else:
                        break
                else:
                    break

            except PlaywrightTimeoutError:
                raise ScrapingBlockedException(
                    "Délai d'attente dépassé lors du chargement de la page (Timeout). Le site est trop lent ou bloque les requêtes automatisées."
                )
            except ScrapingBlockedException:
                raise
            except Exception as e:
                log.error(f"[Scraper Error] {e}")
                raise ScrapingBlockedException(
                    f"Erreur technique lors du rendu de la page : {str(e)}"
                )

        browser.close()

    if not all_texts:
        raise ScrapingBlockedException(
            "Aucun contenu textuel exploitable n'a pu être extrait de la page web."
        )

    print(f"[Scraper] Total de pages scrappées : {all_texts}")

    # Écriture du résultat final
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_texts))

    return {
        "output_txt_path": output_txt_path,
        "test_output_txt_path": all_texts,
        "total_pages_scraped": len(all_texts),
    }


def extract_markdown_links(markdown_text: str, source_url: str) -> List[Dict[str, str]]:
    """Extrait TOUS les liens pertinents (Markdown, autolinks <http...>, et URLs brutes)."""
    found_links = []
    seen_urls = set()

    source_parsed = urlparse(source_url)
    source_domain = source_parsed.netloc.lower()

    excluded_domains = {
        "facebook.com",
        "x.com",
        "twitter.com",
        "linkedin.com",
        "youtube.com",
        "github.com",
        "reddit.com",
        "threads.net",
        "bsky.app",
        "instagram.com",
    }

    def is_valid_url(url: str) -> bool:
        """Vérifie si l'URL est exploitable et non exclue."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Ignorer les ancres pures (# section), les schémas non HTTP, et les réseaux sociaux
        if not parsed.scheme.startswith("http"):
            return False
        if any(ex in domain for ex in excluded_domains):
            return False
        # Éliminer les fichiers images / médias directes
        if any(
            parsed.path.lower().endswith(ext)
            for ext in [".png", ".jpg", ".jpeg", ".gif", ".svg", ".css", ".js"]
        ):
            return False
        return True

    # PASSAGE 1 : Extraction des liens Markdown [Texte](URL)
    for match in MD_LINK_REGEX.finditer(markdown_text):
        label = match.group(1).strip()
        url = match.group(2).strip()

        # Nettoyage des paramètres de tracking Google Analytics (_gl, _ga) si besoin
        clean_url = url.split("?_gl=")[0].split("&_gl=")[0]

        if is_valid_url(clean_url) and clean_url not in seen_urls:
            # Éviter les labels d'images (![alt])
            if not label.startswith("!") and len(label) > 1:
                seen_urls.add(clean_url)
                domain = urlparse(clean_url).netloc.lower()
                found_links.append(
                    {
                        "label": label,
                        "url": clean_url,
                        "is_internal": domain == source_domain
                        or domain.endswith("dolibarr.org"),
                    }
                )

    # PASSAGE 2 : Extraction des autolinks <https://...> ou URLs brutes non encore capturées
    for match in RAW_URL_REGEX.finditer(markdown_text):
        raw_url = match.group(1).strip()
        clean_url = raw_url.split("?_gl=")[0].split("&_gl=")[0]

        if is_valid_url(clean_url) and clean_url not in seen_urls:
            seen_urls.add(clean_url)
            domain = urlparse(clean_url).netloc.lower()

            # Création d'un label propre basé sur le domaine ou le chemin
            label_text = (
                f"Lien externe : {domain}"
                if domain != source_domain
                else f"Page : {urlparse(clean_url).path or domain}"
            )

            found_links.append(
                {
                    "label": label_text,
                    "url": clean_url,
                    "is_internal": domain == source_domain
                    or domain.endswith("dolibarr.org"),
                }
            )

    return found_links


def crawl_multiple_selected_urls_sync(start_url: str, selected_urls: List[str]) -> str:
    """
    Scrappe l'URL de départ ainsi que la liste d'URLs complémentaires choisies par l'utilisateur.
    """
    return asyncio.run(_crawl_urls_process(start_url, selected_urls))


async def _crawl_urls_process(start_url: str, selected_urls: List[str]) -> str:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, remove_overlay_elements=True
    )

    all_urls = [start_url] + [u for u in selected_urls if u != start_url]
    merged_markdown_blocks = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for idx, url in enumerate(all_urls, start=1):
            log.info(
                f"[Crawl4AI] Scraping de la page sélectionnée {idx}/{len(all_urls)} : {url}"
            )
            res = await crawler.arun(url=url, config=run_config)

            if res.success:
                md_text = (
                    res.markdown.raw_markdown
                    if hasattr(res.markdown, "raw_markdown")
                    else str(res.markdown)
                )
                if md_text and len(md_text.strip()) > 30:
                    merged_markdown_blocks.append(
                        f"--- [Page {idx} : {url}] ---\n{md_text.strip()}"
                    )

    return "\n\n".join(merged_markdown_blocks)


def crawl_and_accumulate_sync(
    urls_to_crawl: List[str], already_visited_urls: set
) -> tuple[str, List[Dict[str, str]], set]:
    """
    Wrapper synchrone pour exécuter la passe d'exploration sur une liste d'URLs.
    """
    return asyncio.run(
        _crawl_and_accumulate_process(urls_to_crawl, already_visited_urls)
    )


# Dans app/ingestion/web_scrapy.py


async def _crawl_and_accumulate_process(
    urls_to_crawl: List[str], already_visited_urls: set
) -> tuple[str, List[Dict[str, str]], set]:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, remove_overlay_elements=True
    )

    # Filtrer les URLs déjà visitées
    urls_to_fetch = [u for u in urls_to_crawl if u not in already_visited_urls]
    if not urls_to_fetch:
        return "", [], already_visited_urls

    updated_visited = set(already_visited_urls)
    for u in urls_to_fetch:
        updated_visited.add(u)

    new_markdown_blocks = []
    discovered_links = []

    async with AsyncWebCrawler(config=browser_config) as crawler:
        log.info(
            f"[Crawl4AI Parallel] Lancement du scraping en parallèle pour {len(urls_to_fetch)} URL(s)..."
        )

        # --- MAGIE DE LA PERFORMANCE : arun_many télécharge tout en simultané ---
        results = await crawler.arun_many(urls=urls_to_fetch, config=run_config)

        for res in results:
            if res.success:
                url = res.url
                md_text = (
                    res.markdown.raw_markdown
                    if hasattr(res.markdown, "raw_markdown")
                    else str(res.markdown)
                )
                if md_text and len(md_text.strip()) > 30:
                    new_markdown_blocks.append(
                        f"--- [Source : {url}] ---\n{md_text.strip()}"
                    )

                    # Extraction des liens de cette page
                    links_on_page = extract_markdown_links(md_text, url)
                    for l in links_on_page:
                        if l["url"] not in updated_visited:
                            discovered_links.append(l)

    # Déduplication des liens découverts
    unique_discovered = []
    seen = set()
    for l in discovered_links:
        if l["url"] not in seen and l["url"] not in updated_visited:
            seen.add(l["url"])
            unique_discovered.append(l)

    accumulated_text = "\n\n".join(new_markdown_blocks)
    return accumulated_text, unique_discovered, updated_visited
