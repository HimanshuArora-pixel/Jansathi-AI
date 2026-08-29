"""
Central utility to strip <think> tags from LLM responses.
Qwen and other reasoning models output internal chain-of-thought inside
<think>...</think> blocks. This strips them before showing the user.
"""
import re

def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    if not text:
        return text
    # Remove think blocks (including multi-line)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also strip any orphaned opening/closing tags
    cleaned = re.sub(r'</?think>', '', cleaned)
    return cleaned.strip()

def inject_verified_links(text: str) -> str:
    """Inject markdown links for known entities if they aren't already linked."""
    if not text:
        return text
    
    try:
        from utils.link_dictionary import VERIFIED_LINKS
    except ImportError:
        return text
        
    for name, url in VERIFIED_LINKS.items():
        # Match the word boundary, ensure it's not followed by ]( which indicates it's already a markdown link
        pattern = re.compile(rf'\b({re.escape(name)})\b(?!\s*\]\()', re.IGNORECASE)
        # If the name is already inside a markdown link's text block e.g. [some NALSA link](url), 
        # this might still match, but since the AI shouldn't be generating URLs anymore, it's safe.
        text = pattern.sub(rf'[\1]({url})', text)
        
    return text
