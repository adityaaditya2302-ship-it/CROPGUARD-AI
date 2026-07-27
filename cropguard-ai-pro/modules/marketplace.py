"""
CropGuard AI - Marketplace Module (Phase 5)
When disease is detected, show the farmer where to buy the right products.

Features:
  - Product recommendations by disease
  - Nearby dealer locator (Google Maps link)
  - Price comparisons
  - Government subsidy information
  - Online purchase links
"""
from datetime import datetime


# Product catalog (expand with real e-commerce API integration)
PESTICIDE_CATALOG = {
    "Mancozeb 75% WP": {
        "brands":      ["Dithane M-45", "Indofil M-45", "Kavach"],
        "price_range": "₹180-280 per kg",
        "online_url":  "https://www.bijak.in/search?q=mancozeb",
        "subsidy":     "Available under NMSA scheme for small farmers",
        "use_for":     ["Late Blight", "Early Blight", "Downy Mildew", "Anthracnose"],
    },
    "Copper Oxychloride 50% WP": {
        "brands":      ["Blitox 50", "Fytolan", "Kocide"],
        "price_range": "₹200-350 per kg",
        "online_url":  "https://www.bijak.in/search?q=copper+fungicide",
        "subsidy":     "Often subsidized for organic farming",
        "use_for":     ["Bacterial diseases", "Downy Mildew", "Leaf Spots"],
    },
    "Imidacloprid 17.8% SL": {
        "brands":      ["Confidor", "Tatamida", "Admire"],
        "price_range": "₹400-600 per 100ml",
        "online_url":  "https://www.bijak.in/search?q=imidacloprid",
        "subsidy":     "No subsidy (synthetic insecticide)",
        "use_for":     ["Whitefly", "Aphid", "Thrips", "Jassid"],
    },
    "Neem Oil 10000 PPM": {
        "brands":      ["Achook", "Fortune Neem", "Krishibio"],
        "price_range": "₹150-250 per liter",
        "online_url":  "https://www.bijak.in/search?q=neem+oil",
        "subsidy":     "Fully subsidized for organic farmers in many states",
        "use_for":     ["Organic pest control", "All fungal diseases", "Mites"],
    },
    "Tricyclazole 75% WP": {
        "brands":      ["Beam", "Trycalm", "Blasticidin"],
        "price_range": "₹300-450 per 100g",
        "online_url":  "https://www.bijak.in/search?q=tricyclazole",
        "subsidy":     "Available in rice-growing states",
        "use_for":     ["Rice Blast", "Brown Spot"],
    },
    "Emamectin Benzoate 5% SG": {
        "brands":      ["Proclaim", "Nobel"],
        "price_range": "₹800-1200 per 100g",
        "online_url":  "https://www.bijak.in/search?q=emamectin",
        "subsidy":     "No subsidy",
        "use_for":     ["Bollworm", "Fruit Borer", "Armyworm"],
    },
}

# Government scheme catalog
GOVERNMENT_SCHEMES = [
    {
        "name":        "PM-KISAN",
        "benefit":     "₹6,000/year direct income support",
        "eligibility": "All small and marginal farmers with land",
        "apply_at":    "pmkisan.gov.in or nearest CSC",
        "documents":   ["Aadhaar", "Bank account", "Land records"],
    },
    {
        "name":        "PMFBY (Crop Insurance)",
        "benefit":     "Crop loss compensation at 1.5-5% premium",
        "eligibility": "All farmers growing notified crops",
        "apply_at":    "Nearest bank or pmfby.gov.in",
        "documents":   ["Land records", "Bank account", "Sowing certificate"],
    },
    {
        "name":        "Soil Health Card",
        "benefit":     "Free soil testing every 2 years",
        "eligibility": "All farmers",
        "apply_at":    "Nearest KVK or agriculture department",
        "documents":   ["Aadhaar", "Land records"],
    },
    {
        "name":        "PM Krishi Sinchayee Yojana (PMKSY)",
        "benefit":     "Subsidy on drip/sprinkler irrigation (up to 55%)",
        "eligibility": "All farmers, priority to SC/ST/small farmers",
        "apply_at":    "State agriculture department / pmksy.gov.in",
        "documents":   ["Land records", "Bank account", "Quotation from vendor"],
    },
    {
        "name":        "Sub-Mission on Agricultural Mechanization (SMAM)",
        "benefit":     "50-80% subsidy on farm machinery including drones",
        "eligibility": "Small/marginal farmers, FPOs",
        "apply_at":    "State agriculture department",
        "documents":   ["Land records", "Bank account", "Machine quotation"],
    },
    {
        "name":        "Kisan Credit Card (KCC)",
        "benefit":     "Short-term crop loan at 4% interest rate",
        "eligibility": "All farmers",
        "apply_at":    "Nearest bank or cooperative society",
        "documents":   ["Land records", "Bank account", "Aadhaar"],
    },
]


class MarketplaceModule:
    """Provides product recommendations and dealer/scheme lookup."""

    def get_product_recommendations(self, disease_name: str = "", pest_name: str = "") -> dict:
        """Get recommended products for a specific disease/pest."""
        target = (disease_name or pest_name).lower()
        recommendations = []

        for product, info in PESTICIDE_CATALOG.items():
            for use_case in info["use_for"]:
                if target in use_case.lower() or use_case.lower() in target:
                    recommendations.append({
                        "product":     product,
                        "brands":      info["brands"],
                        "price_range": info["price_range"],
                        "online_url":  info["online_url"],
                        "subsidy":     info["subsidy"],
                        "use_for":     info["use_for"],
                    })
                    break

        # If no specific match, return general fungicides
        if not recommendations:
            recommendations = [
                {
                    "product":  "Mancozeb 75% WP",
                    **PESTICIDE_CATALOG["Mancozeb 75% WP"],
                    "note": "General purpose fungicide",
                },
                {
                    "product":  "Neem Oil 10000 PPM",
                    **PESTICIDE_CATALOG["Neem Oil 10000 PPM"],
                    "note": "Organic option",
                },
            ]

        return {
            "success":         True,
            "query":           disease_name or pest_name,
            "recommendations": recommendations,
            "buy_online_tip":  "Compare prices on Bijak, DeHaat, or AgriBazaar for best deals.",
            "note":            "Always check label instructions and pre-harvest intervals before use.",
        }

    def get_dealer_search_url(self, lat: float, lon: float,
                              product: str = "pesticide") -> dict:
        """Generate map search URL for nearby agri dealers."""
        query    = f"agriculture+pesticide+dealer+near+me"
        maps_url = f"https://www.google.com/maps/search/{query}/@{lat},{lon},14z"
        just_dial = f"https://www.justdial.com/search?q={product}+dealer"

        return {
            "google_maps_url": maps_url,
            "justdial_url":    just_dial,
            "tip": "Search 'Krishi Seva Kendra' or 'Agriculture Inputs Shop' on Google Maps.",
        }

    def get_schemes(self, crop: str = None, state: str = None) -> dict:
        """Get applicable government schemes."""
        schemes = GOVERNMENT_SCHEMES
        if crop:
            # Filter crop-specific schemes (simplified)
            pass
        return {
            "success":        True,
            "schemes":        schemes,
            "count":          len(schemes),
            "helpline":       "Kisan Call Center: 1551 or 1800-180-1551 (free, 24x7)",
            "state_dept_tip": "Contact your State Agriculture Department for state-level schemes.",
        }

    def get_market_prices(self, crop: str) -> dict:
        """Link to live mandi prices (external APIs)."""
        crop_lower = crop.lower()
        return {
            "success":      True,
            "crop":         crop,
            "live_price_sources": [
                {
                    "name": "agmarknet.gov.in",
                    "url":  f"https://agmarknet.gov.in/SearchCmmMkt.aspx?Tx_Commodity={crop_lower}",
                    "note": "Government mandi prices (free)",
                },
                {
                    "name": "eNAM",
                    "url":  "https://www.enam.gov.in/web/",
                    "note": "National Agriculture Market for best prices",
                },
                {
                    "name": "Kisan Suvidha App",
                    "url":  "https://play.google.com/store/apps/details?id=in.gov.dacnet.kisansuvidha",
                    "note": "Mobile app with live mandi rates",
                },
            ],
            "tip": "Sell 2-4 weeks after peak harvest for 15-25% better prices.",
        }


# Singleton
_marketplace_instance: MarketplaceModule | None = None

def get_marketplace() -> MarketplaceModule:
    global _marketplace_instance
    if _marketplace_instance is None:
        _marketplace_instance = MarketplaceModule()
    return _marketplace_instance
