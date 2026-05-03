"""Region utilities for GeoIA stadium detection.

Provides region bounds lookup and country/region definitions for flexible detection anywhere.
"""
from typing import Dict, Tuple, Optional

# Region bounds: (min_lon, min_lat, max_lon, max_lat)
REGIONS = {
    "madagascar": {
        "bounds": (41.5, -26.0, 50.5, -11.0),
        "title": "Madagascar",
        "center": (45.5, -18.5),
    },
    "usa": {
        "bounds": (-125.0, 24.5, -66.9, 49.4),
        "title": "United States",
        "center": (-95.7, 37.1),
    },
    "california": {
        "bounds": (-124.5, 32.5, -114.1, 42.0),
        "title": "California",
        "center": (-119.3, 37.3),
    },
    "texas": {
        "bounds": (-106.6, 25.8, -93.5, 36.5),
        "title": "Texas",
        "center": (-99.7, 31.2),
    },
    "africa": {
        "bounds": (-18.0, -35.0, 52.0, 37.0),
        "title": "Africa",
        "center": (17.0, 1.0),
    },
    "europe": {
        "bounds": (-10.0, 35.0, 40.0, 71.0),
        "title": "Europe",
        "center": (15.0, 52.0),
    },
    "asia": {
        "bounds": (26.0, -10.0, 150.0, 70.0),
        "title": "Asia",
        "center": (90.0, 30.0),
    },
    "australia": {
        "bounds": (113.0, -44.0, 154.0, -10.0),
        "title": "Australia",
        "center": (133.5, -27.0),
    },
    "south_america": {
        "bounds": (-82.0, -56.0, -35.0, 13.0),
        "title": "South America",
        "center": (-58.5, -20.0),
    },
    "world": {
        "bounds": (-180.0, -85.0, 180.0, 85.0),
        "title": "World",
        "center": (0.0, 0.0),
    },
}


def list_regions() -> list:
    """Return list of available region names."""
    return list(REGIONS.keys())


def get_region(region_name: str) -> Optional[Dict]:
    """Get region definition by name.
    
    Args:
        region_name: Name of region (case-insensitive)
    
    Returns:
        Dict with 'bounds', 'title', 'center' keys, or None if not found.
    """
    return REGIONS.get(region_name.lower())


def get_bounds(region_name: str) -> Optional[Tuple[float, float, float, float]]:
    """Get bounding box (min_lon, min_lat, max_lon, max_lat) for a region.
    
    Args:
        region_name: Name of region (case-insensitive)
    
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) or None if not found.
    """
    region = get_region(region_name)
    if region:
        return region["bounds"]
    return None


def get_title(region_name: str) -> str:
    """Get display title for a region.
    
    Args:
        region_name: Name of region (case-insensitive)
    
    Returns:
        Human-readable region title, or the region name if not found.
    """
    region = get_region(region_name)
    if region:
        return region["title"]
    return region_name.replace("_", " ").title()


def add_custom_region(name: str, bounds: Tuple[float, float, float, float], title: str = None, center: Tuple[float, float] = None):
    """Register a custom region.
    
    Args:
        name: Region name (lowercase recommended)
        bounds: (min_lon, min_lat, max_lon, max_lat)
        title: Human-readable title (auto-generated if None)
        center: Center point (auto-calculated if None)
    """
    if title is None:
        title = name.replace("_", " ").title()
    if center is None:
        min_lon, min_lat, max_lon, max_lat = bounds
        center = ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)
    
    REGIONS[name.lower()] = {
        "bounds": bounds,
        "title": title,
        "center": center,
    }


if __name__ == "__main__":
    # List all regions
    print("Available regions:")
    for name in list_regions():
        region = get_region(name)
        bounds = region["bounds"]
        print(f"  {name}: {bounds}")
