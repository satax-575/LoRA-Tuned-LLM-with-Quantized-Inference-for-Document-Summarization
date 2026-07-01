"""
Pytest configuration and shared fixtures.
"""

import sys
import os
from pathlib import Path

import pytest

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))


# ── Pytest Async Mode ─────────────────────────────────────

def pytest_configure(config):
    """Configure asyncio mode for pytest-asyncio."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


# ── Shared fixtures ───────────────────────────────────────

@pytest.fixture(scope="session")
def sample_article():
    return (
        "Scientists at MIT and Stanford have developed a revolutionary new battery "
        "technology that could transform electric vehicles and renewable energy storage. "
        "The new lithium-sulfur battery design uses a novel cathode material that prevents "
        "the common problem of sulfur dissolving into the electrolyte. In lab tests, the new "
        "batteries maintained 80% capacity after 1,500 charge cycles, comparable to the best "
        "lithium-ion batteries available today but at a fraction of the cost. The breakthrough "
        "was published in Nature Energy and funded by the Department of Energy."
    )


@pytest.fixture(scope="session")
def sample_summary():
    return (
        "MIT and Stanford scientists developed a lithium-sulfur battery maintaining 80% "
        "capacity for 1,500 cycles, enabling 600-mile EV range at lower cost."
    )


@pytest.fixture(scope="session")
def sample_articles():
    return [
        "Climate scientists have issued a stark warning that global temperatures will "
        "exceed the 1.5°C threshold within the next decade unless emissions are drastically "
        "cut. The new report from the IPCC cites accelerating Arctic ice melt, rising sea "
        "levels, and more frequent extreme weather events as evidence that climate change is "
        "advancing faster than previous models predicted. World leaders are being urged to "
        "accelerate their transition to renewable energy and implement carbon pricing mechanisms.",

        "The Federal Reserve raised its benchmark interest rate by a quarter percentage "
        "point on Wednesday, the tenth increase in just over a year. Fed Chair Jerome Powell "
        "said officials were watching economic data carefully and noted that the labor market "
        "remains very tight with unemployment at historically low levels. Consumer prices rose "
        "4.9% in April from a year earlier, still more than double the Fed's 2% target.",
    ]
