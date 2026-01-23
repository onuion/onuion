"""
Training Data Generator

Generates synthetic session data in JSON format for training.
Creates various risk scenarios: low risk, medium risk, high risk.
"""

import json
import random
import os
from typing import Dict, Any, List
from datetime import datetime, timedelta


def generate_ip() -> str:
    """Generates a random IP address."""
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"


def generate_fingerprint() -> str:
    """Generates a random device fingerprint."""
    chars = "abcdef0123456789"
    return "".join(random.choice(chars) for _ in range(12))


def generate_session_id() -> str:
    """Generates a random session ID."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "sess_" + "".join(random.choice(chars) for _ in range(8))


def generate_user_agent() -> str:
    """Generates a random user agent."""
    browsers = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
    ]
    return random.choice(browsers)


def generate_geo(country: str = None, city: str = None) -> Dict[str, str]:
    """Generates geo-location data."""
    countries = {
        "TR": ["Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"],
        "US": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        "DE": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne"],
        "FR": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"],
        "GB": ["London", "Manchester", "Birmingham", "Liverpool", "Leeds"],
        "JP": ["Tokyo", "Osaka", "Yokohama", "Nagoya", "Sapporo"]
    }
    
    if country is None:
        country = random.choice(list(countries.keys()))
    
    if city is None:
        city = random.choice(countries[country])
    
    timezones = {
        "TR": "Europe/Istanbul",
        "US": "America/New_York",
        "DE": "Europe/Berlin",
        "FR": "Europe/Paris",
        "GB": "Europe/London",
        "JP": "Asia/Tokyo"
    }
    
    return {
        "country": country,
        "city": city,
        "timezone": timezones.get(country, "UTC")
    }


def generate_requests(count: int, start_timestamp: int, method_ratio: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """Generates request list."""
    if method_ratio is None:
        method_ratio = {"GET": 0.7, "POST": 0.2, "PUT": 0.05, "DELETE": 0.05}
    
    endpoints = [
        "/api/users", "/api/login", "/api/dashboard", "/api/profile",
        "/api/products", "/api/orders", "/api/cart", "/api/search",
        "/api/logout", "/api/settings", "/api/notifications", "/api/messages"
    ]
    
    requests = []
    current_time = start_timestamp
    
    for i in range(count):
        # Select method based on ratio
        rand = random.random()
        cumulative = 0
        method = "GET"
        for m, ratio in method_ratio.items():
            cumulative += ratio
            if rand <= cumulative:
                method = m
                break
        
        requests.append({
            "timestamp": current_time,
            "method": method,
            "endpoint": random.choice(endpoints)
        })
        
        # Time between requests (0.1 to 5 seconds for normal, faster for bots)
        interval = random.uniform(0.1, 5.0)
        current_time += int(interval)
    
    return requests


def generate_low_risk_session() -> Dict[str, Any]:
    """Generates a low-risk session (normal user behavior)."""
    base_ip = generate_ip()
    base_geo = generate_geo("TR", "Istanbul")
    fingerprint = generate_fingerprint()
    session_id = generate_session_id()
    user_agent = generate_user_agent()
    
    start_time = int(datetime.now().timestamp())
    duration = random.uniform(30, 300)  # 30 seconds to 5 minutes
    request_count = random.randint(5, 20)
    
    return {
        "current_ip": base_ip,
        "initial_ip": base_ip,  # Same IP
        "ip_history": [base_ip],
        "current_geo": base_geo,
        "initial_geo": base_geo,  # Same geo
        "current_device": {
            "user_agent": user_agent,
            "screen_resolution": "1920x1080",
            "platform": "Win32",
            "fingerprint": fingerprint
        },
        "initial_device": {
            "user_agent": user_agent,
            "screen_resolution": "1920x1080",
            "platform": "Win32",
            "fingerprint": fingerprint
        },
        "current_browser": {
            "name": "Chrome",
            "version": "120.0",
            "language": "tr-TR"
        },
        "initial_browser": {
            "name": "Chrome",
            "version": "120.0",
            "language": "tr-TR"
        },
        "requests": generate_requests(request_count, start_time),
        "session_duration_seconds": duration,
        "current_session_id": session_id,
        "initial_session_id": session_id,  # Same session
        "current_cookies": {
            "session_id": session_id
        },
        "initial_cookies": {
            "session_id": session_id
        },
        "current_referrer": "https://example.com",
        "initial_referrer": "https://example.com"
    }


def generate_medium_risk_session() -> Dict[str, Any]:
    """Generates a medium-risk session (some suspicious activity)."""
    initial_ip = generate_ip()
    current_ip = generate_ip()  # Different IP
    initial_geo = generate_geo("TR", "Istanbul")
    current_geo = generate_geo("TR", "Ankara")  # Same country, different city
    fingerprint = generate_fingerprint()
    session_id = generate_session_id()
    user_agent = generate_user_agent()
    
    start_time = int(datetime.now().timestamp())
    duration = random.uniform(10, 120)  # 10 seconds to 2 minutes
    request_count = random.randint(15, 50)  # More requests
    
    return {
        "current_ip": current_ip,
        "initial_ip": initial_ip,
        "ip_history": [initial_ip, current_ip],
        "current_geo": current_geo,
        "initial_geo": initial_geo,
        "current_device": {
            "user_agent": user_agent,
            "screen_resolution": "1920x1080",
            "platform": "Win32",
            "fingerprint": fingerprint
        },
        "initial_device": {
            "user_agent": user_agent,
            "screen_resolution": "1920x1080",
            "platform": "Win32",
            "fingerprint": fingerprint
        },
        "current_browser": {
            "name": "Chrome",
            "version": "120.0",
            "language": "tr-TR"
        },
        "initial_browser": {
            "name": "Chrome",
            "version": "120.0",
            "language": "tr-TR"
        },
        "requests": generate_requests(request_count, start_time, {"GET": 0.6, "POST": 0.3, "PUT": 0.05, "DELETE": 0.05}),
        "session_duration_seconds": duration,
        "current_session_id": session_id,
        "initial_session_id": session_id,
        "current_cookies": {
            "session_id": session_id,
            "csrf_token": "token_xyz"
        },
        "initial_cookies": {
            "session_id": session_id
        },
        "current_referrer": "https://example.com",
        "initial_referrer": "https://example.com"
    }


def generate_high_risk_session() -> Dict[str, Any]:
    """Generates a high-risk session (suspicious activity)."""
    initial_ip = generate_ip()
    current_ip = generate_ip()  # Different IP
    # Multiple IP changes
    ip_history = [initial_ip]
    for _ in range(random.randint(2, 5)):
        ip_history.append(generate_ip())
    ip_history.append(current_ip)
    
    initial_geo = generate_geo("TR", "Istanbul")
    current_geo = generate_geo("US", "New York")  # Different country!
    
    initial_fingerprint = generate_fingerprint()
    current_fingerprint = generate_fingerprint()  # Different fingerprint
    
    initial_session_id = generate_session_id()
    current_session_id = generate_session_id()  # Different session ID
    
    initial_user_agent = generate_user_agent()
    current_user_agent = generate_user_agent()  # Different user agent
    
    start_time = int(datetime.now().timestamp())
    duration = random.uniform(5, 60)  # Very short duration
    request_count = random.randint(50, 200)  # Many requests (bot-like)
    
    # Bot-like request pattern (very fast)
    requests = []
    current_time = start_time
    for i in range(request_count):
        requests.append({
            "timestamp": current_time,
            "method": random.choice(["GET", "POST", "PUT", "DELETE"]),
            "endpoint": random.choice(["/api/users", "/api/login", "/api/products"])
        })
        current_time += random.randint(0, 2)  # Very fast requests
    
    return {
        "current_ip": current_ip,
        "initial_ip": initial_ip,
        "ip_history": ip_history,
        "current_geo": current_geo,
        "initial_geo": initial_geo,
        "current_device": {
            "user_agent": current_user_agent,
            "screen_resolution": "1920x1080",
            "platform": "Win32",
            "fingerprint": current_fingerprint
        },
        "initial_device": {
            "user_agent": initial_user_agent,
            "screen_resolution": "1366x768",  # Different resolution
            "platform": "Win32",
            "fingerprint": initial_fingerprint
        },
        "current_browser": {
            "name": "Firefox",
            "version": "121.0",
            "language": "en-US"
        },
        "initial_browser": {
            "name": "Chrome",
            "version": "120.0",
            "language": "tr-TR"
        },
        "requests": requests,
        "session_duration_seconds": duration,
        "current_session_id": current_session_id,
        "initial_session_id": initial_session_id,
        "current_cookies": {
            "session_id": current_session_id,
            "csrf_token": "token_xyz",
            "auth_token": "auth_abc"
        },
        "initial_cookies": {
            "session_id": initial_session_id
        },
        "current_referrer": "https://suspicious-site.com",
        "initial_referrer": "https://example.com"
    }


def generate_training_data(
    output_dir: str = "data/training_sessions",
    n_low_risk: int = 1000,
    n_medium_risk: int = 500,
    n_high_risk: int = 500
):
    """
    Generates training data in JSON format.
    
    Args:
        output_dir: Output directory for JSON files
        n_low_risk: Number of low-risk sessions
        n_medium_risk: Number of medium-risk sessions
        n_high_risk: Number of high-risk sessions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating training data...")
    print(f"  Low risk: {n_low_risk}")
    print(f"  Medium risk: {n_medium_risk}")
    print(f"  High risk: {n_high_risk}")
    print(f"  Output directory: {output_dir}")
    
    # Generate low-risk sessions
    for i in range(n_low_risk):
        session = generate_low_risk_session()
        filename = os.path.join(output_dir, f"low_risk_{i:05d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_low_risk} low-risk sessions")
    
    # Generate medium-risk sessions
    for i in range(n_medium_risk):
        session = generate_medium_risk_session()
        filename = os.path.join(output_dir, f"medium_risk_{i:05d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_medium_risk} medium-risk sessions")
    
    # Generate high-risk sessions
    for i in range(n_high_risk):
        session = generate_high_risk_session()
        filename = os.path.join(output_dir, f"high_risk_{i:05d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_high_risk} high-risk sessions")
    
    print(f"\n[OK] Training data generation complete!")
    print(f"  Total sessions: {n_low_risk + n_medium_risk + n_high_risk}")
    print(f"  Files saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data for onuion")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training_sessions",
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--n-low-risk",
        type=int,
        default=1000,
        help="Number of low-risk sessions"
    )
    parser.add_argument(
        "--n-medium-risk",
        type=int,
        default=500,
        help="Number of medium-risk sessions"
    )
    parser.add_argument(
        "--n-high-risk",
        type=int,
        default=500,
        help="Number of high-risk sessions"
    )
    
    args = parser.parse_args()
    
    generate_training_data(
        output_dir=args.output_dir,
        n_low_risk=args.n_low_risk,
        n_medium_risk=args.n_medium_risk,
        n_high_risk=args.n_high_risk
    )

