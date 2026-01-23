"""
Performance Benchmark Script

Inference süresini ölçer ve performans metriklerini raporlar.
"""

import time
import json
import numpy as np
from typing import List
from onuion.inference import InferencePipeline


def generate_test_sessions(n: int = 1000) -> List[dict]:
    """Test için session data'ları üretir."""
    sessions = []
    
    for i in range(n):
        session = {
            "current_ip": f"192.168.1.{100 + i % 50}",
            "initial_ip": f"192.168.1.{50 + i % 50}",
            "ip_history": [f"192.168.1.{j}" for j in range(50, 100, 10)],
            "current_geo": {
                "country": "TR",
                "city": "Istanbul",
                "timezone": "Europe/Istanbul"
            },
            "initial_geo": {
                "country": "TR",
                "city": "Ankara",
                "timezone": "Europe/Istanbul"
            },
            "current_device": {
                "user_agent": "Mozilla/5.0",
                "screen_resolution": "1920x1080",
                "platform": "Win32",
                "fingerprint": f"fp{i}"
            },
            "initial_device": {
                "user_agent": "Mozilla/5.0",
                "screen_resolution": "1920x1080",
                "platform": "Win32",
                "fingerprint": f"fp{i % 10}"
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
            "requests": [
                {
                    "timestamp": 1706000000 + j,
                    "method": "GET",
                    "endpoint": f"/api/endpoint{j}"
                }
                for j in range(10)
            ],
            "session_duration_seconds": 10.0 + i % 100,
            "current_session_id": f"sess_{i}",
            "initial_session_id": f"sess_{i % 10}",
            "current_cookies": {},
            "initial_cookies": {},
            "current_referrer": "",
            "initial_referrer": ""
        }
        sessions.append(session)
    
    return sessions


def benchmark_single_inference(pipeline: InferencePipeline, n_iterations: int = 1000):
    """Single inference benchmark."""
    sessions = generate_test_sessions(n_iterations)
    
    inference_times = []
    
    print(f"Single inference benchmark başlıyor ({n_iterations} iterasyon)...")
    
    start_total = time.perf_counter()
    
    for session in sessions:
        result = pipeline.analyze(session)
        inference_times.append(result.inference_time_ms)
    
    end_total = time.perf_counter()
    total_time = (end_total - start_total) * 1000  # ms
    
    # İstatistikler
    avg_time = np.mean(inference_times)
    median_time = np.median(inference_times)
    p95_time = np.percentile(inference_times, 95)
    p99_time = np.percentile(inference_times, 99)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    throughput = n_iterations / (total_time / 1000)  # requests per second
    
    print("\n" + "=" * 60)
    print("SINGLE INFERENCE BENCHMARK SONUÇLARI")
    print("=" * 60)
    print(f"Toplam iterasyon: {n_iterations}")
    print(f"Toplam süre: {total_time:.2f} ms")
    print(f"Ortalama inference süresi: {avg_time:.3f} ms")
    print(f"Median inference süresi: {median_time:.3f} ms")
    print(f"P95 inference süresi: {p95_time:.3f} ms")
    print(f"P99 inference süresi: {p99_time:.3f} ms")
    print(f"Min inference süresi: {min_time:.3f} ms")
    print(f"Max inference süresi: {max_time:.3f} ms")
    print(f"Throughput: {throughput:.0f} requests/second")
    print("=" * 60)
    
    return {
        "n_iterations": n_iterations,
        "total_time_ms": total_time,
        "avg_time_ms": avg_time,
        "median_time_ms": median_time,
        "p95_time_ms": p95_time,
        "p99_time_ms": p99_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "throughput_rps": throughput
    }


def benchmark_batch_inference(pipeline: InferencePipeline, batch_sizes: List[int] = [10, 50, 100, 500]):
    """Batch inference benchmark."""
    print("\n" + "=" * 60)
    print("BATCH INFERENCE BENCHMARK")
    print("=" * 60)
    
    results = {}
    
    for batch_size in batch_sizes:
        sessions = generate_test_sessions(batch_size)
        
        start_time = time.perf_counter()
        results_batch = pipeline.analyze_batch(sessions)
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000  # ms
        avg_time_per_item = total_time / batch_size
        throughput = batch_size / (total_time / 1000)
        
        results[batch_size] = {
            "total_time_ms": total_time,
            "avg_time_per_item_ms": avg_time_per_item,
            "throughput_rps": throughput
        }
        
        print(f"\nBatch Size: {batch_size}")
        print(f"  Toplam süre: {total_time:.2f} ms")
        print(f"  Öğe başına ortalama: {avg_time_per_item:.3f} ms")
        print(f"  Throughput: {throughput:.0f} requests/second")
    
    return results


def main():
    """Main benchmark function."""
    print("onuion Performance Benchmark")
    print("=" * 60)
    
    # Pipeline oluştur
    pipeline = InferencePipeline()
    
    # Single inference benchmark
    single_results = benchmark_single_inference(pipeline, n_iterations=1000)
    
    # Batch inference benchmark
    batch_results = benchmark_batch_inference(pipeline, batch_sizes=[10, 50, 100, 500])
    
    # Sonuçları JSON'a kaydet
    all_results = {
        "single_inference": single_results,
        "batch_inference": batch_results
    }
    
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\nBenchmark sonuçları 'benchmark_results.json' dosyasına kaydedildi.")


if __name__ == "__main__":
    main()

