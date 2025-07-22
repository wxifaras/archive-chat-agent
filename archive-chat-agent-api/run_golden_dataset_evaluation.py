#!/usr/bin/env python
"""
Script to run golden dataset evaluation against the Archive Chat Agent API.
This script calls the actual API endpoints to perform real evaluation.
"""

import requests
import time
import json
import sys
import os
from datetime import datetime
import glob
import csv
from pathlib import Path

def load_env_file(env_path=".env"):
    """Load environment variables from .env file."""
    env_vars = {}
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Split on first = only
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip().strip('"').strip("'")
                        env_vars[key.strip()] = value
    
    return env_vars

def check_api_health(base_url):
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{base_url}/")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def trigger_golden_dataset_evaluation(base_url):
    """Trigger the golden dataset evaluation via API."""
    endpoint = f"{base_url}/api/evaluate/golden-dataset"
    
    print(f"Triggering golden dataset evaluation at: {endpoint}")
    
    try:
        # First, check the golden dataset exists and get question count
        csv_path = "tests/data/golden_dataset_template.csv"
        try:
            question_count = 0
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    question_count = sum(1 for row in reader if row.get('question'))
                # Load env vars for config values
                env_vars = load_env_file()
                max_concurrent = os.environ.get('EVALUATION_MAX_CONCURRENT') or env_vars.get('EVALUATION_MAX_CONCURRENT', '5')
                eval_temp = os.environ.get('EVALUATION_TEMPERATURE') or env_vars.get('EVALUATION_TEMPERATURE', '0.0')
                print(f"\n📊 Golden dataset contains {question_count} questions")
                print(f"   With EVALUATION_MAX_CONCURRENT={max_concurrent} parallel evaluations")
                print(f"   Using EVALUATION_TEMPERATURE={eval_temp}")
        except:
            pass
        
        response = requests.post(endpoint, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get('task_id')
            print(f"\n✓ Evaluation started successfully")
            print(f"  Status: {result.get('status')}")
            print(f"  Task ID: {task_id}")
            print(f"  Message: {result.get('message')}")
            print(f"  Output will be saved to: tests/validation_results/validation_results_*.json")
            return True, task_id
        else:
            print(f"\n✗ Failed to start evaluation")
            print(f"  Status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"✗ Error triggering evaluation: {str(e)}")
        return False, None

def monitor_evaluation_status(base_url, task_id=None, max_wait_minutes=30):
    """Monitor the evaluation status until completion with detailed progress."""
    # Use task-specific endpoint if task_id is provided
    if task_id:
        endpoint = f"{base_url}/api/evaluate/status/{task_id}"
    else:
        endpoint = f"{base_url}/api/evaluate/status"
    
    print(f"\nMonitoring evaluation progress...")
    print(f"Timeout set to {max_wait_minutes} minutes")
    print(f"{'-'*60}")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    last_active_count = None
    last_status_time = start_time
    status_interval = 10  # Show status every 10 seconds
    evaluation_started = False
    zero_count_streak = 0  # Track consecutive zero counts
    
    while time.time() - start_time < max_wait_seconds:
        try:
            response = requests.get(endpoint)
            
            if response.status_code == 200:
                status = response.json()
                elapsed = int(time.time() - start_time)
                
                # Handle task-specific endpoint response
                if task_id:
                    # For task-specific endpoint, response is the task info directly
                    task_info = status
                    is_complete = task_info.get('is_complete', False)
                    active_count = 0 if is_complete else 1
                    tasks_info = {task_id: task_info}
                else:
                    # For general status endpoint
                    active_count = status.get('active_evaluations', 0)
                    tasks_info = status.get('tasks', {})
                    # Filter to only show tasks for current run if we have a task_id
                    if task_id and task_id in tasks_info:
                        tasks_info = {task_id: tasks_info[task_id]}
                
                # Track if evaluation has started
                if active_count > 0:
                    evaluation_started = True
                    zero_count_streak = 0
                elif evaluation_started and active_count == 0:
                    zero_count_streak += 1
                
                # Show detailed status every interval or when count changes
                if active_count != last_active_count or (time.time() - last_status_time) >= status_interval:
                    minutes = elapsed // 60
                    seconds = elapsed % 60
                    
                    if active_count > 0:
                        print(f"\n  [{minutes:02d}:{seconds:02d}] Evaluations in progress: {active_count}")
                        
                        # Show detailed progress for each task
                        for tid, task_info in tasks_info.items():
                            if task_info and not task_info.get('is_complete', True):
                                total = task_info.get('total_questions', 0)
                                completed = task_info.get('completed', 0)
                                failed = task_info.get('failed', 0)
                                in_progress = task_info.get('in_progress', 0)
                                progress_pct = task_info.get('progress_percentage', 0)
                                
                                print(f"\n           Task Progress: {completed + failed}/{total} questions ({progress_pct}% complete)")
                                print(f"           - Total Tests: {total}")
                                print(f"           - Completed: {completed}")
                                print(f"           - Failed: {failed}")
                                print(f"           - In Progress: {in_progress}")
                                pending = total - (completed + failed + in_progress)
                                if pending > 0:
                                    print(f"           - Pending: {pending}")
                                
                                # Show current questions being processed
                                current_questions = task_info.get('current_questions', [])
                                if current_questions:
                                    print(f"\n           Currently processing:")
                                    for q in current_questions[:3]:  # Show up to 3
                                        print(f"           • {q['set_id']}/{q['question_id']}: {q['question'][:60]}...")
                    else:
                        if not evaluation_started:
                            print(f"  [{minutes:02d}:{seconds:02d}] Waiting for evaluation to start...")
                        else:
                            print(f"  [{minutes:02d}:{seconds:02d}] No active evaluations (checking for completion...)")
                    
                    last_status_time = time.time()
                    last_active_count = active_count
                
                # Check if evaluation is complete
                # Complete if: evaluation started, no active tasks, and we've seen zero for a few checks
                if evaluation_started and active_count == 0 and zero_count_streak >= 3:
                    # Show final summary from completed tasks
                    print(f"\n{'='*60}")
                    print(f"EVALUATION SUMMARY")
                    print(f"{'='*60}")
                    
                    # Get final task info - use the already filtered tasks_info
                    for tid, task_info in tasks_info.items():
                        if task_info and task_info.get('is_complete', False):
                            total = task_info.get('total_questions', 0)
                            completed = task_info.get('completed', 0)
                            failed = task_info.get('failed', 0)
                            progress_pct = task_info.get('progress_percentage', 0)
                            
                            print(f"\n✓ All tests completed in {elapsed} seconds!")
                            print(f"\n  Final Results:")
                            print(f"  - Total Tests Run: {total}")
                            print(f"  - Tests Passed: {completed}")
                            print(f"  - Tests Failed: {failed}")
                            print(f"  - Success Rate: {(completed/total*100) if total > 0 else 0:.1f}%")
                            
                            if task_info.get('results_file'):
                                results_file = task_info.get('results_file')
                                # Normalize the path to use forward slashes for display
                                normalized_path = results_file.replace('\\', '/')
                                abs_path = os.path.abspath(results_file)
                                file_url = f"file://{abs_path}"
                                hyperlink = f"\033]8;;{file_url}\033\\{normalized_path}\033]8;;\033\\"
                                print(f"\n  Results saved to: {hyperlink}")
                            
                            # Show performance metrics
                            avg_time = elapsed / total if total > 0 else 0
                            # Load env vars for config values
                            env_vars = load_env_file()
                            max_concurrent = os.environ.get('EVALUATION_MAX_CONCURRENT') or env_vars.get('EVALUATION_MAX_CONCURRENT', '5')
                            print(f"\n  Performance:")
                            print(f"  - Total Time: {elapsed} seconds")
                            print(f"  - Average Time per Test: {avg_time:.1f} seconds")
                            print(f"  - Parallel Workers: {max_concurrent}")
                            
                            break  # Only show the first completed task
                    
                    print(f"\n{'='*60}")
                    
                    # Wait a bit more for files to be written
                    time.sleep(3)
                    return True
                    
            else:
                print(f"\n✗ Error checking status: {response.status_code}")
                
        except Exception as e:
            print(f"\n✗ Error monitoring status: {str(e)}")
        
        time.sleep(2)  # Check every 2 seconds
    
    print(f"\n✗ Evaluation timed out after {max_wait_minutes} minutes")
    return False

def find_latest_results_file():
    """Find the most recently created validation results file."""
    result_files = glob.glob("tests/validation_results/validation_results_*.json")
    
    if not result_files:
        return None
    
    # Sort by modification time, most recent first
    result_files.sort(key=os.path.getmtime, reverse=True)
    return result_files[0]

def display_evaluation_results(results_file):
    """Display the evaluation results from the JSON file."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Create clickable file path (works in terminals that support OSC 8)
    # Normalize the path to use forward slashes for display
    normalized_path = results_file.replace('\\', '/')
    abs_path = os.path.abspath(results_file)
    file_url = f"file://{abs_path}"
    # OSC 8 hyperlink format: \033]8;;URL\033\\TEXT\033]8;;\033\\
    hyperlink = f"\033]8;;{file_url}\033\\{normalized_path}\033]8;;\033\\"
    print(f"Results file: {hyperlink}")
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both dict with 'results' key and direct list format
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            summary = data.get('summary', {})
        else:
            results = data
            # Calculate summary
            total = len(results)
            five_stars = sum(1 for r in results if r.get('stars') == 5)
            three_stars = sum(1 for r in results if r.get('stars') == 3)
            one_star = sum(1 for r in results if r.get('stars') == 1)
            
            summary = {
                'total_evaluations': total,
                'five_star_count': five_stars,
                'three_star_count': three_stars,
                'one_star_count': one_star,
                'five_star_percentage': (five_stars/total*100) if total > 0 else 0,
                'three_star_percentage': (three_stars/total*100) if total > 0 else 0,
                'one_star_percentage': (one_star/total*100) if total > 0 else 0
            }
        
        # Display summary
        print(f"\nRESULTS SUMMARY:")
        print(f"  Total evaluations: {summary.get('total_evaluations', 0)}")
        print(f"  ⭐⭐⭐⭐⭐ (5-star): {summary.get('five_star_count', 0)} ({summary.get('five_star_percentage', 0):.1f}%)")
        print(f"  ⭐⭐⭐ (3-star): {summary.get('three_star_count', 0)} ({summary.get('three_star_percentage', 0):.1f}%)")
        print(f"  ⭐ (1-star): {summary.get('one_star_count', 0)} ({summary.get('one_star_percentage', 0):.1f}%)")
        
        # Show warning if seeing mock answers
        mock_count = sum(1 for r in results if 'mock answer' in r.get('answer', '').lower())
        if mock_count > 0:
            print(f"\n⚠️  WARNING: Found {mock_count} mock answers in results!")
            print(f"   This suggests the chat API may not be returning real answers.")
            print(f"   Please ensure:")
            print(f"   1. Your Azure AI Search index has content")
            print(f"   2. Your Azure OpenAI credentials are configured correctly")
            print(f"   3. The chat endpoint is working properly")
            
    except Exception as e:
        print(f"✗ Error reading results file: {str(e)}")

def main():
    """Main function to run the golden dataset evaluation."""
    api_base_url = "http://localhost:8000"
    
    # Load environment variables from .env file
    env_vars = load_env_file()
    
    print(f"Archive Chat Agent - Golden Dataset Evaluation")
    print(f"{'='*60}")
    print(f"API URL: {api_base_url}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Display model configuration upfront
    # First try OS environment, then fall back to .env file
    rag_model = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME') or env_vars.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'Not configured')
    eval_model = os.environ.get('AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME') or env_vars.get('AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME', 'Not configured')
    eval_temp = os.environ.get('EVALUATION_TEMPERATURE') or env_vars.get('EVALUATION_TEMPERATURE', '0.0')
    print(f"\nModel Configuration:")
    print(f"  RAG Pipeline: {rag_model}")
    print(f"  Evaluation: {eval_model} (temperature={eval_temp})")
    print()
    
    # Step 1: Check API health
    print("Step 1: Checking API health...")
    if not check_api_health(api_base_url):
        print("✗ API is not running or not healthy")
        print("\nPlease start the API server first:")
        print("  cd archive-chat-agent-api")
        print("  python run.py")
        return 1
    
    print("✓ API is healthy\n")
    
    # Step 2: Trigger evaluation
    print("Step 2: Triggering golden dataset evaluation...")
    success, task_id = trigger_golden_dataset_evaluation(api_base_url)
    if not success:
        return 1
    
    # Step 3: Monitor progress
    print("\nStep 3: Monitoring evaluation progress...")
    
    # Display model information
    rag_model = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME') or env_vars.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'Not configured')
    eval_model = os.environ.get('AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME') or env_vars.get('AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME', 'gpt-4.1')
    eval_temp = os.environ.get('EVALUATION_TEMPERATURE') or env_vars.get('EVALUATION_TEMPERATURE', '0.0')
    
    print("\n🤖 Model Configuration:")
    print(f"   - RAG Pipeline Model: {rag_model}")
    print(f"   - Evaluation Model: {eval_model} (temperature={eval_temp})")
    
    print("\n💡 Tips while waiting:")
    print("   - Each question goes through the full RAG pipeline")
    print(f"   - Answers are generated by {rag_model} then evaluated by {eval_model}")
    print("   - Check API logs for detailed processing info")
    print("   - Increase EVALUATION_MAX_CONCURRENT in .env for faster processing")
    
    if not monitor_evaluation_status(api_base_url, task_id=task_id, max_wait_minutes=30):
        print("\nThe evaluation may still be running in the background.")
        print("You can check the status manually at:")
        print(f"  curl {api_base_url}/api/evaluate/status")
        print("\nTo check the latest results file:")
        print("  ls -la validation_results_*.json")
        return 1
    
    # Step 4: Find and display results
    print("\nStep 4: Looking for results file...")
    
    # Wait a moment for file to be written
    time.sleep(2)
    
    results_file = find_latest_results_file()
    if results_file:
        print(f"✓ Found results file: {results_file}")
        display_evaluation_results(results_file)
    else:
        print("✗ No results file found")
        print("  Results files should be in: tests/validation_results/validation_results_YYYYMMDD_HHMMSS.json")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())