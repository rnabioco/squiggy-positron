#!/usr/bin/env python3
"""
Simple test script for the JSON-RPC server

Sends test requests to the server via stdin/stdout to verify it works.
"""

import json
import subprocess
import sys

def send_request(proc, method, params=None, request_id=1):
    """Send a JSON-RPC request and get response"""
    request = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or {},
        "id": request_id
    }

    # Send request
    request_json = json.dumps(request) + "\n"
    proc.stdin.write(request_json)
    proc.stdin.flush()

    # Read response
    response_line = proc.stdout.readline()
    response = json.loads(response_line)

    return response


def main():
    """Run basic tests on the JSON-RPC server"""
    print("Starting JSON-RPC server test...")

    # Start server process
    proc = subprocess.Popen(
        [sys.executable, "src/python/server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        # Test 1: Invalid method
        print("\nTest 1: Invalid method")
        response = send_request(proc, "invalid_method", {}, 1)
        print(f"Response: {json.dumps(response, indent=2)}")
        assert "error" in response
        assert response["error"]["code"] == -32601
        print("✓ Passed")

        # Test 2: Valid method with missing file (should error)
        print("\nTest 2: open_pod5 with missing file")
        response = send_request(proc, "open_pod5", {"file_path": "/nonexistent.pod5"}, 2)
        print(f"Response: {json.dumps(response, indent=2)}")
        assert "error" in response
        print("✓ Passed")

        # Test 3: get_references without BAM file
        print("\nTest 3: get_references without BAM file")
        response = send_request(proc, "get_references", {}, 3)
        print(f"Response: {json.dumps(response, indent=2)}")
        assert "error" in response
        print("✓ Passed")

        print("\n✅ All tests passed!")

    finally:
        # Terminate server
        proc.terminate()
        proc.wait(timeout=5)

        # Print server stderr
        stderr = proc.stderr.read()
        if stderr:
            print("\nServer stderr:")
            print(stderr)


if __name__ == "__main__":
    main()
