#!/bin/bash
# Test Sample 4 (ragas-cuad-0021) with t15_experimental mode

QUESTION='Under the First Amended and Restated License Agreement dated September 16, 2019, how does the termination of the media license agreement impact the rights and obligations of the Village Media Company and Hall of Fame Media Group, LLC, particularly in relation to their shared media-related opportunities and exclusive communication protocols with the National Football League and its affiliated entities?'

# Using the local YouTu API endpoint
curl -s -X POST "http://127.0.0.1:8000/api/ask-question/ragas-cuad-0021-test-$(date +%s)" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"${QUESTION}\", \"dataset\": \"cuad_v3\", \"graph_type\": \"default\", \"route\": \"auto\"}" | python3 -m json.tool 2>/dev/null || echo "API request failed"
