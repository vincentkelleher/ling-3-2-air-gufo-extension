#!/usr/bin/env python3
"""Launch disposable one-slot servers; compare restored greedy bytes with cold runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gufo', default='gufo')
parser.add_argument('--model', required=True)
parser.add_argument('--port', type=int, default=8147)
args = parser.parse_args()
prefix = 'Here is one fact about the red fox. ' * 8
original = prefix + 'Where does it live?'
edited = prefix + 'What does it eat?'

def run(prompts, require_restore=False):
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory, 'serve.log')
        with log.open('w') as output:
            proc = subprocess.Popen([args.gufo, 'serve', 'llm', '--model', args.model,
                                     '--port', str(args.port), '--sessions', '1'],
                                    stdout=output, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'LING3_RUNNER_TRACE': '1'})
        results = []
        try:
            for _ in range(240):
                if 'event=listening' in log.read_text():
                    break
                if proc.poll() is not None:
                    raise RuntimeError(log.read_text()[-2000:])
                time.sleep(.5)
            else:
                raise TimeoutError('server did not listen')
            for index, prompt in enumerate(prompts):
                log_start = log.stat().st_size
                body = {'prompt': prompt, 'model': args.model, 'temperature': 0, 'max_tokens': 12}
                req = Request(f'http://127.0.0.1:{args.port}/v1/completions',
                              data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
                with urlopen(req, timeout=180) as resp:
                    result = json.load(resp)
                    results.append(result['choices'][0]['text'].encode())
                    if not results[-1]:
                        raise AssertionError('empty completion')
                if require_restore and index > 0:
                    tail = log.read_bytes()[log_start:].decode()
                    if not any('[runner] restore ' in line and 'ok=1' in line
                               for line in tail.splitlines()):
                        raise AssertionError(f'no successful private restore for request {index + 1}')
            return results
        except Exception:
            print(log.read_text()[-8000:])
            raise
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()

original_reference, = run([original])
reference, = run([edited])
# Seed a retained checkpoint strictly before both suffixes. End at a word
# boundary without trailing whitespace so tokenization preserves the prefix.
_, a, b, c = run([prefix.rstrip(), original, original, edited], require_restore=True)
if a != original_reference or a != b or c != reference:
    raise SystemExit(f'byte mismatch: original={a == original_reference}, repeated={a == b}, edit={c == reference}')
print(f'OK: repeated completion {len(a)} bytes; edited continuation {len(c)} bytes matches cold reference')
