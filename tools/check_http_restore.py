#!/usr/bin/env python3
"""Compare restored and concurrent greedy responses with isolated cold servers."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import hashlib
import re
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
parser.add_argument('--dump', default=None,
                    help='write each completion to DIR, named by the prompt hash, '
                         'so a spec-off run and a spec-on run can be diffed byte for byte')
args = parser.parse_args()
prefix = 'Here is one fact about the red fox. ' * 8
original = prefix + 'Where does it live?'
edited = prefix + 'What does it eat?'

run_count = 0

def run(prompts, require_restore=False, concurrent=False):
    global run_count
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory, 'serve.log')
        with log.open('w') as output:
            proc = subprocess.Popen([args.gufo, 'serve', 'llm', '--model', args.model,
                                     '--port', str(args.port), '--sessions', '2' if concurrent else '1'],
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
            barrier = Barrier(len(prompts)) if concurrent else None
            def ask(prompt):
                if barrier:
                    barrier.wait(timeout=10)
                body = {'prompt': prompt, 'model': args.model, 'temperature': 0, 'max_tokens': 12}
                req = Request(f'http://127.0.0.1:{args.port}/v1/completions',
                              data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
                with urlopen(req, timeout=180) as resp:
                    result = json.load(resp)
                    text = result['choices'][0]['text'].encode()
                    if not text:
                        raise AssertionError('empty completion')
                    if args.dump:
                        directory = Path(args.dump)
                        directory.mkdir(parents=True, exist_ok=True)
                        name = hashlib.sha256(prompt.encode()).hexdigest()[:16] + '.txt'
                        (directory / name).write_bytes(text)
                    return text
            if concurrent:
                with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
                    results = list(pool.map(ask, prompts))
                trace = log.read_text()
                banks = re.findall(r'\[runner\] bank (\d+) engine (\w+) slot (\w+)', trace)
                if (len({b[0] for b in banks}) != 2 or len({b[1] for b in banks}) != 1 or
                        len({b[2] for b in banks}) != 2):
                    raise AssertionError('two resident banks did not share exactly one engine')
                captures = set(re.findall(r'\[runner\] capture slot (\w+)', trace))
                if captures != {b[2] for b in banks}:
                    raise AssertionError('both slots did not participate')
                # Close a real SSE response after the first generated piece,
                # and verify that the scheduler actually noticed the disconnect.
                start = log.stat().st_size
                body = dict(prompt='Count from 1 to 100:', model=args.model,
                            temperature=0, max_tokens=128, stream=True)
                req = Request(f'http://127.0.0.1:{args.port}/v1/completions',
                              data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
                with urlopen(req, timeout=60) as stream:
                    for line in stream:
                        if line.startswith(b'data: {') and json.loads(line[6:])['choices'][0]['text']:
                            break
                    else:
                        raise AssertionError('no SSE text chunk')
                for _ in range(100):
                    if b'finish=cancelled' in log.read_bytes()[start:]:
                        break
                    time.sleep(.1)
                else:
                    raise AssertionError('stream closed but server did not record cancellation')
                print('OK: SSE yielded text; disconnect confirmed as cancelled by server')
                return results
            for index, prompt in enumerate(prompts):
                log_start = log.stat().st_size
                results.append(ask(prompt))
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
            if args.dump:
                # Keep the server log beside the dumps: the runner's spec trace is
                # the evidence that speculation ran for this run.
                run_count += 1
                (Path(args.dump) / f'serve-{run_count}.log').write_text(log.read_text())

original_reference, = run([original])
reference, = run([edited])
# Seed a retained checkpoint strictly before both suffixes. End at a word
# boundary without trailing whitespace so tokenization preserves the prefix.
_, a, b, c = run([prefix.rstrip(), original, original, edited], require_restore=True)
if a != original_reference or a != b or c != reference:
    raise SystemExit(f'byte mismatch: original={a == original_reference}, repeated={a == b}, edit={c == reference}')
print(f'OK: repeated completion {len(a)} bytes; edited continuation {len(c)} bytes matches cold reference')
pair = run([original, edited], concurrent=True)
if pair != [original_reference, reference] or original_reference == reference:
    raise SystemExit('concurrent completions did not match distinguishable cold references')
print('OK: both slots used one engine; both concurrent outputs matched their cold references')
