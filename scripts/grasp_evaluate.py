"""Sequential, resumable GRASP experiment. No LLM dependencies or API calls."""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
GRASP = Path(os.environ.get('TFM_GRASP_DIR', ROOT.parent / 'mulcia-tfm-grasp')).resolve()
ARTIFACTS = Path(os.environ.get('TFM_ARTIFACTS_DIR', ROOT.parent / 'mulcia-tfm-artifacts')).resolve()
SELECTION = Path(os.environ.get('TFM_GRASP_SELECTION_DIR', ROOT / 'outputs/grasp/selection')).resolve()
DEFAULT = ROOT / 'outputs/grasp/cross-evaluation'
GRASP_INPUTS = ARTIFACTS / 'inputs/03-grasp'
PROTOCOL = json.loads((GRASP_INPUTS / '02-protocol/protocol.json').read_text(encoding='utf-8'))
SELECT_CASES = PROTOCOL['selection']
FINAL_CASES = PROTOCOL['final']
BASE_HASH = 'fb28997aaf3b251e947ac0b57cd00f7a9341d607af686e5282d1447ab579217a'
FIELDS = ['phase', 'family', 'candidate', 'instance', 'solved', 'runtime', 'cost',
          'first_solution_time', 'first_solution_cost', 'time_to_best', 'iterations',
          'routes', 'seed', 'alpha', 'time_limit', 'status', 'error']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def source(candidate):
    if candidate == 'baseline':
        return GRASP / 'src/heuristic.py'
    family, number = candidate.split('-')
    return SELECTION / family / number / 'out.py'


def instance_file(name):
    return GRASP_INPUTS / '01-instances' / f'{name}.vrp'


def candidates(family):
    return ['baseline'] + [f'{family}-{i:02}' for i in range(1, 11)]


def inspect_source(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    allowed = {'math', 'typing', '__future__', 'src.cvrp'}
    forbidden = {'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals',
                 'getattr', 'setattr', 'delattr', 'vars', 'input', 'print', 'breakpoint',
                 'help', 'exit', 'quit', 'read_vrp', 'get_type_hints'}
    imports = []
    module_aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name] = alias.name
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(a.name.startswith('_') or a.name == '*' for a in node.names):
                raise ValueError('private/wildcard imports forbidden')
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError('relative imports forbidden')
            for name in names:
                if name not in allowed:
                    raise ValueError(f'unreviewed import: {name}')
                imports.append(name)
            if isinstance(node, ast.ImportFrom):
                permitted = {'src.cvrp': {'Instance'}, 'typing': {'TYPE_CHECKING', 'List', 'Tuple', 'Optional', 'Sequence', 'Any'}, '__future__': {'annotations'}}
                if node.module in permitted and any(a.name not in permitted[node.module] for a in node.names):
                    raise ValueError(f'forbidden imported symbol: {ast.unparse(node)}')
        if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith('__')):
            raise ValueError(f'forbidden name: {node.id}')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in forbidden):
            raise ValueError(f'forbidden attribute: {node.attr}')
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            module = module_aliases.get(node.value.id)
            if module == 'typing' and node.attr not in {'TYPE_CHECKING', 'List', 'Tuple', 'Optional', 'Sequence', 'Any'}:
                raise ValueError('unreviewed typing attribute')
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError('global/nonlocal state forbidden')
    return sorted(set(imports))


def load_score(candidate):
    path = source(candidate)
    inspect_source(path)
    spec = importlib.util.spec_from_file_location('generated_score', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.heuristic


def clean_env():
    keep = {'SYSTEMROOT', 'WINDIR', 'SYSTEMDRIVE', 'COMSPEC', 'TEMP', 'TMP'}
    env = {k: v for k, v in os.environ.items() if k.upper() in keep}
    env.update(PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    return env


def verify_metrics(instance, metrics):
    from src.cvrp import Solution
    if not metrics['feasible']:
        assert metrics['best_cost'] is None and not metrics['routes']
        return
    for routes, cost in [(metrics['routes'], metrics['best_cost']),
                         (metrics['first_routes'], metrics['first_solution_cost'])]:
        solution = Solution(instance)
        solution.routes = routes
        assert solution.is_feasible(), 'infeasible routes'
        assert math.isclose(solution.get_quality(), cost, rel_tol=1e-12), 'cost mismatch'
    history = metrics['improvements']
    assert history[0]['cost'] == metrics['first_solution_cost']
    assert history[-1]['cost'] == metrics['best_cost']
    assert history[-1]['time'] == metrics['time_to_best']
    assert 0 <= metrics['first_solution_time'] <= metrics['time_to_best'] <= metrics['elapsed']
    assert all(a['cost'] > b['cost'] and a['time'] <= b['time'] for a, b in zip(history, history[1:]))


def execute(candidate, instance, budget, seed):
    from src.operators import GraspPfi, TwoOpt
    from src.utils import Grasp, RandomManager
    first_routes = []

    class CaptureFirst(GraspPfi):
        def construct(self, instance, deadline=None):
            result = super().construct(instance, deadline)
            if not first_routes:
                first_routes.extend(copy.deepcopy(result.routes))
            return result

    RandomManager.set_seed(seed)
    solver = Grasp(CaptureFirst(load_score(candidate)), TwoOpt(), iterations=None, time_limit=budget)
    solution = solver.execute(instance)
    metrics = dict(solver.metrics, first_routes=first_routes)
    verify_metrics(instance, metrics)
    assert metrics['elapsed'] <= budget + 1, 'cooperative deadline exceeded'
    return metrics, solution


def smoke(candidate):
    from src.cvrp import Instance, read_vrp
    score = load_score(candidate)
    samples = []
    # Initial RCL: every feasible customer, including singleton/exact capacity.
    for demands, capacity in [([0, 2], 2), ([0, 1, 2, 3], 3)]:
        inst = Instance('empty-route', [(i, i % 2) for i in range(len(demands))], demands, capacity)
        for customer in inst.customers():
            samples.append((inst, [[]], inst.customers(), 0, customer, 0))
    family = 'E' if candidate == 'baseline' else candidate[0]
    for case in SELECT_CASES[family]:
        inst = read_vrp(str(instance_file(case)))
        for customer in inst.customers():
            if inst.demands[customer] <= inst.capacity:
                samples.append((inst, [[]], inst.customers(), 0, customer, 0))
    for scale in (0.0, 1.0, 1000.0):
        inst = Instance('smoke', [(scale * i, scale * (i * i % 7)) for i in range(9)],
                        [0] + [1, 2] * 4, 12)
        for route in ([], [1], [1, 2, 3]):
            for customer in (4, 5, 6):
                for position in range(len(route) + 1):
                    samples.append((inst, [[7], route[:]], [4, 5, 6, 8], 1, customer, position))
    values = []
    for args in samples:
        before = copy.deepcopy((args[0].__dict__, args[1:]))
        value = score(**dict(zip(('instance', 'solution', 'unassigned', 'route_index', 'customer', 'position'), args)))
        assert isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        assert before == (args[0].__dict__, args[1:]), 'input mutation'
        values.append(value)
    for args, expected in reversed(list(zip(samples, values))):
        before = copy.deepcopy((args[0].__dict__, args[1:]))
        assert score(**dict(zip(('instance', 'solution', 'unassigned', 'route_index', 'customer', 'position'), args))) == expected, 'nondeterministic score'
        assert before == (args[0].__dict__, args[1:]), 'input mutation'
    # Exercise the solver on a tiny synthetic instance, not five timed selection
    # trials. Deadline exhaustion is diagnostic, never a score-contract failure.
    inst = Instance('construction-smoke', [(0, 0), (1, 0), (0, 1)], [0, 1, 1], 2)
    metrics, _ = execute(candidate, inst, 0.05, 1)
    checks = [{'instance': inst.name, 'cost': metrics['best_cost'], 'runtime': metrics['elapsed'],
               'status': 'ok' if metrics['feasible'] else 'deadline_no_solution'}]
    return {'samples': len(samples), 'scores': values, 'checks': checks}


def worker(args):
    print(f'WORKER pid={os.getpid()} candidate={args.candidate} instance={args.instance}', flush=True)
    sys.path.insert(0, str(GRASP))
    if args.mode == 'smoke-worker':
        save(args.result, smoke(args.candidate))
    else:
        from src.cvrp import read_vrp
        instance = read_vrp(str(instance_file(args.instance)), alpha=args.alpha)
        metrics, solution = execute(args.candidate, instance, args.time_limit, args.seed)
        save(args.result, metrics)
        print('METRICS', json.dumps(metrics, allow_nan=False), flush=True)
        args.result.with_suffix('.sol').write_text(str(solution) + '\n', encoding='utf-8')


def child(args, candidate, result, log, mode, instance=None):
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), mode,
               '--candidate', candidate, '--result', str(result), '--seed', str(args.seed),
               '--alpha', str(args.alpha), '--time-limit', str(args.time_limit)]
    if instance:
        command += ['--instance', instance]
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='grasp-sandbox-') as sandbox, log.open('a', encoding='utf-8') as output:
        try:
            run = subprocess.run(command, cwd=sandbox, env=clean_env(), stdout=output,
                                 stderr=subprocess.STDOUT, timeout=20 if mode == 'smoke-worker' else args.external_timeout)
            status = 'ok' if run.returncode == 0 else 'error'
        except subprocess.TimeoutExpired:
            status = 'timeout'
        except OSError:
            traceback.print_exc(file=output)
            status = 'error'
        output.write(f'CHILD_STATUS {status} wall_seconds={time.perf_counter() - start}\n')
    return status, time.perf_counter() - start


def configuration(args):
    families = [args.family] if args.mode == 'evaluate' else list(SELECT_CASES)
    cases = {args.family: SELECT_CASES[args.family]} if args.mode == 'evaluate' else FINAL_CASES
    paths = [source(c) for f in families for c in candidates(f)]
    paths += list((GRASP / 'src').glob('*.py'))
    paths += [instance_file(case) for group in (*SELECT_CASES.values(), *FINAL_CASES.values()) for case in group]
    paths += [GRASP_INPUTS / '02-protocol/protocol.json', Path(__file__).resolve()]
    assert digest(source('baseline')) == BASE_HASH, 'baseline changed'
    return dict(stage=args.mode, family=args.family, seed=args.seed, alpha=args.alpha, time_limit=args.time_limit,
                external_timeout=args.external_timeout, cases=cases,
                selection_cases=SELECT_CASES, final_cases=FINAL_CASES,
                hashes={str(p): digest(p) for p in sorted(set(paths))})


def signature(args):
    return hashlib.sha256(json.dumps(configuration(args), sort_keys=True).encode()).hexdigest()


@contextmanager
def single_run():
    # A named Windows mutex disappears automatically when the process exits.
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    name = hashlib.sha256(str(ROOT).encode()).hexdigest()
    handle = kernel.CreateMutexW(None, False, 'Local\\grasp-' + name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        if ctypes.get_last_error() == 183:
            raise RuntimeError('another GRASP evaluator is active')
        yield
    finally:
        kernel.CloseHandle(handle)


def validate(args):
    records = {}
    for candidate in args.selected:
        record = {'valid': False}
        log = args.output / 'logs' / f'{candidate}__validation.log'
        with log.open('a', encoding='utf-8') as output:
            output.write('VALIDATION_CONFIG ' + signature(args) + '\n')
        try:
            record['imports'] = inspect_source(source(candidate))
            scores = []
            with tempfile.TemporaryDirectory(prefix='grasp-validation-') as tmp:
                for repeat in range(2):
                    result = Path(tmp) / f'{repeat}.json'
                    status, runtime = child(args, candidate, result, log, 'smoke-worker')
                    record[f'runtime_{repeat}'] = runtime
                    if status == 'timeout':
                        # Unfinished checks do not prove a contract violation.
                        record.update(valid=None, status='timeout', error=short_error('validation timeout', log))
                        break
                    if status != 'ok':
                        raise ValueError('validation worker error')
                    payload = json.loads(result.read_text(encoding='utf-8'))
                    scores.append(payload['scores'])
                    with log.open('a', encoding='utf-8') as output:
                        output.write('SMOKE_CHECKS ' + json.dumps(payload.get('checks', [])) + '\n')
            if record.get('status') != 'timeout':
                assert scores[0] == scores[1], 'cross-process nondeterminism'
                record.update(valid=True, status='ok')
        except Exception as error:
            with log.open('a', encoding='utf-8') as output:
                traceback.print_exc(file=output)
            record.update(status='invalid', error=short_error(type(error).__name__, log))
        records[candidate] = record
        with log.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        print('VALIDATION', candidate, record, flush=True)
    return records


def short_error(reason, log):
    return ' '.join(str(reason).split())[:120] + f'; see logs/{log.name}'


class StoredRow(dict):
    """Keep original CSV field strings while exposing typed values to ranking."""


def write_csv(path, rows, fields):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            values = row.raw if isinstance(row, StoredRow) and row == row.parsed else row
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in values.items() if k in fields})
    temp.replace(path)


def complete(rows, family):
    expected = {(c, i) for c in candidates(family) for i in SELECT_CASES[family]}
    own = [r for r in rows if r['phase'] == 'dev' and r['family'] == family]
    keys = [(r['candidate'], r['instance']) for r in own]
    return (len(keys) == len(expected) and set(keys) == expected
            and all((r.get('seed'), r.get('alpha'), r.get('time_limit')) == (1, 0.3, 300) for r in own))


def winner(rows, family):
    if not complete(rows, family):
        count = len(candidates(family)) * len(SELECT_CASES[family])
        raise ValueError(f'{family}: requires all {count} distinct selection trials with seed=1, alpha=0.3, budget=300')
    eligible = [r for r in ranking(rows, family) if r['candidate'] != 'baseline' and r['coverage'] > 0]
    if not eligible:
        raise ValueError(f'{family}: no solved generated candidate')
    return eligible[0]['candidate']


def ranking(rows, family, first=False):
    data = [r for r in rows if r['phase'] == 'dev' and r['family'] == family]
    cost_key = 'first_solution_cost' if first else 'cost'
    time_key = 'first_solution_time' if first else 'time_to_best'
    def measured(row):
        return row['solved'] is True and row.get(cost_key) is not None and math.isfinite(row[cost_key])

    best = {case: min((r[cost_key] for r in data if r['instance'] == case and measured(r)), default=None)
            for case in SELECT_CASES[family]}
    result = []
    for candidate in candidates(family):
        own = [r for r in data if r['candidate'] == candidate]
        solved = [r for r in own if measured(r)]
        gaps = [(r[cost_key] - best[r['instance']]) / best[r['instance']] for r in solved]
        times = [r[time_key] for r in solved if r.get(time_key) is not None and math.isfinite(r[time_key])]
        result.append(dict(candidate=candidate, coverage=len(solved) / len(SELECT_CASES[family]), completed=len(own),
                           mean_relative_gap=statistics.mean(gaps) if gaps else None,
                            mean_time_to_best=statistics.mean(times) if times else None,
                           status='complete' if complete(rows, family) else 'PARTIAL_PENDING'))
    result.sort(key=lambda r: (-r['coverage'], r['mean_relative_gap'] if r['mean_relative_gap'] is not None else math.inf,
                               r['mean_time_to_best'] if r['mean_time_to_best'] is not None else math.inf, r['candidate']))
    if first:
        for r in result:
            r['mean_first_solution_time'] = r.pop('mean_time_to_best')
    return result


def trial(args, validation, phase, family, candidate, instance):
    key = f'{phase}-{family}-{instance}-{candidate}'
    log = args.output / 'logs' / f'{candidate}__{instance}.log'
    log.write_text('CONFIG ' + signature(args) + '\n', encoding='utf-8')
    row = dict(phase=phase, family=family, candidate=candidate, instance=instance,
               solved=False, runtime=None, cost=None, first_solution_time=None, first_solution_cost=None,
               time_to_best=None, iterations=None, routes=None, seed=args.seed, alpha=args.alpha,
               time_limit=args.time_limit, status='invalid', error='')
    if validation[candidate]['valid'] is False:
        with log.open('a', encoding='utf-8') as output:
            output.write(json.dumps(validation[candidate], ensure_ascii=False) + '\n')
        row['error'] = short_error('contract validation failed', args.output / 'logs' / f'{candidate}__validation.log')
    else:
        print('START', key, flush=True)
        with tempfile.TemporaryDirectory(prefix='grasp-result-') as tmp:
            result = Path(tmp) / 'result.json'
            status, runtime = child(args, candidate, result, log, 'worker', instance)
            row.update(status=status)
            if status == 'ok':
                try:
                    m = json.loads(result.read_text(encoding='utf-8'))
                    (args.output / 'plans' / f'{candidate}__{instance}.sol').write_bytes(result.with_suffix('.sol').read_bytes())
                    row.update(solved=m['feasible'], runtime=m['elapsed'], cost=m['best_cost'],
                               first_solution_time=m['first_solution_time'], first_solution_cost=m['first_solution_cost'],
                               time_to_best=m['time_to_best'], iterations=m['iterations'], routes=m['routes'],
                               status='ok' if m['feasible'] else 'unsolved')
                except (OSError, ValueError, KeyError, TypeError):
                    with log.open('a', encoding='utf-8') as output:
                        traceback.print_exc(file=output)
                    row.update(status='error', error=short_error('invalid worker result', log))
            else:
                row['error'] = short_error(status, log)
    print('FINISH', key, row['status'], row['cost'], flush=True)
    return row


def read_rows(folder):
    path = folder / 'metrics.csv'
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        rows = [StoredRow(row) for row in csv.DictReader(handle)]
    for row in rows:
        row.raw = dict(row)
        row['solved'] = {'True': True, 'False': False, '': None}[row['solved']]
        row['routes'] = json.loads(row['routes']) if row['routes'] else None
        for key in ('runtime', 'cost', 'first_solution_time', 'first_solution_cost', 'time_to_best', 'alpha', 'time_limit'):
            row[key] = float(row[key]) if row[key] else None
        for key in ('seed', 'iterations'):
            row[key] = int(row[key]) if row[key] else None
        row.parsed = copy.deepcopy(dict(row))
    return rows


def verify_resume(args, rows):
    expected = signature(args)
    seen = set()
    schedule = FINAL_CASES if args.mode == 'cross' else {args.family: SELECT_CASES[args.family]}
    for row in rows:
        key = (row['family'], row['candidate'], row['instance'])
        if (key in seen or row['phase'] != ('cross' if args.mode == 'cross' else 'dev')
                or row['instance'] not in schedule.get(row['family'], [])
                or row['candidate'] not in args.selected
                or (row['seed'], row['alpha'], row['time_limit']) != (args.seed, args.alpha, args.time_limit)):
            raise ValueError('stale or duplicate CSV configuration')
        seen.add(key)
        log = args.output / 'logs' / f'{row["candidate"]}__{row["instance"]}.log'
        with log.open(encoding='utf-8') as handle:
            original = handle.readline().strip()
            # Explicit, audited evaluator-only migration; retain historical CONFIG.
            migrated = any(line.startswith('RESUME_CONFIG ' + expected + ' evaluator-only ') for line in handle)
            if original != 'CONFIG ' + expected and not migrated:
                raise ValueError('stale log configuration: code/input/budget changed')


def publish(args, rows):
    write_csv(args.output / 'metrics.csv', rows, FIELDS)
    if args.mode == 'evaluate':
        ranks = ranking(rows, args.family)
        write_csv(args.output / 'ranking.csv', ranks, list(ranks[0]))


def cross_winners(args):
    winners = {}
    for family in SELECT_CASES:
        folder = SELECTION / family
        dev_args = argparse.Namespace(**vars(args))
        dev_args.mode, dev_args.family = 'evaluate', family
        dev_args.output, dev_args.selected = folder, candidates(family)
        rows = read_rows(folder)
        verify_resume(dev_args, rows)
        chosen = winner(rows, family)
        # Require the published ranking to match the complete raw records exactly.
        with (folder / 'ranking.csv').open(encoding='utf-8', newline='') as handle:
            published = list(csv.DictReader(handle))
        expected = ranking(rows, family)
        if len(published) != len(expected) or any(
                set(actual) != set(want) or any(
                    (actual[k] != ('' if v is None else str(v)))
                    if not isinstance(v, (int, float)) else float(actual[k]) != v
                    for k, v in want.items()) for actual, want in zip(published, expected)):
            raise ValueError(f'{family}: incomplete or stale ranking.csv')
        winners[family] = chosen
    return winners


def parent(args):
    with single_run():
        evaluate(args)


def evaluate(args):
    winners = cross_winners(args) if args.mode == 'cross' else None
    args.selected = ['baseline', *winners.values()] if winners else candidates(args.family)
    config = configuration(args)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'plans'):
        (args.output / name).mkdir(exist_ok=True)
    rows = read_rows(args.output)
    verify_resume(args, rows)
    publish(args, rows)
    validation = validate(args)
    if all(validation[c]['valid'] is False for c in args.selected if c != 'baseline'):
        raise ValueError('all generated candidates invalid; stopping before trials')
    schedule = FINAL_CASES if winners else {args.family: SELECT_CASES[args.family]}
    finished = {(r['family'], r['candidate'], r['instance']) for r in rows}
    ranks = {f: digest(SELECTION / f / 'ranking.csv') for f in winners or {}}
    for family, cases in schedule.items():
        for case in cases:
            for candidate in args.selected:
                if configuration(args) != config or any(digest(SELECTION / f / 'ranking.csv') != h for f, h in ranks.items()):
                    raise ValueError('configuration changed during experiment')
                if (family, candidate, case) in finished:
                    continue
                row = trial(args, validation, 'cross' if winners else 'dev', family, candidate, case)
                if configuration(args) != config:
                    raise ValueError('configuration changed during trial')
                rows.append(row)
                publish(args, rows)
    if winners:
        (args.output / 'README.md').write_text(
            f'Ganadores: E: {winners["E"]}; P: {winners["P"]}.\n'
            'Comparados con baseline en cuatro instancias finales (12 ensayos).\n'
            'Seed 1, alpha 0.3, 300 segundos por ensayo. Resultados en metrics.csv.\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['evaluate', 'cross', 'worker', 'smoke-worker'])
    parser.add_argument('--family', choices=['E', 'P'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--time-limit', type=float, default=300)
    parser.add_argument('--external-timeout', type=float, default=330)
    parser.add_argument('--candidate')
    parser.add_argument('--instance')
    parser.add_argument('--result', type=Path)
    args = parser.parse_args()
    if not (math.isfinite(args.time_limit) and args.time_limit > 0 and 0 <= args.alpha <= 1
            and math.isfinite(args.external_timeout) and args.external_timeout > args.time_limit):
        parser.error('invalid budget/alpha/external timeout')
    if args.mode == 'evaluate' and not args.family:
        parser.error('evaluate requires --family E or P')
    if args.mode == 'cross' and args.family:
        parser.error('cross evaluates both families; omit --family')
    if args.mode in ('evaluate', 'cross') and (args.seed, args.alpha, args.time_limit) != (1, 0.3, 300):
        parser.error('protocol requires seed=1, alpha=0.3, time-limit=300')
    args.output = (args.output or (SELECTION / args.family if args.mode == 'evaluate' else DEFAULT)).resolve()
    if args.mode.endswith('worker'):
        worker(args)
    else:
        parent(args)


if __name__ == '__main__':
    main()
