"""Synthetic contracts only: no solver, real instances, workers or APIs."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import grasp_evaluate as ev


def arguments(folder, family='E', mode='evaluate'):
    return argparse.Namespace(output=folder, family=family, mode=mode,
                              seed=1, alpha=0.3, time_limit=300, external_timeout=330)


def rows(family):
    return [dict(phase='dev', family=family, candidate=c, instance=i, solved=True,
                  cost=10 if c == 'baseline' else 20, time_to_best=1, runtime=300,
                  iterations=10, routes=[[1, 2]], status='ok', error='',
                  first_solution_cost=30, first_solution_time=0.1,
                  seed=1, alpha=0.3, time_limit=300)
            for i in ev.SELECT_CASES[family] for c in ev.candidates(family)]


class EvaluationContracts(unittest.TestCase):
    def test_smoke_deadline_advisory_and_contract_errors(self):
        class Instance:
            def __init__(self, name, coordinates, demands, capacity):
                self.name, self.demands, self.capacity = name, demands, capacity
            def customers(self):
                return list(range(1, len(self.demands)))
        module = SimpleNamespace(Instance=Instance, read_vrp=lambda p: Instance('fixture', [], [0, 1], 2))
        def score(**kw):
            return 1.0
        with patch.dict(ev.sys.modules, {'src.cvrp': module}), \
                patch.object(ev, 'load_score', return_value=score), \
                patch.object(ev, 'execute', return_value=({'feasible': False, 'best_cost': None, 'elapsed': .05}, None)):
            result = ev.smoke('P-03')
            self.assertEqual(result['checks'][0]['status'], 'deadline_no_solution')
            self.assertGreater(len(result['scores']), 50)
            for bad in [lambda **kw: float('nan'), lambda **kw: kw['solution'].append([]) or 1]:
                with patch.object(ev, 'load_score', return_value=bad), self.assertRaises(AssertionError):
                    ev.smoke('P-03')
            with patch.object(ev, 'execute', side_effect=RuntimeError('real bug')), self.assertRaises(RuntimeError):
                ev.smoke('P-03')

    def test_validation_full_log_short_error_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'logs').mkdir()
            args = arguments(folder)
            args.selected = ['baseline']
            detail = 'Traceback\n' + 'huge exception\n' * 1000
            def child(args, candidate, result, log, mode):
                with log.open('a') as output:
                    output.write(detail)
                return 'error', .1
            with patch.object(ev, 'inspect_source', return_value=[]), patch.object(ev, 'child', side_effect=child):
                records = ev.validate(args)
            self.assertFalse(records['baseline']['valid'])
            self.assertIn(detail, (folder / 'logs/baseline__validation.log').read_text())
            row = ev.trial(args, records, 'dev', 'E', 'baseline', 'E-n23-k3')
            ev.publish(args, [row])
            text = (folder / 'metrics.csv').read_text()
            self.assertEqual(len(text.splitlines()), 2)
            self.assertLess(len(row['error']), 200)
            self.assertNotIn('Traceback', text)
            with patch.object(ev, 'inspect_source', return_value=[]), patch.object(ev, 'child', return_value=('timeout', 20)):
                records = ev.validate(args)
            self.assertIsNone(records['baseline']['valid'])
            with patch.object(ev, 'child', return_value=('timeout', 330)) as child_mock:
                timed = ev.trial(args, records, 'dev', 'E', 'baseline', 'E-n23-k3')
            child_mock.assert_called_once()
            self.assertEqual(timed['status'], 'timeout')

    def test_child_failure_and_timeout_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            args = arguments(folder)
            with patch.object(ev.subprocess, 'run', side_effect=[
                    ev.subprocess.TimeoutExpired('fixture', 330), SimpleNamespace(returncode=7),
                    OSError('spawn\nfailed'), SimpleNamespace(returncode=0)]):
                statuses = [ev.child(args, 'baseline', folder / 'r.json', folder / 'trial.log', 'worker')[0]
                            for _ in range(4)]
            self.assertEqual(statuses, ['timeout', 'error', 'error', 'ok'])
            self.assertIn('Traceback', (folder / 'trial.log').read_text())

    def test_unknown_fields_and_exact_resume_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = rows('E')[:2]
            data[0].update(runtime='3.000e2', time_limit='300', cost='10.000000000000000', routes='[[1,2]]')
            data[1].update(solved='', cost=None, time_to_best=None, iterations=None, routes=None)
            ev.write_csv(folder / 'metrics.csv', data, ev.FIELDS)
            original = (folder / 'metrics.csv').read_bytes()
            loaded = ev.read_rows(folder)
            self.assertIsNone(loaded[1]['solved'])
            self.assertIsNone(loaded[1]['iterations'])
            ev.publish(arguments(folder), loaded)
            self.assertEqual((folder / 'metrics.csv').read_bytes(), original)
            rank = next(r for r in ev.ranking(loaded, 'E') if r['candidate'] == 'E-01')
            self.assertEqual(rank['coverage'], 0)
            self.assertIsNone(rank['mean_time_to_best'])
            self.assertEqual(rank['status'], 'PARTIAL_PENDING')

    def test_resume_marker_only_exact_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'logs').mkdir()
            args = arguments(folder)
            args.selected = ev.candidates('E')
            data = rows('E')[:1]
            log = folder / 'logs/baseline__E-n23-k3.log'
            log.write_text('CONFIG original\nRESUME_CONFIG new evaluator-only proof=checked\n')
            with patch.object(ev, 'signature', return_value='new'):
                ev.verify_resume(args, data)
            with patch.object(ev, 'signature', return_value='changed'), self.assertRaises(ValueError):
                ev.verify_resume(args, data)

    def test_ast_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.py'
            for code in ['import os', 'from src.cvrp import read_vrp', 'open("x")',
                         'eval("1")', 'x.__class__', 'from typing import get_type_hints']:
                path.write_text(code)
                with self.subTest(code=code), self.assertRaises(ValueError):
                    ev.inspect_source(path)
        with patch.dict(ev.os.environ, {'API_KEY': 'secret', 'PYTHONPATH': 'bad'}):
            self.assertNotIn('API_KEY', ev.clean_env())
            self.assertNotIn('PYTHONPATH', ev.clean_env())

    def test_stale_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'logs').mkdir()
            args = arguments(folder)
            args.selected = ev.candidates('E')
            data = rows('E')[:1]
            log = folder / 'logs' / f'baseline__{data[0]["instance"]}.log'
            log.write_text('CONFIG a\n')
            with patch.object(ev, 'signature', return_value='a'):
                ev.verify_resume(args, data)
            with patch.object(ev, 'signature', return_value='b'), self.assertRaisesRegex(ValueError, 'stale log'):
                ev.verify_resume(args, data)

    def test_stage_configs_and_all_invalid_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.object(ev, 'digest', return_value=ev.BASE_HASH):
                selection = ev.configuration(arguments(folder))
                final = ev.configuration(arguments(folder, None, 'cross'))
            self.assertEqual(selection['cases'], {'E': ev.SELECT_CASES['E']})
            self.assertEqual(final['cases'], ev.FINAL_CASES)
            self.assertNotEqual(selection, final)
            invalid = {c: {'valid': c == 'baseline'} for c in ev.candidates('E')}
            with patch.object(ev, 'configuration', return_value=selection), \
                    patch.object(ev, 'validate', return_value=invalid), \
                    patch.object(ev, 'trial', side_effect=AssertionError('must stop')):
                with self.assertRaisesRegex(ValueError, 'all generated candidates invalid'):
                    ev.parent(arguments(folder))

    def test_ranking_and_complete_guard(self):
        data = rows('E')
        self.assertEqual(ev.ranking(data, 'E')[0]['candidate'], 'baseline')
        self.assertEqual(ev.winner(data, 'E'), 'E-01')
        self.assertEqual(ev.ranking(data, 'E')[1]['mean_relative_gap'], 1)
        for bad in [data[:-1], data[:-1] + [data[0]], data + [data[0]]]:
            with self.assertRaises(ValueError):
                ev.winner(bad, 'E')
        data[1]['solved'] = False
        self.assertEqual(ev.winner(data, 'E'), 'E-02')
        self.assertEqual(next(r for r in ev.ranking(data, 'E') if r['candidate'] == 'E-01')['coverage'], 4 / 5)
        for key, value in [('seed', 2), ('alpha', 0.4), ('time_limit', 30)]:
            bad = rows('E')
            bad[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ev.winner(bad, 'E')

    def test_first_metric_and_protocol(self):
        self.assertEqual(ev.SELECT_CASES, {
            'E': ['E-n23-k3', 'E-n33-k4', 'E-n51-k5', 'E-n76-k14', 'E-n101-k8'],
            'P': ['P-n16-k8', 'P-n23-k8', 'P-n50-k10', 'P-n76-k4', 'P-n101-k4']})
        self.assertEqual(ev.FINAL_CASES, {'E': ['E-n30-k3', 'E-n22-k4'], 'P': ['P-n19-k2', 'P-n40-k5']})
        data = rows('E')
        for r in data:
            if r['candidate'] == 'E-03':
                r.update(first_solution_cost=5, first_solution_time=0.01)
        self.assertEqual(ev.ranking(data, 'E', first=True)[0]['candidate'], 'E-03')
        self.assertEqual(ev.ranking(data, 'E', first=True)[0]['mean_first_solution_time'], 0.01)
        self.assertEqual(ev.winner(data, 'E'), 'E-01')

    def test_resume_and_interrupted_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'logs').mkdir()
            (folder / 'plans').mkdir()
            log = folder / 'logs/baseline__E-n23-k3.log'
            log.write_text('interrupted evidence')
            def child(args, candidate, result, log, mode, instance):
                self.assertFalse(result.is_relative_to(folder))
                return 'timeout', 330
            args = arguments(folder)
            with patch.object(ev, 'child', side_effect=child):
                row = ev.trial(args, {'baseline': {'valid': True}}, 'dev', 'E', 'baseline', 'E-n23-k3')
            ev.publish(args, [row])
            self.assertEqual(ev.read_rows(folder), [row])
            self.assertNotIn('interrupted evidence', log.read_text())
            self.assertFalse((folder / 'runs').exists())

    def test_separate_scheduling_and_cross_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = {c: {'valid': True} for f in ev.SELECT_CASES for c in ev.candidates(f)}
            calls = []
            def trial(args, validation, phase, family, candidate, instance):
                calls.append((phase, family, candidate, instance))
                row = dict(rows(family)[0], instance=instance, phase=phase, candidate=candidate,
                           cost=10 if candidate == 'baseline' else 20)
                (args.output / 'logs' / f'{candidate}__{instance}.log').write_text('CONFIG ' + ev.signature(args) + '\n')
                return row
            with patch.object(ev, 'ROOT', root), patch.object(ev, 'configuration',
                    side_effect=lambda a: {'stage': a.mode, 'family': a.family, 'seed': a.seed}), \
                    patch.object(ev, 'validate', return_value=valid), patch.object(ev, 'trial', side_effect=trial):
                cross = arguments(root / 'experiments/grasp-evaluations', None, 'cross')
                for family in 'EP':
                    ev.parent(arguments(root / f'experiments/grasp-{family}', family))
                    self.assertFalse(cross.output.exists())
                    if family == 'E':
                        with self.assertRaises((ValueError, FileNotFoundError)):
                            ev.parent(cross)
                self.assertEqual(len(calls), 110)
                ev.parent(arguments(root / 'experiments/grasp-E', 'E'))
                self.assertEqual(len(calls), 110, 'completed CSV rows must not run again')
                self.assertTrue(all(c[0] == 'dev' for c in calls))
                for family in 'EP':
                    self.assertEqual(sum(c[1] == family for c in calls), 55)
                cross.seed = 2
                with self.assertRaisesRegex(ValueError, 'configuration'):
                    ev.parent(cross)
                cross.seed = 1
                ev.parent(cross)
                self.assertEqual(len(calls), 122)
                self.assertEqual(len(set(calls[110:])), 12)
                self.assertEqual({c[2] for c in calls[110:]}, {'baseline', 'E-01', 'P-01'})
                self.assertFalse((cross.output / 'ranking.csv').exists())
                self.assertIn('E-01', (cross.output / 'README.md').read_text())
                self.assertFalse((cross.output / 'config.json').exists())
                ranking = root / 'experiments/grasp-P/ranking.csv'
                ranking.write_text('candidate,status\nP-01,complete\n')
                with self.assertRaisesRegex(ValueError, 'ranking'):
                    ev.parent(cross)


if __name__ == '__main__':
    unittest.main()
