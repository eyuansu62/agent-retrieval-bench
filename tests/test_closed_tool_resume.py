import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_retrieval_bench import closed_tool_eval as evaluator
from agent_retrieval_bench.indexed_tools import IndexedTools
from agent_retrieval_bench.trajectory import write_jsonl


class ClosedToolResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = [dict(
            id=f's{i}', repo='example/repo', base_commit='base',
            task_type='trace2code', query={'pr_title': 'anchorapple'},
            gold={'root_cause_files': ['subject.py']},
        ) for i in (1, 2)]
        self.sample_path = self.root/'samples.jsonl'
        self.write(self.sample_path, self.samples)
        self.chunks = self.root/'chunks.jsonl'
        self.write(self.chunks, [dict(
            path='subject.py', kind='file', text='def anchorapple(): return 1\n',
        )])
        self.write(self.root/'corpus_manifest.jsonl', [dict(
            repo='example/repo', base_commit='base', status='ok',
            chunks_path=str(self.chunks),
        )])
        self.calls = []

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')

    def run_arm(self, arm, *, failure_ids=(), **options):
        real = evaluator.run_closed_tool_sample
        keys = ('sample', 'gold_files', 'query_text', 'files', 'max_tool_calls',
                'max_read_tokens', 'max_read_tokens_per_file', 'final_k', 'grep_top_k')

        def policy(**kwargs):
            self.calls.append(kwargs['sample']['id'])
            detail = real(**{key: kwargs[key] for key in keys},
                          max_grep_calls=kwargs.get('max_grep_calls', 8))
            detail['closed_tool']['model_usage'] = {'input_tokens': 7, 'output_tokens': 1}
            if kwargs['sample']['id'] in failure_ids:
                error = RuntimeError('offline policy failure')
                error.partial_detail = detail
                raise error
            return detail

        module = types.ModuleType('openai')
        module.OpenAI = lambda: object()
        functions = {'grep': evaluator.evaluate_closed_tool_grep,
                     'openai': evaluator.evaluate_closed_tool_openai,
                     'codex': evaluator.evaluate_closed_tool_codex}
        target = {'grep': 'run_closed_tool_sample', 'openai': 'run_closed_tool_llm_sample',
                  'codex': 'run_closed_tool_codex_sample'}[arm]
        kwargs = dict(sample_paths=[self.sample_path], corpus_dir=self.root,
                      out_path=self.root/'summary.json', details_path=self.root/'details.jsonl',
                      markdown_out_path=self.root/'report.md')
        if arm == 'codex':
            kwargs['codex_bin'] = Path(sys.executable)
        kwargs.update(options)
        with patch.dict(sys.modules, {'openai': module}), patch.dict(
            os.environ, {'OPENAI_API_KEY': 'offline-test', 'CODEX_HOME': str(self.root/'codex-home')},
        ), patch.object(evaluator, target, side_effect=policy):
            return functions[arm](**kwargs)

    def saved(self):
        return evaluator.load_existing_details(self.root/'details.jsonl')

    def test_subset_and_empty_resume_preserve_cumulative_records(self):
        for arm in ('grep', 'openai', 'codex'):
            for selected in (['s1'], []):
                with self.subTest(arm=arm, selected=selected):
                    self.calls.clear()
                    self.run_arm(arm, force=True)
                    keep = self.root/'keep.jsonl'
                    self.write(keep, [{'sample_id': sid} for sid in selected])
                    summary = self.run_arm(arm, keep_list=keep)
                    self.assertEqual(summary['evaluated'], len(selected))
                    self.assertEqual(set(self.saved()), {'s1', 's2'})
                    self.run_arm(arm)
                    self.assertEqual(self.calls, ['s1', 's2'])

    def test_partial_reorder_narrow_input_and_explicit_reset(self):
        for arm in ('grep', 'openai', 'codex'):
            with self.subTest(arm=arm):
                self.calls.clear()
                self.run_arm(arm, force=True, limit_samples=1)
                self.run_arm(arm)
                self.write(self.sample_path, self.samples[::-1])
                self.run_arm(arm)
                self.write(self.sample_path, self.samples[:1])
                self.run_arm(arm)
                self.assertEqual(set(self.saved()), {'s1', 's2'})
                self.assertEqual(self.calls, ['s1', 's2'])
                self.run_arm(arm, force=True)
                self.assertEqual(set(self.saved()), {'s1'})
                self.write(self.sample_path, self.samples)

    def test_narrower_input_file_list_preserves_unselected_samples(self):
        paths = [self.root/'first.jsonl', self.root/'second.jsonl']
        for path, sample in zip(paths, self.samples):
            self.write(path, [sample])
        for arm in ('grep', 'openai', 'codex'):
            with self.subTest(arm=arm):
                self.run_arm(arm, force=True, sample_paths=paths)
                self.calls.clear()
                summary = self.run_arm(arm, sample_paths=paths[1:])
                self.assertEqual(summary['evaluated'], 1)
                self.assertEqual(set(self.saved()), {'s1', 's2'})
                self.run_arm(arm, sample_paths=paths[::-1])
                self.assertEqual(self.calls, [])

    def test_empty_selection_preserves_failed_attempt_history(self):
        keep = self.root/'empty.jsonl'
        self.write(keep, [])
        for arm in ('openai', 'codex'):
            with self.subTest(arm=arm):
                self.run_arm(arm, force=True, failure_ids=['s2'])
                before = self.saved()['s2']
                summary = self.run_arm(arm, keep_list=keep)
                self.assertEqual(summary['evaluated'], 0)
                self.assertEqual(self.saved()['s2'], before)
                self.run_arm(arm)
                self.assertEqual(self.saved()['s2']['closed_tool']['prior_attempts'][0]['model_usage'],
                                 before['closed_tool']['model_usage'])

    def test_interrupted_compaction_keeps_previous_checkpoint(self):
        path = self.root/'details.jsonl'
        self.write(path, [{'sample_id': 's1'}, {'sample_id': 's2'}])
        before = path.read_bytes()

        def interrupted_rows():
            yield {'sample_id': 's1'}
            raise RuntimeError('interrupted compaction')

        with self.assertRaisesRegex(RuntimeError, 'interrupted compaction'):
            write_jsonl(path, interrupted_rows())
        self.assertEqual(path.read_bytes(), before)

    def test_failed_atomic_replace_keeps_checkpoint_and_removes_temporary_file(self):
        path = self.root/'details.jsonl'
        self.write(path, [{'sample_id': 's1'}])
        before = path.read_bytes()
        with patch('agent_retrieval_bench.trajectory.os.replace', side_effect=OSError('injected replace failure')):
            with self.assertRaisesRegex(OSError, 'replace failure'):
                write_jsonl(path, [{'sample_id': 's2'}])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('.details.jsonl.*.tmp')), [])

    def test_complete_final_record_without_newline_is_preserved(self):
        path = self.root/'details.jsonl'
        path.write_text('{"sample_id":"s1"}', encoding='utf-8')
        existing, handle = evaluator.prepare_details_resume(path, force=False)
        with handle:
            evaluator.append_detail_jsonl(handle, {'sample_id': 's2'})
        self.assertEqual(set(existing), {'s1'})
        self.assertEqual(set(evaluator.load_existing_details(path)), {'s1', 's2'})

    def test_partial_utf8_tail_recovers_but_terminated_corruption_rejects(self):
        path = self.root/'details.jsonl'
        self.write(path, [{'sample_id': 's1'}])
        with path.open('ab') as handle:
            handle.write(b'{"sample_id":"\xe4\xb8')
        with self.assertWarnsRegex(RuntimeWarning, 'incomplete trailing'):
            existing, handle = evaluator.prepare_details_resume(path, force=False)
        handle.close()
        self.assertEqual(set(existing), {'s1'})
        self.assertEqual(len(path.read_bytes().splitlines()), 1)
        path.write_bytes(b'{"sample_id":"\xe4\xb8\n')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Invalid details record'):
            evaluator.prepare_details_resume(path, force=False)
        self.assertEqual(path.read_bytes(), before)

    def test_partial_tail_recovers_but_interior_corruption_is_rejected(self):
        path = self.root/'details.jsonl'
        self.write(path, [{'sample_id': 's1'}])
        with path.open('ab') as handle:
            handle.write(b'{"sample_id":"unfinished')
        with self.assertWarnsRegex(RuntimeWarning, 'incomplete trailing'):
            existing, handle = evaluator.prepare_details_resume(path, force=False)
        with handle:
            evaluator.append_detail_jsonl(handle, {'sample_id': 's2'})
        self.assertEqual(set(existing), {'s1'})
        self.assertEqual(set(evaluator.load_existing_details(path)), {'s1', 's2'})
        path.write_text('{broken}\n{"sample_id":"s2"}\n', encoding='utf-8')
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            evaluator.prepare_details_resume(path, force=False)
        self.assertEqual(path.read_bytes(), before)

    def test_default_prefix_report_deduplicates_retry_journal(self):
        self.run_arm('grep')
        rows = list(self.saved().values())
        failed = dict(rows[1], closed_tool={**rows[1]['closed_tool'],
                                         'failure': {'error': 'retained attempt'}})
        self.write(self.root/'journal.jsonl', [rows[0], failed, rows[1]])
        report = evaluator.report_closed_tool_budget_curve(
            details_path=self.root/'journal.jsonl', out_path=self.root/'curve.json',
            markdown_out_path=self.root/'curve.md', budgets=[16],
        )
        self.assertEqual(report['rows'][0]['evaluated'], 2)
        self.assertEqual(report['rows'][0]['metrics']['overall']['final_file_f1'], 1.0)

    def test_retry_journal_costs_are_reported_once_with_or_without_selection(self):
        self.run_arm('openai', failure_ids=['s2'])
        failed = self.saved()['s2']
        self.run_arm('openai')
        rows = self.saved()
        self.write(self.root/'journal.jsonl', [rows['s1'], failed, rows['s2']])
        keep = self.root/'keep.jsonl'
        self.write(keep, [{'sample_id': 's1'}, {'sample_id': 's2'}])
        for selection in (None, keep):
            report = evaluator.report_closed_tool_budget_curve(
                details_path=self.root/'journal.jsonl', out_path=self.root/'curve.json',
                markdown_out_path=self.root/'curve.md', budgets=[16], keep_list=selection,
            )
            self.assertEqual(report['rows'][0]['evaluated'], 2)
            self.assertEqual(report['rows'][0]['metrics']['overall']['final_file_f1'], 1.0)
            costs = report['observed_run_costs']['overall']
            self.assertEqual(costs['mean_model_input_tokens'], 7)
            self.assertEqual(costs['prior_attempt_model_input_tokens'], 7)
            self.assertEqual(costs['prior_attempts'], 1)
        self.write(keep, [{'sample_id': 's2'}, {'sample_id': 'missing'}])
        report = evaluator.report_closed_tool_budget_curve(
            details_path=self.root/'journal.jsonl', out_path=self.root/'curve.json',
            markdown_out_path=self.root/'curve.md', budgets=[16], keep_list=keep,
        )
        self.assertEqual(report['missing_selected'], ['missing'])
        self.assertEqual(report['rows'][0]['evaluated'], 2)
        self.assertEqual(report['rows'][0]['metrics']['overall']['final_file_f1'], 0.5)

    def test_failed_preparation_does_not_hide_successful_cold_build(self):
        row = dict(repo='example/repo', base_commit='base', closed_tool={
            'prior_attempts': [{'index_status': {'available': False, 'prepare_seconds': 0.5}}],
            'index_status': {'available': True, 'cached': False, 'build_seconds': 2.5},
        })
        builds = evaluator.index_builds([row], {})
        self.assertEqual(len(builds), 1)
        self.assertTrue(builds[0]['available'])
        self.assertEqual(builds[0]['build_seconds'], 2.5)

    def test_policy_changes_reject_before_dispatch_or_output_mutation(self):
        for arm in ('grep', 'openai', 'codex'):
            changes = [dict(max_tool_calls=9), dict(max_read_tokens=100), dict(final_k=2)]
            if arm != 'grep':
                changes += [dict(model='another-model')]
            if arm == 'openai':
                changes += [dict(reasoning_effort='low'), dict(require_read_submit=True)]
            for change in changes:
                with self.subTest(arm=arm, change=change):
                    self.run_arm(arm, force=True)
                    paths = [self.root/name for name in ('summary.json', 'details.jsonl', 'report.md')]
                    before = [path.read_bytes() for path in paths]
                    self.calls.clear()
                    with self.assertRaisesRegex(ValueError, 'experiment'):
                        self.run_arm(arm, **change)
                    self.assertEqual(self.calls, [])
                    self.assertEqual([path.read_bytes() for path in paths], before)

    def test_changed_sample_or_corpus_rejects_before_any_new_sample_dispatch(self):
        for arm in ('grep', 'openai', 'codex'):
            for changed in ('sample', 'corpus'):
                with self.subTest(arm=arm, changed=changed):
                    self.write(self.sample_path, self.samples)
                    self.write(self.chunks, [dict(path='subject.py', kind='file', text='anchorapple')])
                    self.run_arm(arm, force=True)
                    selected = [dict(self.samples[0], id='new-sample'), self.samples[1]]
                    if changed == 'sample':
                        selected[1] = dict(selected[1], query={'pr_title': 'changed task'})
                    else:
                        self.write(self.chunks, [dict(path='subject.py', kind='file', text='changed corpus')])
                    self.write(self.sample_path, selected)
                    before = (self.root/'details.jsonl').read_bytes()
                    self.calls.clear()
                    with self.assertRaisesRegex(ValueError, 'Sample identity'):
                        self.run_arm(arm)
                    self.assertEqual(self.calls, [])
                    self.assertEqual((self.root/'details.jsonl').read_bytes(), before)

    def test_legacy_records_require_explicit_reset(self):
        for arm in ('grep', 'openai', 'codex'):
            with self.subTest(arm=arm):
                self.write(self.root/'details.jsonl', [{'sample_id': 's1'}])
                before = (self.root/'details.jsonl').read_bytes()
                self.calls.clear()
                with self.assertRaisesRegex(ValueError, 'legacy'):
                    self.run_arm(arm)
                self.assertEqual(self.calls, [])
                self.assertEqual((self.root/'details.jsonl').read_bytes(), before)
                self.run_arm(arm, force=True)
                self.assertEqual(self.calls, ['s1', 's2'])

    def test_endpoint_and_codex_config_changes_reject_before_dispatch(self):
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://one.invalid/v1'}):
            self.run_arm('openai', force=True)
        self.calls.clear()
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://two.invalid/v1'}):
            with self.assertRaisesRegex(ValueError, 'experiment'):
                self.run_arm('openai')
        self.assertEqual(self.calls, [])
        config = self.root/'codex-home'/'config.toml'
        config.parent.mkdir()
        config.write_text('model_provider="one"', encoding='utf-8')
        self.run_arm('codex', force=True)
        config.write_text('model_provider="two"', encoding='utf-8')
        before = (self.root/'details.jsonl').read_bytes()
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, 'experiment'):
            self.run_arm('codex')
        self.assertEqual(self.calls, [])
        self.assertEqual((self.root/'details.jsonl').read_bytes(), before)

    def test_changed_seed_file_selection_rejects_before_codex_dispatch(self):
        seeds = self.root/'seed.jsonl'
        self.write(seeds, [{'sample_id': 's1', 'top_files': ['subject.py']}])
        self.run_arm('codex', seed_details=seeds, max_seed_tokens=10)
        self.write(seeds, [{'sample_id': 's1', 'top_files': []}])
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, 'Sample identity'):
            self.run_arm('codex', seed_details=seeds, max_seed_tokens=10)
        self.assertEqual(self.calls, [])

    def test_failure_history_survives_reporting_change_and_successful_retry(self):
        for arm in ('openai', 'codex'):
            with self.subTest(arm=arm):
                first = {'retain_failures': True} if arm == 'openai' else {}
                later = {'retain_failures': False} if arm == 'openai' else {}
                self.run_arm(arm, force=True, failure_ids=['s2'], **first)
                self.run_arm(arm, failure_ids=['s2'], **later)
                failed = self.saved()['s2']['closed_tool']
                self.assertIn('failure', failed)
                self.assertEqual(len(failed['prior_attempts']), 1)
                self.assertEqual(failed['prior_attempts'][0]['model_usage']['input_tokens'], 7)
                self.run_arm(arm, **later)
                done = self.saved()['s2']['closed_tool']
                self.assertNotIn('failure', done)
                self.assertEqual(len(done['prior_attempts']), 2)
                self.calls.clear()
                self.run_arm(arm, **later)
                self.assertEqual(self.calls, [])

    def test_codex_failure_keeps_observed_exec_events(self):
        record = dict(turn=1, returncode=1, json_events=[{'type': 'error'}],
                      protocol_violations=['failed call'], stderr_excerpt='offline failure')
        client = types.SimpleNamespace(call_records=[record])
        partial = evaluator.failed_closed_tool_detail(
            self.samples[0], ['subject.py'], {'error': 'RuntimeError'}, policy='codex_closed_tool',
        )
        error = RuntimeError('offline failure')
        error.partial_detail = partial
        with patch.object(evaluator, 'CodexActionClient', return_value=client), patch.object(
            evaluator, 'run_closed_tool_llm_sample', side_effect=error,
        ):
            with self.assertRaises(RuntimeError):
                evaluator.run_closed_tool_codex_sample(
                    sample=self.samples[0], gold_files=['subject.py'], query_text='anchorapple',
                    files={'subject.py': 'anchorapple'}, model='offline', codex_bin=Path(sys.executable),
                    work_root=self.root, timeout_seconds=1, max_tool_calls=2, max_read_tokens=10,
                    max_read_tokens_per_file=10, final_k=1, grep_top_k=1, max_model_turns=1,
                )
        self.assertEqual(partial['closed_tool']['codex_exec_calls'], 1)
        self.assertEqual(partial['closed_tool']['codex_exec_returncodes'], [1])
        self.assertEqual(partial['closed_tool']['protocol_violations'], ['failed call'])
        self.assertEqual(len(evaluator.prior_attempts(partial)[0]['codex_exec_events']), 1)

    def test_index_tool_description_and_configuration_are_resume_identity(self):
        class Provider:
            def __init__(self, version='one'):
                self.version = version
                self.preparations = 0

            def describe(self):
                return {'version': self.version}

            def prepare(self, **kwargs):
                self.preparations += 1
                return {'available': False}

            def call(self, **kwargs):
                return []

        def tools(provider, names=('symbol_search',), **options):
            return IndexedTools(provider, names, **options)

        for change in ('version', 'tools', 'options', 'cap'):
            with self.subTest(change=change):
                original = tools(Provider(), provider_configuration={'setting': 'one'})
                self.run_arm('openai', force=True, mode='indexed_interactive', indexed_tools=original)
                provider = Provider('two' if change == 'version' else 'one')
                names = ('symbol_search', 'references') if change == 'tools' else ('symbol_search',)
                options = {'provider_configuration': {'setting': 'two' if change == 'options' else 'one'}}
                if change == 'cap':
                    options['max_results'] = 2
                changed = tools(provider, names, **options)
                before = (self.root/'details.jsonl').read_bytes()
                self.calls.clear()
                with self.assertRaisesRegex(ValueError, 'experiment'):
                    self.run_arm('openai', mode='indexed_interactive', indexed_tools=changed)
                self.assertEqual(provider.preparations, 0)
                self.assertEqual(self.calls, [])
                self.assertEqual((self.root/'details.jsonl').read_bytes(), before)
